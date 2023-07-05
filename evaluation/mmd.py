"""
Based on GDSS code, EDP-GNN code, GRAN code and GraphRNN code (modified).
https://github.com/harryjo97/GDSS
https://github.com/lrjconan/GRAN
https://github.com/ermongroup/GraphScoreMatching
https://github.com/JiaxuanYou/graph-generation
"""

import concurrent.futures
from functools import partial

import numpy as np
import pyemd
from scipy.linalg import toeplitz
from sklearn.metrics.pairwise import pairwise_kernels
from eden.graph import vectorize


def pad_array(x, y):
    """
    Utility function to make two numpy array equal length by padding with zeros.
    @param x: numpy array
    @param y: numpy array
    @return: (x, y), two arrays padded with zeros to be equal length
    """
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x, y


def emd(x, y, distance_scaling=1.0):
    """
    Earth mover's distance (EMD) between two histograms.
    @param x: numpy array
    @param y: numpy array
    @param distance_scaling: scaling factor for EMD
    @return: scalar, EMD value
    """
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)  # diagonal-constant matrix
    distance_mat = d_mat / distance_scaling
    x, y = pad_array(x, y)

    emd_value = pyemd.emd(x, y, distance_mat)
    return np.abs(emd_value)


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """
    Gaussian kernel with squared distance in exponential term replaced by EMD.
    @param x: numpy array
    @param y: numpy array
    @param sigma: Gaussian kernel parameter
    @param distance_scaling: EMD kernel parameter
    @return: scalar, Gaussian EMD kernel value
    """
    emd_value = emd(x, y, distance_scaling)
    return np.exp(-emd_value * emd_value / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    """
    Gaussian kernel.
    @param x: numpy array
    @param y: numpy array
    @param sigma: Gaussian kernel parameter
    @return: scalar, Gaussian kernel value
    """
    x = x.astype(float)
    y = y.astype(float)
    x, y = pad_array(x, y)
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x, y, sigma=1.0):
    """
    Gaussian kernel with total variation distance.
    @param x: numpy array
    @param y: numpy array
    @param sigma: Gaussian kernel parameter
    @return: scalar, Gaussian TV kernel value
    """
    x = x.astype(float)
    y = y.astype(float)
    x, y = pad_array(x, y)

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    """
    Helper function for parallel computation of kernel.
    """
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    """
    Helper function for parallel computation of kernel.
    """
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """
    Discrepancy between two set of samples.
    @param samples1: list of numpy arrays
    @param samples2: list of numpy arrays
    @param kernel: kernel function
    @param is_parallel: whether to use parallel computation
    @param args: arguments for kernel function
    @param kwargs: keyword arguments for kernel function
    @return: scalar, discrepancy value
    """
    d = 0
    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dist in executor.map(kernel_parallel_worker,
                                     [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]):
                d += dist
    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """
    MMD between two set of samples.
    @param samples1: list of numpy arrays
    @param samples2: list of numpy arrays
    @param kernel: kernel function
    @param is_hist: whether the samples are histograms
    @param args: arguments for kernel function
    @param kwargs: keyword arguments for kernel function
    """
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / np.sum(s1) if np.sum(s1) != 0 else s1 for s1 in samples1]
        samples2 = [s2 / np.sum(s2) if np.sum(s2) != 0 else s2 for s2 in samples2]
    # print('===============================')
    # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
    # print('===============================')
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
        disc(samples2, samples2, kernel, *args, **kwargs) - \
        2 * disc(samples1, samples2, kernel, *args, **kwargs)


def compute_nspdk_mmd(samples1, samples2, n_jobs=None):
    """
    Compute MMD between two sets of samples using NSPDK kernel.
    Code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
    """
    def _kernel_compute(x, y=None, n_jobs_=None):
        x = vectorize(x, complexity=4, discrete=True)
        if y is not None:
            y = vectorize(y, complexity=4, discrete=True)
        return pairwise_kernels(x, y, metric='linear', n_jobs=n_jobs_)

    X = _kernel_compute(samples1, n_jobs_=n_jobs)
    Y = _kernel_compute(samples2, n_jobs_=n_jobs)
    Z = _kernel_compute(samples1, y=samples2, n_jobs_=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)


def test():
    s1 = np.array([0.2, 0.8])
    s2 = np.array([0.3, 0.7])
    samples1 = [s1, s2]

    s3 = np.array([0.25, 0.75])
    s4 = np.array([0.35, 0.65])
    samples2 = [s3, s4]

    s5 = np.array([0.8, 0.2])
    s6 = np.array([0.7, 0.3])
    samples3 = [s5, s6]

    print('between samples1 and samples2: ', compute_mmd(samples1, samples2, kernel=gaussian,
                                                         is_parallel=True, sigma=1.0))
    print('between samples1 and samples3: ', compute_mmd(samples1, samples3, kernel=gaussian,
                                                         is_parallel=True, sigma=1.0))
    print('between samples1 and samples2: ', compute_mmd(samples1, samples2, kernel=gaussian_tv,
                                                         is_parallel=True, sigma=1.0))
    print('between samples1 and samples3: ', compute_mmd(samples1, samples3, kernel=gaussian_tv,
                                                         is_parallel=True, sigma=1.0))


if __name__ == '__main__':
    test()
