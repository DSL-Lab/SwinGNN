# SwinGNN: Rethinking Permutation Invariance in Diffusion Models for Graph Generation
This repository contains the official implementation of the SwinGNN model in PyTorch.
## Get started
### Install dependencies
```bash
# option 1: python venv
python -m venv venvscorenet
source venvscorenet/bin/activate
pip install -U pip
pip install -r setup/requirements.txt

# option 2: conda
conda env create -f setup/conda.yaml
conda activate scorenet

# compile ORCA for orbit statistics evaluation
export PROJ_DIR=$(pwd)
cd evaluation/orca && g++ -O2 -std=c++11 -o orca orca.cpp && cd $PROJ_DIR
```
### Setup datasets
```bash
# prepare datasets
python setup/gen_graph_data.py  # prepare various synthetic and real-world graph datasets
python setup/mol_preprocess.py --dataset ZINC250k  # prepare ZINC250k dataset
python setup/mol_preprocess.py --dataset QM9  # prepare QM9 dataset
```


## Training command
Below we provide the training commands for SwinGNN on graph datasets and molecule datasets.
Please refer to `/config/edm_swin_gnn` for more training configurations.
```bash
# training cmds on graph dataset (without node/edge attributes), e.g., to train on grid dataset
python train.py -c config/edm_swin_gnn/grid_edm_swin_gnn_80.yaml --batch_size 10 -m=grid

# our code also supports DDP training
export NUM_GPUS=4
torchrun --nproc_per_node=$NUM_GPUS train.py -c config/edm_swin_gnn/grid_edm_swin_gnn_80.yaml --batch_size 40 --ddp -m=grid_ddp

# training cmds on molecule dataset (with node/edge attributes), e.g., to train on QM9 dataset
torchrun --nproc_per_node=$NUM_GPUS train.py -c config/edm_swin_gnn/qm9_edm_swin_gnn.yaml --feature_dims 60 --node_encoding one_hot --edge_encoding one_hot --batch_size 10240 --ddp -m qm9
```

## Testing command
We release the checkpoints at [Google Drive](https://drive.google.com/drive/folders/1qCHD6c0Fr5Dymo8qru8UakwIwpHmU9eA?usp=sharing) and [oneDrive](https://1drv.ms/f/s!AnkbqTET-eNqgoYjheispuweUMWkxA?e=jfZ7UO).
Below we provide the sampling commands for SwinGNN on graph datasets and molecule datasets.
```bash
# ego-small
python eval.py -p swinGNN-checkpoints/ego_small/ego_small_dim_60/ego_small_dim_60.pth --use_ema 0.9 -m eval_ego_small 
python eval.py -p swinGNN-checkpoints/ego_small/ego_small_dim_96/ego_small_dim_96.pth --use_ema 0.99 -m eval_ego_small 

# community-small
python eval.py -p swinGNN-checkpoints/com_small/community_small_dim_60/community_small_dim_60.pth --use_ema 0.99 -m eval_com_small 
python eval.py -p swinGNN-checkpoints/com_small/community_small_dim_96/community_small_dim_96.pth --use_ema 0.95 -m eval_com_small

# grid
python eval.py -p swinGNN-checkpoints/grid/grid_dim_60/grid_dim_60.pth --use_ema 0.99 -m eval_grid 
python eval.py -p swinGNN-checkpoints/grid/grid_dim_96/grid_dim_96.pth --use_ema 0.95 -m eval_grid 

# dd-protein
python eval.py -p swinGNN-checkpoints/dd_protein/dd_dim_60/dd_dim_60.pth --use_ema 0.9999 -m eval_dd_protein
python eval.py -p swinGNN-checkpoints/dd_protein/dd_dim_96/dd_dim_96.pth --use_ema 0.9999 -m eval_dd_protein 

# qm9
python eval.py -p swinGNN-checkpoints/qm9/qm9_scalar_dim_60/qm9_scalar_dim_60.pth --use_ema 0.9999 -m eval_qm9
python eval.py -p swinGNN-checkpoints/qm9/qm9_scalar_dim_96/qm9_scalar_dim_96.pth --use_ema 0.9999 -m eval_qm9

# zinc250k
python eval.py -p swinGNN-checkpoints/zinc250k/zinc250k_scalar_dim_60/zinc250k_scalar_dim_60.pth --use_ema 0.9999 -m eval_zinc250k
python eval.py -p swinGNN-checkpoints/zinc250k/zinc250k_scalar_dim_96/zinc250k_scalar_dim_96.pth --use_ema 0.9999 -m eval_zinc250k
```
Due to the randomness in the sampling process or difference in hardware, the results may be slightly different from the reported results in the paper.