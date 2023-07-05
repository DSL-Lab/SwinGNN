class TrainingObjectiveGenerator(object):
    """
    Template for training data objective generator.
    """
    def __init__(self,
                 objective,
                 dev,
                 **kwargs):
        super().__init__()

        self.objective = objective
        assert objective in ['diffusion', 'score', 'edm']

        self.dev = dev

    def get_network_input(self, **kwargs):
        pass

    def get_network_target(self, **kwargs):
        pass

    def get_conditions(self, **kwargs):
        pass

    def get_input_output(self, **kwargs):
        return self.get_network_input(), self.get_conditions(), self.get_network_target()
