"""
base.py.

The base model class for TF binding site prediction models.
"""


class BaseModel:
    def __init__(self):
        pass

    def train(self, data):
        raise NotImplementedError("Train method not implemented.")

    def predict(self, data):
        raise NotImplementedError("Predict method not implemented.")

    def evaluate(self, data):
        """
        Evaluation of the data should be the same across all models,
        relying on the predict function.
        """
