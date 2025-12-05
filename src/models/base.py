"""
base.py.

The base model class for TF binding site prediction models.
"""

import mlflow


class BaseModel:
    def __init__(self, config):
        self.config = config

    def train(self, data):
        """
        Training function wrapper that includes the MLFlow baseline.
        This function then calls _train for the actual training loop.
        """
        with mlflow.start_run():
            print(self.config.__dict__)
            mlflow.log_params(self.config.__dict__)
            self._train(data)

    def _train(self, data):
        raise NotImplementedError("Train method not implemented.")

    def predict(self, data):
        raise NotImplementedError("Predict method not implemented.")

    def evaluate(self, data):
        """
        Evaluation of the data should be the same across all models,
        relying on the predict function.
        """
        # TODO: implement evaluation metrics and logging via MLFlow
        exit()
