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

        # check if a model with the same config already exists, and we
        # don't specify the restart flag
        if not self.config.restart_train:
            search_query = ""
            for key, value in self.config.__dict__.items():
                search_query += f'params.{key} = "{value}" AND '
            search_query += 'attributes.status = "FINISHED"'

            runs = mlflow.search_runs(filter_string=search_query)
            if len(runs) > 1:
                print(f"Warning: found multiple runs with the same config!")
            if len(runs) > 0:
                run_id = runs.iloc[0].run_id
                self._load_model(run_id)
                print(f"Model with same config found, loading the model...")
                return

        # otherwise let's train the new model
        with mlflow.start_run():
            mlflow.log_params(self.config.__dict__)
            self._train(data)

    def _train(self, data):
        raise NotImplementedError("Train method not implemented.")

    def _predict(self, data):
        raise NotImplementedError("Predict method not implemented.")

    def evaluate(self, data):
        """
        Evaluation of the data should be the same across all models,
        relying on the predict function.
        """
        # TODO: implement evaluation metrics and logging via MLFlow
        exit()

    def _load_model(self, run_id):
        raise NotImplementedError("Load model method not implemented.")
