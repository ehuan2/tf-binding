"""
base.py.

The base model class for TF binding site prediction models.
"""

import mlflow
import os


class MLFlowWrapper(mlflow.pyfunc.PythonModel):
    """
    A class that wraps around the model to be used with MLFlow.
    """

    def __init__(self, config, tf_len):
        super().__init__()
        self.config = config
        self.tf_len = tf_len

    def load_context(self, context):
        # here let's instantiate the model, and call its load function
        self.model = self.config.get_model_instance(self.tf_len)
        self.model._load_model()

    def predict(self, context, model_input):
        # implement the prediction logic here, first we need to transform
        # the model input into the right format

        # TODO: pandas dataframe to what we had before

        return self.model._predict(model_input)


class BaseModel:
    def __init__(self, config, tf_len):
        self.config = config
        self.tf_len = tf_len

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
                # skip the restart flag itself
                if key == "restart_train":
                    continue
                search_query += f'params.{key} = "{value}" AND '
            search_query += 'attributes.status = "FINISHED"'

            runs = mlflow.search_runs(filter_string=search_query)
            if len(runs) > 1:
                print(f"Warning: found multiple runs with the same config!")
            if len(runs) > 0:
                run_id = runs.iloc[0].run_id
                self.model_uri = f"runs:/{run_id}/{self.model_name}"
                print(
                    f"Model with same config found, loading from uri {self.model_uri}"
                )
                return

        # before we train, let's ensure that a few things are set
        assert hasattr(
            self, "model_name"
        ), "Model name not specified in the model class."
        assert hasattr(
            self, "pth_path"
        ), "Path to model file not specified in the model class."

        # otherwise let's train the new model
        with mlflow.start_run():
            mlflow.log_params(self.config.__dict__)
            self._train(data)

        # after training, we need to save the model
        self.save_model()

    def _train(self, data):
        raise NotImplementedError("Train method not implemented.")

    def _predict(self, data):
        raise NotImplementedError("Predict method not implemented.")

    def evaluate(self, data):
        """
        Evaluation of the data should be the same across all models,
        relying on the predict function.
        """

        # df = self.batch_to_pandas(data)

        # result = mlflow.models.evaluate(
        #     self.model_uri,
        #     df,
        #     targets='label',
        #     model_type="classifier",
        # )
        # print(result.metrics)

    def _load_model(self):
        raise NotImplementedError("Load model method not implemented.")

    def _save_model(self):
        raise NotImplementedError("Save model method not implemented.")

    def save_model(self):
        """
        Save the model to the specified artifact path.
        """
        print("Saving model...")
        self._save_model()
        self.model_uri = mlflow.pyfunc.log_model(
            python_model=MLFlowWrapper(self.config, self.tf_len),
            name=self.model_name,
            artifacts={self.model_name: self.pth_path},
        ).model_uri

        # clean up the local model file
        os.remove(self.pth_path)
