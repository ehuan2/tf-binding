"""
base.py.

The base model class for TF binding site prediction models.
"""

import mlflow
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt


class BaseModel:
    def __init__(self, config, tf_len):
        self.config = config
        self.tf_len = tf_len

    def train_and_eval(self, train_data, eval_data):
        """
        Combined training and evaluation function.
        """
        with mlflow.start_run():
            self.train(train_data)
            self.evaluate(eval_data)

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

        # otherwise let's train the new model
        mlflow.log_params(self.config.__dict__)
        self._train(data)

        # after training, we need to save the model
        self._save_model()

    def _train(self, data):
        raise NotImplementedError("Train method not implemented.")

    def _predict(self, data_loader):
        raise NotImplementedError("Predict method not implemented.")

    def evaluate(self, data):
        """
        Evaluation of the data should be the same across all models,
        relying on the predict function.
        """
        print(f"Evaluating model from {self.model_uri}")
        self._load_model()
        data_loader = DataLoader(data, batch_size=len(data), shuffle=False)

        scores = self._predict(data_loader)
        labels = [batch for batch in data_loader][0]["label"].cpu().numpy()
        predictions = (scores >= 0.5).astype(float)

        assert (
            labels.shape == predictions.shape
        ), "Labels and predictions shape mismatch"

        accuracy = (predictions == labels).mean()
        print(f"Evaluation accuracy: {accuracy}")

        # now we log the metrics to mlflow
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        mlflow.log_metrics(
            {
                "eval_accuracy": accuracy,
                "eval_roc_auc": roc_auc,
                "eval_pr_auc": pr_auc,
            }
        )

        # now we plot the ROC and PR curves
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        plt.close()

        mlflow.log_artifact("roc_curve.png")
        os.remove("roc_curve.png")

    def _load_model(self):
        raise NotImplementedError("Load model method not implemented.")

    def _save_model(self):
        raise NotImplementedError("Save model method not implemented.")
