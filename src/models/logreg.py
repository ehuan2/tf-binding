# models/logreg.py
from models.base import BaseModel
from dataloaders import dataset_to_scikit

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import mlflow


class LogisticRegressionModel(BaseModel):
    def __init__(self, config, tf_len: int):
        super().__init__(config, tf_len)
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000, penalty="l2", C=1.0, solver="lbfgs", n_jobs=-1
            ),
        )

    def _train(self, dataset):
        X, y = dataset_to_scikit(self.config, dataset)
        self.model.fit(X, y)

    def _predict(self, dataset):
        X, y = dataset_to_scikit(self.config, dataset)
        # Get prediction probabilities for the positive class
        preds = self.model.predict_proba(X)[:, 1]
        return preds, y

    def _save_model(self):
        # save the model with mlflow
        print(f"Saving model to MLFlow with name {self.__class__.__name__}")
        self.model_uri = mlflow.sklearn.log_model(
            self.model, name=self.__class__.__name__
        ).model_uri

    def _load_model(self):
        self.model = mlflow.sklearn.load_model(
            model_uri=self.model_uri,
        )
