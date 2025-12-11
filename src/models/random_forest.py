# models/random_forest.py
from models.base import BaseModel
from dataloaders import dataset_to_scikit

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import mlflow


class RandomForestModel(BaseModel):
    def __init__(self, config, tf_len: int):
        super().__init__(config, tf_len)
        self.model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1),
        )

    def _train(self, train_dataset, val_dataset):
        X, y = dataset_to_scikit(self.config, train_dataset)
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
