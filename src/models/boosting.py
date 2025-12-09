# models/xgboost.py
from models.base import BaseModel
from dataloaders import dataset_to_scikit

import xgboost as xgb
import mlflow


class BoostingModel(BaseModel):
    def __init__(self, config, tf_len: int):
        super().__init__(config, tf_len)
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
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
        self.model_uri = mlflow.xgboost.log_model(
            self.model, name=self.__class__.__name__
        ).model_uri

    def _load_model(self):
        self.model = mlflow.xgboost.load_model(
            model_uri=self.model_uri,
        )
