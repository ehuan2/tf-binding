from models.base import BaseModel
from dataloaders import dataset_to_scikit

import mlflow
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


class SVMModel(BaseModel):
    """
    SVM classifier version of the TFBS binding/unbinding model.
    Uses the same feature flattening as the MLPModel.
    """

    def __init__(self, config, tf_len: int):
        super().__init__(config, tf_len)

        if config.svm_kernel == 'linear':
            base = LinearSVC(C=config.svm_C, max_iter=10000)
            self.model = CalibratedClassifierCV(base)
        
        else: # non-linear kernel, much slower
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel=config.svm_kernel,
                    C=config.svm_C,
                    gamma=config.svm_gamma,
                    degree=config.svm_degree,
                    probability=True,
                )),
            ])

    def _train(self, dataset):
        # load all training data into one single array
        X, y = dataset_to_scikit(self.config, dataset)

        self.model.fit(X, y)

    def _predict(self, dataset):
        X, y = dataset_to_scikit(self.config, dataset)
        preds = self.model.predict_proba(X)[:, 1]
        return preds, y

    def _save_model(self):
        print(f"Saving model to MLFlow with name {self.__class__.__name__}")
        self.model_uri = mlflow.sklearn.log_model(
            self.model, name=self.__class__.__name__
        ).model_uri

    def _load_model(self):
        self.model = mlflow.sklearn.load_model(
            model_uri=self.model_uri
        )

