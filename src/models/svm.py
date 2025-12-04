from models.base import BaseModel
from helpers import (
    TFColumns,
    one_hot_encode
)

from sklearn.svm import (
    SVC, LinearSVC
)

import numpy as np


class SVMModel(BaseModel):
    """
    SVM classifier using sequence one-hot + predicted structure features.
    Assumes each batch item is a dictionary from IntervalDataset.
    """

    def __init__(self, config):
        self.config = config
        
        # Choose kernel
        if config.svm_kernel == "linear":
            self.model = LinearSVC(C=config.svm_c)
        else:
            self.model = SVC(
                kernel=config.svm_kernel,
                C=config.svm_c,
                gamma=config.svm_gamma,
            )

    def _flatten_item(self, item):
        """
        Convert a single IntervalDataset dict into flattened input vectors (X,y) for SVM
        """
        # one-hot encode sequence
        seq = item['interval'][TFColumns.SEQ.value]
        seq_vec = one_hot_encode(seq).astype(np.float32).reshape(-1)

        struct_feats = []
        # convert tensors to numpy for svm
        for name, tensor_val in item["structure_features"].items():
            arr = tensor_val.detach().cpu().numpy().astype(np.float32)
            struct_feats.append(arr)

        if len(struct_feats) > 0:
            struct_vec = np.concatenate(struct_feats)
        else:
            struct_vec = np.array([], dtype=np.float32)

        # concat seq + struct feats into single input vector
        X = np.concatenate([seq_vec, struct_vec]).astype(np.float32)

        # label vector
        y = float(item["label"].item())

        return X, y

    def train(self, dataset):
        """
        dataset: IntervalDataset (NOT a DataLoader, since SVM is not batch-trained).
        """
        X_list = []
        y_list = []

        print("Preparing SVM training data...")

        for i in range(len(dataset)):
            X_i, y_i = self._flatten_item(dataset[i])
            X_list.append(X_i)
            y_list.append(y_i)

        X = np.vstack(X_list)     # SVM expects 2D (N, D)
        y = np.array(y_list)

        print(f"SVM training on shape: X={X.shape}, y={y.shape}")
        self.model.fit(X, y)

    def predict(self, dataset):
        preds = []

        for i in range(len(dataset)):
            X_i, _ = self._flatten_item(dataset[i])
            pred = self.model.predict([X_i])[0]
            preds.append(pred)

        return np.array(preds)

    def evaluate(self, dataset):
        preds = self.predict(dataset)
        trues = np.array([float(dataset[i]["label"].item()) for i in range(len(dataset))])

        accuracy = (preds == trues).mean()
        print(f"SVM Accuracy: {accuracy:.4f}")

        return {"accuracy": accuracy}
