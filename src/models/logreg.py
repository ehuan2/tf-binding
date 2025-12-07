# models/logreg.py
from models.base import BaseModel
from helpers import TFColumns

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt


NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot_encode(seq: str) -> np.ndarray:
    """
    One-hot encode a DNA sequence.
    Returns a (4, L) array with rows [A, C, G, T].
    Ambiguous bases (e.g. N) get all zeros at that position.
    """
    seq = seq.upper()
    L = len(seq)
    arr = np.zeros((4, L), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = NUC_TO_IDX.get(base)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr

class LogisticRegressionModel:
    def __init__(self, config, tf_len: int):
        self.config = config
        self.tf_len = tf_len

        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                n_jobs=-1
            )
        )

    def _flatten_item(self, item):
        # --- sequence (optional for ablations) ---
        seq_vec = np.array([], dtype=np.float32)
        if getattr(self.config, "use_seq", True):
            seq = item["interval"][TFColumns.SEQ.value]
            seq_vec = one_hot_encode(seq).reshape(-1)  # (4*L,)

        # --- structure features (from DNAshape + PWM/etc.) ---
        struct_feats = []
        for _, tensor_val in item["structure_features"].items():
            arr = tensor_val.detach().cpu().numpy().astype(np.float32)
            struct_feats.append(arr)

        if struct_feats:
            struct_vec = np.concatenate(struct_feats)
        else:
            struct_vec = np.array([], dtype=np.float32)

        # --- concatenate only the non-empty parts ---
        parts = []
        if seq_vec.size > 0:
            parts.append(seq_vec)
        if struct_vec.size > 0:
            parts.append(struct_vec)

        if not parts:
            raise ValueError("Both sequence and structure features are disabled!")

        X = np.concatenate(parts).astype(np.float32)
        y = float(item["label"].item())
        return X, y


    def train(self, dataset):
        X_list, y_list = [], []
        for i in range(len(dataset)):
            X_i, y_i = self._flatten_item(dataset[i])
            X_list.append(X_i)
            y_list.append(y_i)

        X = np.vstack(X_list)
        y = np.array(y_list)
        print(f"LogReg training on shape: X={X.shape}, y={y.shape}")
        self.model.fit(X, y)

    def predict(self, dataset):
        preds = []
        for i in range(len(dataset)):
            X_i, _ = self._flatten_item(dataset[i])
            pred = self.model.predict([X_i])[0]
            preds.append(pred)
        return np.array(preds)

    def evaluate(self, dataset, out_prefix: str = "logreg_pax5"):
        """
        Evaluate the model on a dataset and:
        - print accuracy + classification report
        - save confusion matrix and ROC curve as PNGs
        """
        # Build X, y for the test set
        X, y_true = self._get_X_y(dataset)

        # Class predictions and probabilities
        y_pred = self.model.predict(X)
        # predict_proba gives probabilities for both classes; [:, 1] is P(class=1)
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X)[:, 1]
        else:
            # fallback: use decision_function and squash, if needed
            y_proba = None

        # --- Basic metrics ---
        accuracy = (y_pred == y_true).mean()
        print(f"Logistic Regression Accuracy: {accuracy:.4f}")
        print("\nClassification report:\n")
        print(classification_report(y_true, y_pred, digits=3))

        # --- Confusion matrix plot ---
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm)
        ax.set_title("LogReg Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])

        # add numbers on cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                )

        fig.tight_layout()
        fig.savefig(f"{out_prefix}_confusion_matrix.png", dpi=300)
        plt.close(fig)

        # --- ROC curve plot (if we have probabilities) ---
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
            ax2.plot([0, 1], [0, 1], linestyle="--", label="Random")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("LogReg ROC Curve")
            ax2.legend(loc="lower right")

            fig2.tight_layout()
            fig2.savefig(f"{out_prefix}_roc_curve.png", dpi=300)
            plt.close(fig2)

        return {
            "accuracy": float(accuracy),
            "auc": float(roc_auc),
        }

    
    def _get_X_y(self, dataset):
        """
        Flatten all items in the dataset into X, y arrays.
        Used for both training and evaluation with plots.
        """
        X_list, y_list = [], []
        for i in range(len(dataset)):
            X_i, y_i = self._flatten_item(dataset[i])
            X_list.append(X_i)
            y_list.append(y_i)

        X = np.vstack(X_list)
        y = np.array(y_list)
        return X, y
    def get_arrays(self, dataset):
        """
        Public helper: returns (X, y) for MLflow evaluation.
        """
        return self._get_X_y(dataset)
