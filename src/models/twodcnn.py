"""
cnn2d.py

2D CNN version of the TF model.
Structural features are stacked into a (batch, 1, num_features, tf_len) image.
A 2D CNN processes this image, then an MLP classifies the final embedding.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm

from models.base import BaseModel


class CNN2DTFModel(BaseModel):
    class CNN2DModule(nn.Module):
        def __init__(self, config, tf_len):
            super().__init__()
            self.config = config
            self.tf_len = tf_len

            # ---------------------------------------------------------
            # 1. Build feature list
            # ---------------------------------------------------------
            feature_names = list(config.pred_struct_features)
            if config.use_probs:
                feature_names.append("pwm_scores")

            self.feature_names = feature_names

            # Expected input shape:
            # (batch, 1, number of features, tf_len)
            #
            # This behaves like a grayscale image with height=num_features, width=tf_len.

            # ---------------------------------------------------------
            # 2. 2D CNN layers
            # ---------------------------------------------------------
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),  # reduce width tf_len/2
                nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),  # tf_len/2 -> tf_len/4
                nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((1, 1)),  # collapse spatial dims
            )

            # Output shape from CNN = (batch, 128, 1, 1)
            cnn_output_size = 128
            fc_input_size = cnn_output_size + 1

            # ---------------------------------------------------------
            # 3. Final classifier (MLP)
            # ---------------------------------------------------------
            self.fc = nn.Sequential(
                nn.Linear(fc_input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),  # output logits
            )

        def forward(self, batch):
            # ---------------------------------------------------------
            # Build the 2D input tensor
            # ---------------------------------------------------------
            # Each structural feature is (batch, tf_len)
            rows = []
            for name in self.feature_names:
                feat = batch["structure_features"][name]  # (batch, tf_len)
                rows.append(feat.unsqueeze(1))  # → (batch, 1, tf_len)

            # Concatenate rows along "height"
            # → (batch, number of features, tf_len)
            x = torch.cat(rows, dim=1)

            # Add channel dimension for 2D CNN
            # → (batch, 1, number of features, tf_len)
            x = x.unsqueeze(1)

            # ---------------------------------------------------------
            # Forward through CNN
            # ---------------------------------------------------------
            x = self.cnn(x)  # → (batch, 64, 1, 1)
            x = x.view(x.size(0), -1)  # → (batch, 64)

            # ---------------------------------------------------------
            # Add scalar score
            # ---------------------------------------------------------
            score = batch["interval"]["Log_Prob"].unsqueeze(1)  # (batch, 1)
            x = torch.cat([x, score], dim=1)

            # ---------------------------------------------------------
            # MLP classifier
            # ---------------------------------------------------------
            return self.fc(x)

    # ======================================================================
    # BaseModel Integration
    # ======================================================================
    def __init__(self, config, tf_len):
        super().__init__(config, tf_len)
        self.model = self.CNN2DModule(config, tf_len).to(
            device=self.config.device,
            dtype=self.config.dtype,
        )

    def _train(self, train_data, val_data):
        loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()

        step = 0
        for epoch in range(self.config.epochs):
            for batch in tqdm(loader):
                optimizer.zero_grad()

                logits = self.model(batch)
                labels = batch["label"].float().unsqueeze(1).to(logits.device)

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                preds = (torch.sigmoid(logits) >= 0.5).float()
                acc = (preds == labels).float().mean().item()

                mlflow.log_metrics(
                    {"train_loss": loss.item(), "train_acc": acc}, step=step
                )
                step += 1

            # then validate
            probs, labels = self._predict(val_data)
            val_preds = (probs >= 0.5).astype(float)
            val_acc = (val_preds == labels).mean()
            mlflow.log_metrics({"val_acc": val_acc}, step=epoch)

    def _save_model(self):
        self.model_uri = mlflow.pytorch.log_model(
            self.model, name=self.__class__.__name__
        ).model_uri

    def _load_model(self):
        self.model = mlflow.pytorch.load_model(
            model_uri=self.model_uri, map_location=self.config.device
        )

    def _predict(self, data):
        self.model.eval()
        loader = DataLoader(data, batch_size=len(data), shuffle=False)

        all_outputs = []
        labels = []

        with torch.no_grad():
            for batch in loader:
                logits = self.model(batch)
                probs = torch.sigmoid(logits).cpu()

                all_outputs.append(probs)
                labels.append(batch["label"].unsqueeze(1).cpu())

        return torch.cat(all_outputs).numpy(), torch.cat(labels).numpy()
