"""
cnn.py

Here we will attempt to build a 1D CNN features for each structural feature.
Then we will concatenate these features together into a simple MLP.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm

from models.base import BaseModel


class CNNTFModel(BaseModel):
    class CNNModule(nn.Module):
        def __init__(self, config, tf_len):
            super().__init__()
            self.config = config
            self.tf_len = tf_len

            # 1. Build feature list from config + optional pwm scores
            feature_names = list(config.pred_struct_features)
            if config.use_probs:
                feature_names.append("pwm_scores")

            self.feature_names = feature_names
            num_channels = len(feature_names)

            # 2. CNN layers
            # Input shape: (batch, num_channels, tf_len)
            self.cnn = nn.Sequential(
                nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),  # tf_len -> tf_len/2
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),  # tf_len/2 -> tf_len/4
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),  # reduce each channel to scalar
            )

            # Output of cnn is (batch, 128, 1) → flatten to (batch, 128)
            fc_input_size = 128

            # If including Log_Prob score explicitly (scalar)
            fc_input_size += 1

            # 3. Final classifier
            self.fc = nn.Sequential(
                nn.Linear(fc_input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),  # logits!
            )

        def forward(self, batch):
            # --------------------------------------------------------
            # Retrieve structural features and stack into 1D CNN input
            # Each feature must be shape (batch, tf_len)
            # --------------------------------------------------------
            feature_maps = []

            for name in self.feature_names:
                feat = batch["structure_features"][name]  # (batch, tf_len)
                feature_maps.append(feat.unsqueeze(1))  # → (batch, 1, tf_len)

            # Stack → (batch, num_features, tf_len)
            x = torch.cat(feature_maps, dim=1)

            # CNN forward
            x = self.cnn(x)  # → (batch, 128, 1)
            x = x.squeeze(-1)  # → (batch, 128)

            score = batch["interval"]["Log_Prob"].unsqueeze(1)  # (batch,1)
            x = torch.cat([x, score], dim=1)

            # Final linear classifier → logits
            return self.fc(x)

    # BaseModel integration
    def __init__(self, config, tf_len):
        super().__init__(config, tf_len)
        self.model = self.CNNModule(config, tf_len).to(
            device=self.config.device, dtype=self.config.dtype
        )

    def _train(self, data):
        loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=1e-4
        )
        criterion = nn.BCEWithLogitsLoss()

        step = 0
        for _ in range(self.config.epochs):
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
