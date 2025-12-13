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
    class CNNWindowModule(nn.Module):
        def __init__(self, config, tf_len):
            super().__init__()
            self.config = config
            self.tf_len = tf_len

            # -----------------------------
            # 1. Separate feature categories
            # -----------------------------
            self.struct_feature_names = list(config.pred_struct_features)

            # PWM gets its own separate branch if used
            self.use_pwm = config.use_probs

            # Number of structural feature channels
            num_struct_channels = len(self.struct_feature_names)

            # -----------------------------
            # 2. Structural CNN branch
            # -----------------------------
            self.struct_cnn = nn.Sequential(
                nn.Conv1d(num_struct_channels, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),  # /2
                nn.Conv1d(32, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),  # /4
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),  # → (batch, 256, 1)
            )

            # -----------------------------
            # 3. PWM CNN branch (separate)
            # -----------------------------
            if self.use_pwm:
                # PWM has exactly 1 channel: (batch, 1, tf_len)
                self.pwm_cnn = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(16, 64, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),  # → (batch,128,1)
                )
                pwm_out_dim = 128
            else:
                pwm_out_dim = 0

            # -----------------------------
            # 4. Final FC classifier
            # -----------------------------
            struct_out_dim = 256
            fc_input = struct_out_dim + pwm_out_dim + 1  # + Log_Prob scalar

            self.fc = nn.Sequential(
                nn.Linear(fc_input, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )

        def forward(self, batch):
            # -----------------------------
            # Structural features → CNN
            # -----------------------------
            struct_maps = []
            for name in self.struct_feature_names:
                x = batch["structure_features"][name]  # (batch, tf_len)
                struct_maps.append(x.unsqueeze(1))

            struct_x = torch.cat(struct_maps, dim=1)  # (batch, C_struct, L)
            struct_feat = self.struct_cnn(struct_x).squeeze(-1)  # (batch,256)

            # -----------------------------
            # PWM branch (optional)
            # -----------------------------
            if self.use_pwm:
                pwm = batch["structure_features"]["pwm_scores"]  # (batch, tf_len)
                pwm = pwm.unsqueeze(1)  # (batch,1,tf_len)
                pwm_feat = self.pwm_cnn(pwm).squeeze(-1)  # (batch,128)
            else:
                pwm_feat = None

            # -----------------------------
            # Log prob
            # -----------------------------
            score = batch["interval"]["Log_Prob"].unsqueeze(1)

            # -----------------------------
            # Concatenate all
            # -----------------------------
            if pwm_feat is not None:
                feats = torch.cat([struct_feat, pwm_feat, score], dim=1)
            else:
                feats = torch.cat([struct_feat, score], dim=1)

            return self.fc(feats)

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
                nn.Conv1d(32, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),  # tf_len/2 -> tf_len/4
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),  # reduce each channel to scalar
            )

            # Output of cnn is (batch, 256, 1) → flatten to (batch, 256)
            fc_input_size = 256

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
        if config.context_window > 0:
            self.model = self.CNNWindowModule(config, tf_len).to(
                device=self.config.device, dtype=self.config.dtype
            )
        else:
            self.model = self.CNNModule(config, tf_len).to(
                device=self.config.device, dtype=self.config.dtype
            )

    def _train(self, train_data, val_data):
        loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-5
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(2.0).to(self.config.device)
        )

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
