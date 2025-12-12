from models.base import BaseModel
import mlflow

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm


class MLPModel(BaseModel):
    """
    Simple MLP model that takes in the position weight matrix, the different DNA
    structural features, and combines them to classify whether a region is a TF
    binding site or not.
    """

    class MLPModule(nn.Module):
        def __init__(self, config, tf_len):
            super().__init__()
            # block of encoders first!
            self.encoders = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(tf_len, config.mlp_hidden_size),
                        nn.ReLU(),
                        # nn.Dropout(0.1),
                        nn.Linear(config.mlp_hidden_size, config.mlp_hidden_size),
                        nn.ReLU(),
                        # nn.Dropout(0.1),
                        nn.Linear(config.mlp_hidden_size, config.mlp_hidden_size),
                    )
                    for _ in range(len(config.pred_struct_features or []))
                ]
            )
            self.encoders.append(
                nn.Sequential(
                    nn.Linear(
                        tf_len - 2 * config.context_window, config.mlp_hidden_size
                    ),
                    nn.ReLU(),
                    # nn.Dropout(0.1),
                    nn.Linear(config.mlp_hidden_size, config.mlp_hidden_size),
                    nn.ReLU(),
                    # nn.Dropout(0.1),
                    nn.Linear(config.mlp_hidden_size, config.mlp_hidden_size),
                )
            )
            self.config = config

            # we do the total encoders + 1 for the score
            total_hidden_size = len(self.encoders) * config.mlp_hidden_size + 1

            # then we have the final classifier
            if total_hidden_size > 1:
                self.final_mlp = nn.Sequential(
                    nn.Linear(total_hidden_size, total_hidden_size),
                    nn.ReLU(),
                    # nn.Dropout(0.1),
                    nn.Linear(total_hidden_size, total_hidden_size),
                    nn.ReLU(),
                    # nn.Dropout(0.1),
                    nn.Linear(total_hidden_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )
            else:
                self.final_mlp = nn.Sequential(
                    # last layer for binary classification
                    nn.Linear(total_hidden_size, 1),
                )

        def forward(self, batch):
            scores = batch["interval"]["Log_Prob"]

            # first let's normalize the features
            scores = scores.unsqueeze(1)

            embeds = []
            for idx, key in enumerate(self.config.pred_struct_features):
                encoder = self.encoders[idx]
                embeds.append(encoder(batch["structure_features"][key]))

            if self.config.use_probs:
                embeds.append(
                    self.encoders[-1](batch["structure_features"]["pwm_scores"])
                )

            # concatenate over the features not the batch dimension
            if len(embeds) > 0:
                combined_feat = torch.cat(embeds, dim=1)
                combined_feat = torch.cat([combined_feat, scores], dim=1)
            else:
                combined_feat = scores

            return self.final_mlp(combined_feat)

    def __init__(self, config, tf_len: int):
        super().__init__(config, tf_len)
        self.model = self.MLPModule(config, tf_len).to(
            device=self.config.device, dtype=self.config.dtype
        )

    def _train(self, train_data, val_data):
        train_loader = DataLoader(
            train_data, batch_size=self.config.batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()

        step = 0
        for epoch in range(self.config.epochs):
            # first train
            self.model.train()
            for batch in tqdm(train_loader):
                optimizer.zero_grad()

                outputs = self.model(batch)

                labels = batch["label"].unsqueeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                step += 1
                accuracy = ((outputs >= 0.5).float() == labels).float().mean().item()
                total_pred_true = (outputs >= 0.5).float().sum()
                mlflow.log_metrics(
                    {
                        "train_loss": loss.item(),
                        "train_acc": accuracy,
                        "total_pred_true": total_pred_true.item(),
                    },
                    step=step,
                )

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
            model_uri=self.model_uri,
            map_location=self.config.device,
        )

    def _predict(self, data):
        self.model.eval()

        data_loader = DataLoader(data, batch_size=len(data), shuffle=False)
        all_outputs = []
        labels = []

        with torch.no_grad():
            for batch in data_loader:
                outputs = self.model(batch)
                all_outputs.append(torch.sigmoid(outputs).cpu())
                labels.append(batch["label"].unsqueeze(1).cpu())

        return torch.cat(all_outputs).numpy(), torch.cat(labels).numpy()
