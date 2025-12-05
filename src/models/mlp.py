from models.base import BaseModel
from torch.utils.data import DataLoader
import mlflow

import torch
import torch.nn as nn

from tqdm import tqdm


class MLPModel(BaseModel):
    """
    Simple MLP model that takes in the position weight matrix, the different DNA
    structural features, and combines them to classify whether a region is a TF
    binding site or not.
    """

    def __init__(self, config, tf_len: int):
        super().__init__(config)
        self.tf_len = tf_len

        # TODO: set the following hyperparameters via config
        self.config.hidden_size = 16
        self.config.device = torch.device("cpu")
        self.config.dtype = torch.float64
        self.config.num_epochs = 1

        # block of encoders first!
        self.encoders = [
            self._cast_obj(
                nn.Sequential(nn.Linear(tf_len, self.config.hidden_size), nn.ReLU())
            )
            for _ in range(len(config.pred_struct_features or []) + config.use_probs)
        ]

        # then we have the final classifier
        self.model = self._cast_obj(
            nn.Sequential(
                nn.Linear(len(self.encoders) * self.config.hidden_size + 1, 1),
                nn.Sigmoid(),  # last layer for binary classification
            )
        )

    def _cast_obj(self, obj):
        return obj.to(device=self.config.device, dtype=self.config.dtype)

    def _train(self, data):
        train_loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        step = 0
        for _ in range(self.config.num_epochs):
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                scores, structure_feat, labels = (
                    batch["interval"]["Log_Prob"],
                    batch["structure_features"],
                    batch["label"],
                )

                embeds = []
                for idx, value in enumerate(structure_feat.values()):
                    value = self._cast_obj(value)
                    encoder = self.encoders[idx]
                    embeds.append(encoder(value))

                # concatenate over the features not the batch dimension
                combined_feat = torch.cat(embeds, dim=1)
                combined_feat = torch.cat([combined_feat, scores.unsqueeze(1)], dim=1)
                combined_feat = self._cast_obj(combined_feat)

                outputs = self.model(combined_feat).squeeze()

                labels = self._cast_obj(labels)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                step += 1
                accuracy = ((outputs >= 0.5).float() == labels).float().mean().item()
                mlflow.log_metrics(
                    {"train_loss": loss.item(), "acc": accuracy}, step=step
                )

        mlflow.pytorch.log_model(self.model, "mlp_model")

    def predict(self, data):
        test_loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=False)
        print("Predicting with MLPModel on data:", data)
        return ["prediction"] * len(data)
