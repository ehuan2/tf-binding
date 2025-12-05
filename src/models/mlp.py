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

    class MLPModule(nn.Module):
        def __init__(self, tf_len, config):
            super().__init__()
            # block of encoders first!
            self.encoders = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(tf_len, config.mlp_hidden_size), nn.ReLU())
                    for _ in range(
                        len(config.pred_struct_features or []) + config.use_probs
                    )
                ]
            )

            # then we have the final classifier
            self.final_mlp = nn.Sequential(
                nn.Linear(len(self.encoders) * config.mlp_hidden_size + 1, 1),
                nn.Sigmoid(),  # last layer for binary classification
            )

        def forward(self, scores, structure_feats):
            embeds = []
            for idx, value in enumerate(structure_feats):
                encoder = self.encoders[idx]
                embeds.append(encoder(value))

            # concatenate over the features not the batch dimension
            combined_feat = torch.cat(embeds, dim=1)
            combined_feat = torch.cat([combined_feat, scores.unsqueeze(1)], dim=1)

            return self.final_mlp(combined_feat).squeeze()

    def __init__(self, config, tf_len: int):
        super().__init__(config)
        self.tf_len = tf_len
        self.model = self.MLPModule(tf_len, config).to(
            device=self.config.device, dtype=self.config.dtype
        )

    def _train(self, data):
        train_loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        step = 0
        for _ in range(self.config.num_epochs):
            for batch in tqdm(train_loader):
                optimizer.zero_grad()

                scores = batch["interval"]["Log_Prob"]
                structure_feats = batch["structure_features"].values()
                labels = batch["label"]

                outputs = self.model(scores, structure_feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                step += 1
                accuracy = ((outputs >= 0.5).float() == labels).float().mean().item()
                mlflow.log_metrics(
                    {"train_loss": loss.item(), "acc": accuracy}, step=step
                )

        mlflow.pytorch.log_model(self.model, name="encoders")

    def predict(self, data):
        test_loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=False)
        print("Predicting with MLPModel on data:", data)
        return ["prediction"] * len(data)
