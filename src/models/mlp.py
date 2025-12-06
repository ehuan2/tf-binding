from models.base import BaseModel
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
        def __init__(self, config, tf_len):
            super().__init__()
            # block of encoders first!
            self.encoders = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(tf_len, config.mlp_hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                    )
                    for _ in range(
                        len(config.pred_struct_features or []) + config.use_probs
                    )
                ]
            )

            total_hidden_size = len(self.encoders) * config.mlp_hidden_size + 1

            # then we have the final classifier
            self.final_mlp = nn.Sequential(
                nn.Linear(total_hidden_size, total_hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(total_hidden_size // 2, 1),
                nn.Sigmoid(),  # last layer for binary classification
            )

        def forward(self, batch):
            scores = batch["interval"]["Log_Prob"]
            structure_feats = batch["structure_features"].values()

            embeds = []
            for idx, value in enumerate(structure_feats):
                encoder = self.encoders[idx]
                embeds.append(encoder(value))

            # concatenate over the features not the batch dimension
            combined_feat = torch.cat(embeds, dim=1)
            combined_feat = torch.cat([combined_feat, scores.unsqueeze(1)], dim=1)

            return self.final_mlp(combined_feat).squeeze()

    def __init__(self, config, tf_len: int):
        super().__init__(config, tf_len)
        self.tf_len = tf_len
        self.model = self.MLPModule(config, tf_len).to(
            device=self.config.device, dtype=self.config.dtype
        )
        self.model_name = "MLPModel"
        self.pth_path = "mlp_model.pth"

    def _train(self, data):
        train_loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=0.01
        )
        criterion = nn.BCELoss()

        step = 0
        for _ in range(self.config.epochs):
            for batch in tqdm(train_loader):
                optimizer.zero_grad()

                outputs = self.model(batch)

                labels = batch["label"]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                step += 1
                accuracy = ((outputs >= 0.5).float() == labels).float().mean().item()
                mlflow.log_metrics(
                    {"train_loss": loss.item(), "train_acc": accuracy}, step=step
                )
                break

    def _save_model(self):
        self.model_uri = mlflow.pytorch.log_model(
            self.model, name=self.model_name
        ).model_uri

    def _load_model(self):
        self.model = mlflow.pytorch.load_model(
            model_uri=self.model_uri,
            map_location=self.config.device,
        )

    def _predict(self, data_loader):
        self.model.eval()

        all_outputs = []

        with torch.no_grad():
            for batch in data_loader:
                outputs = self.model(batch)
                pred_labels = (outputs >= 0.5).float()
                accuracy = ((pred_labels) == batch["label"]).float().mean().item()
                print(f"Batch accuracy: {accuracy}")
                all_outputs.append(outputs.cpu())

        return torch.cat(all_outputs).numpy()
