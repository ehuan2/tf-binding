from models.base import BaseModel
import mlflow

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# -------------------------------------------------------------
#   Stage 1: VAE Module
# -------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim

        # Encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# -------------------------------------------------------------
#   Stage 2: Classifier From Latent Space
# -------------------------------------------------------------
class LatentClassifier(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.model(z)


# Now we need to integrate both into a full model
class VAEClassifier(nn.Module):
    def __init__(self, vae: VAE, classifier: LatentClassifier):
        super().__init__()
        self.vae = vae
        self.classifier = classifier


# -------------------------------------------------------------
#   Full Two-Stage Model
# -------------------------------------------------------------
class VAEModel(BaseModel):
    def __init__(self, config, tf_len):
        super().__init__(config, tf_len)

        # Build feature list exactly like your MLP:
        feature_names = list(config.pred_struct_features)
        if config.use_probs:
            feature_names.append("pwm_scores")

        self.feature_names = feature_names
        self.num_features = len(feature_names)

        # Input = all structural features concatenated - 2 * the context window, because use_probs does not include it
        input_dim = self.num_features * tf_len - 2 * config.context_window

        latent_dim = config.vae_latent_dim
        self.model = VAEClassifier(
            vae=VAE(input_dim, latent_dim),
            classifier=LatentClassifier(latent_dim),
        ).to(self.config.device, dtype=self.config.dtype)

    # ---------------------------------------------------------
    # Utility: turn a batch of dict features → flat vector
    # ---------------------------------------------------------
    def _flatten_features(self, batch):
        feats = []
        for name in self.feature_names:
            f = batch["structure_features"][name]  # (batch, tf_len)
            feats.append(f)
        x = torch.cat(feats, dim=1)  # (batch, F * tf_len)
        return x

    # ---------------------------------------------------------
    # Stage 1: Train VAE
    # ---------------------------------------------------------
    def _train_vae(self, train_loader):
        optimizer = torch.optim.Adam(self.model.vae.parameters(), lr=1e-3)

        step = 0
        for epoch in range(self.config.vae_epochs):
            for batch in tqdm(train_loader):
                x = self._flatten_features(batch).to(self.config.device)

                recon, mu, logvar = self.model.vae(x)
                recon_loss = F.mse_loss(recon, x, reduction="mean")
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mlflow.log_metrics(
                    {
                        "vae_recon": recon_loss.item(),
                        "vae_kl": kl.item(),
                        "vae_loss": loss.item(),
                    },
                    step=step,
                )

                step += 1

        # freeze VAE after training
        for p in self.model.vae.parameters():
            p.requires_grad = False

    # ---------------------------------------------------------
    # Stage 2: Train classifier on frozen embeddings
    # ---------------------------------------------------------
    def _train_classifier(self, train_loader, val_data):
        optimizer = torch.optim.AdamW(
            self.model.classifier.parameters(), lr=3e-4, weight_decay=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()

        step = 0
        for epoch in range(self.config.epochs):
            self.model.classifier.train()

            for batch in tqdm(train_loader):
                x = self._flatten_features(batch).to(self.config.device)
                labels = batch["label"].float().unsqueeze(1).to(self.config.device)

                # VAE encoder only:
                with torch.no_grad():
                    mu, logvar = self.model.vae.encode(x)

                logits = self.model.classifier(mu)

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (
                    ((torch.sigmoid(logits) >= 0.5).float() == labels)
                    .float()
                    .mean()
                    .item()
                )

                mlflow.log_metrics(
                    {"train_loss": loss.item(), "train_acc": acc},
                    step=step,
                )
                step += 1

            # Validation
            probs, y = self._predict(val_data)
            preds = (probs >= 0.5).astype(float)
            val_acc = (preds == y).mean()

            mlflow.log_metrics({"val_acc": val_acc}, step=epoch)

    # ---------------------------------------------------------
    # Public train() interface
    # ---------------------------------------------------------
    def _train(self, train_data, val_data):
        train_loader = DataLoader(
            train_data, batch_size=self.config.batch_size, shuffle=True
        )

        # Stage 1: VAE
        print("Training VAE...")
        self._train_vae(train_loader)

        # Stage 2: Classifier
        print("Training classifier...")
        self._train_classifier(train_loader, val_data)

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    def _predict(self, data):
        self.model.classifier.eval()
        loader = DataLoader(data, batch_size=len(data), shuffle=False)

        all_probs = []
        all_y = []

        with torch.no_grad():
            for batch in loader:
                x = self._flatten_features(batch).to(self.config.device)
                mu, _ = self.model.vae.encode(x)
                logits = self.model.classifier(mu)

                all_probs.append(torch.sigmoid(logits).cpu())
                all_y.append(batch["label"].unsqueeze(1).cpu())

        return torch.cat(all_probs).numpy(), torch.cat(all_y).numpy()

    # ---------------------------------------------------------
    # Saving/loading
    # ---------------------------------------------------------
    def _save_model(self):
        self.model_uri = mlflow.pytorch.log_model(
            self.model, name=self.__class__.__name__
        ).model_uri

    def _load_model(self):
        self.model = mlflow.pytorch.load_model(
            self.model_uri, map_location=self.config.device
        )
