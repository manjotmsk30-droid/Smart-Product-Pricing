"""
Model 3: Image Head (CLIP/EfficientNet → Ridge/MLP)
Predicts price from image embeddings.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge


class ImageRegressionHead(nn.Module):
    """MLP head for image embeddings."""
    def __init__(self, input_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x).squeeze(-1)


class ImageDataset(Dataset):
    """Dataset for image embeddings."""
    def __init__(self, embeddings, targets):
        # no interface change; just explicit types
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


class ImageModelTrainer:
    """Trainer for image-based price prediction."""

    def __init__(
        self,
        model_type: str = "mlp",   # 'ridge' or 'mlp'
        embedding_dim: int = 512,
        device: str | None = None,
    ):
        """
        Args:
            model_type: 'ridge' for Ridge regression or 'mlp' for neural network
            embedding_dim: Dimension of input embeddings
            device: Device for training
        """
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "mlp":
            self.model = ImageRegressionHead(embedding_dim).to(self.device)
        elif model_type == "ridge":
            # CHANGED: removed unsupported random_state for Ridge when solver='auto'.
            # sklearn.linear_model.Ridge does not accept random_state unless solver is 'sag'/'saga'.
            # Passing it would raise "TypeError: __init__() got an unexpected keyword argument 'random_state'".
            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        print(f"✓ Initialized Image{model_type.upper()} on {self.device if model_type=='mlp' else 'CPU'}")

    def train(
        self,
        train_embeddings,
        train_targets,
        val_embeddings=None,
        val_targets=None,
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 1e-3,
    ):
        """
        Train image regression model.

        Note: targets are expected in log1p(price) space as per your repo convention.
        """
        if self.model_type == "ridge":
            X_tr = np.asarray(train_embeddings)
            y_tr = np.asarray(train_targets)

            self.model.fit(X_tr, y_tr)

            train_loss = float(np.mean((self.model.predict(X_tr) - y_tr) ** 2))

            val_loss = None
            # CHANGED: validate only when BOTH val arrays are provided to avoid NoneType operations.
            if val_embeddings is not None and val_targets is not None:
                X_val = np.asarray(val_embeddings)
                y_val = np.asarray(val_targets)
                val_loss = float(np.mean((self.model.predict(X_val) - y_val) ** 2))

            # CHANGED: print guard uses 'is not None' (val_loss could be 0.0).
            print(
                f"Ridge - train_loss: {train_loss:.4f}"
                + (f" - val_loss: {val_loss:.4f}" if val_loss is not None else "")
            )
            return {"train_loss": [train_loss], "val_loss": ([val_loss] if val_loss is not None else [])}

        # ----- MLP path -----
        train_dataset = ImageDataset(train_embeddings, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_embeddings is not None and val_targets is not None:
            val_dataset = ImageDataset(val_embeddings, val_targets)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Train
            self.model.train()
            total_loss = 0.0

            for embeddings, targets in train_loader:
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(embeddings)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            train_loss = total_loss / max(1, len(train_loader))
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for embeddings, targets in val_loader:
                        embeddings = embeddings.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(embeddings)
                        loss = criterion(outputs, targets)
                        total_val_loss += float(loss.item())
                val_loss = total_val_loss / max(1, len(val_loader))
                history["val_loss"].append(val_loss)

            # CHANGED: print guard uses 'is not None' (val_loss could be 0.0).
            print(
                f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}"
                + (f" - val_loss: {val_loss:.4f}" if val_loss is not None else "")
            )

        return history

    def predict(self, embeddings):
        """
        Predict from embeddings (log1p space).
        """
        if self.model_type == "ridge":
            return np.asarray(self.model.predict(np.asarray(embeddings)))

        self.model.eval()
        embeddings_tensor = torch.as_tensor(embeddings, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(embeddings_tensor).cpu().numpy()
        return predictions

    def save(self, path: str):
        """Save model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.model_type == "ridge":
            joblib.dump(
                {
                    "model": self.model,
                    "model_type": self.model_type,
                    "embedding_dim": self.embedding_dim,
                },
                path,
            )
        else:
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "model_type": self.model_type,
                    "embedding_dim": self.embedding_dim,
                },
                path,
            )

    def load(self, path: str):
        """Load model."""
        if self.model_type == "ridge":
            data = joblib.load(path)
            self.model = data["model"]  # no API change; stored estimator is reused
        else:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()


if __name__ == "__main__":
    # Lightweight self-test (kept intact)
    print("Testing ImageModel...")

    n_train = 100
    n_val = 20
    embedding_dim = 512

    rng = np.random.default_rng(2025)
    train_embeddings = rng.normal(size=(n_train, embedding_dim))
    train_targets = np.log1p(rng.uniform(50, 500, n_train))

    val_embeddings = rng.normal(size=(n_val, embedding_dim))
    val_targets = np.log1p(rng.uniform(50, 500, n_val))

    print("\n--- Testing Ridge ---")
    ridge_trainer = ImageModelTrainer(model_type="ridge", embedding_dim=embedding_dim)
    _ = ridge_trainer.train(train_embeddings, train_targets, val_embeddings, val_targets)
    ridge_preds = ridge_trainer.predict(val_embeddings[:5])
    print(f"Ridge predictions: {ridge_preds}")

    print("\n--- Testing MLP ---")
    mlp_trainer = ImageModelTrainer(model_type="mlp", embedding_dim=embedding_dim)
    _ = mlp_trainer.train(
        train_embeddings, train_targets, val_embeddings, val_targets, batch_size=32, epochs=3
    )
    mlp_preds = mlp_trainer.predict(val_embeddings[:5])
    print(f"MLP predictions: {mlp_preds}")

    print("\n✓ ImageModel test passed")
