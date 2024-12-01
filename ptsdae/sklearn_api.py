# transformer.py

from typing import Optional, List
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from ae import AutoEncoder  # Import de la classe AutoEncoder

class AutoEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        cuda: Optional[bool] = None,
        batch_size: int = 256,
        epochs: int = 100,
        learning_rate: float = 0.001,
        scheduler_step_size: int = 100,
        scheduler_gamma: float = 0.1,
    ):
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.autoencoder = None
        self.scaler = None  # Pour normaliser les données

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        elif isinstance(X, torch.Tensor):
            X = X.float()
        else:
            raise ValueError("Unsupported data type for X")

        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.autoencoder = AutoEncoder(
            input_dimension=self.input_dimension,
            hidden_dimension=self.hidden_dimension,
            activation=nn.ReLU(),
            tied=False
        )

        if self.cuda:
            self.autoencoder.cuda()

        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        loss_function = nn.MSELoss()

        self.autoencoder.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                inputs = batch[0]
                if self.cuda:
                    inputs = inputs.cuda()

                optimizer.zero_grad()
                outputs = self.autoencoder(inputs)
                loss = loss_function(outputs, inputs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)

            scheduler.step()
            epoch_loss /= len(dataloader.dataset)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        return self

    def transform(self, X):
        if self.autoencoder is None:
            raise NotFittedError("This AutoEncoderTransformer instance is not fitted yet.")

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        elif isinstance(X, torch.Tensor):
            X = X.float()
        else:
            raise ValueError("Unsupported data type for X")

        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.autoencoder.eval()
        features = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0]
                if self.cuda:
                    inputs = inputs.cuda()
                encoded = self.autoencoder.encode(inputs)
                features.append(encoded.cpu())

        return torch.cat(features).numpy()

    def score(self, X, y=None):
        if self.autoencoder is None:
            raise NotFittedError("This AutoEncoderTransformer instance is not fitted yet.")

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        elif isinstance(X, torch.Tensor):
            X = X.float()
        else:
            raise ValueError("Unsupported data type for X")

        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.autoencoder.eval()
        loss_function = nn.MSELoss()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0]
                if self.cuda:
                    inputs = inputs.cuda()
                outputs = self.autoencoder(inputs)
                loss = loss_function(outputs, inputs)
                total_loss += loss.item() * inputs.size(0)

        return total_loss / len(dataloader.dataset)
