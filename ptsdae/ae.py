import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from typing import Optional

class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        activation: Optional[torch.nn.Module] = nn.ReLU(),
        gain: float = nn.init.calculate_gain("relu"),
        tied: bool = False,
    ) -> None:
        """
        Autoencoder composed of two Linear layers with optional activation.

        :param input_dimension: Dimension de l'entrée
        :param hidden_dimension: Dimension de l'espace latent
        :param activation: Fonction d'activation optionnelle, par défaut nn.ReLU()
        :param gain: Gain pour l'initialisation des poids
        :param tied: Si True, les poids du décodeur sont liés aux poids de l'encodeur
        """
        super(AutoEncoder, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.activation = activation
        self.gain = gain
        self.tied = tied

        # Paramètres de l'encodeur
        self.encoder_weight = Parameter(
            torch.Tensor(hidden_dimension, input_dimension)
        )
        self.encoder_bias = Parameter(torch.Tensor(hidden_dimension))
        self._initialise_weight_bias(self.encoder_weight, self.encoder_bias, self.gain)

        # Paramètres du décodeur
        self._decoder_weight = (
            Parameter(torch.Tensor(input_dimension, hidden_dimension))
            if not tied
            else None
        )
        self.decoder_bias = Parameter(torch.Tensor(input_dimension))
        self._initialise_weight_bias(self._decoder_weight, self.decoder_bias, self.gain)

    @property
    def decoder_weight(self):
        return (
            self._decoder_weight
            if self._decoder_weight is not None
            else self.encoder_weight.t()
        )

    @staticmethod
    def _initialise_weight_bias(weight: torch.Tensor, bias: torch.Tensor, gain: float):
        if weight is not None:
            nn.init.xavier_uniform_(weight, gain)
        nn.init.constant_(bias, 0)

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        transformed = F.linear(batch, self.encoder_weight, self.encoder_bias)
        if self.activation is not None:
            transformed = self.activation(transformed)
        return transformed

    def decode(self, batch: torch.Tensor) -> torch.Tensor:
        return F.linear(batch, self.decoder_weight, self.decoder_bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(batch)
        decoded = self.decode(encoded)
        return decoded
