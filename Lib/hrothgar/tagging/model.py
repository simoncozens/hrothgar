"""Tag prediction head on frozen gap features."""

from __future__ import annotations

from torch import nn

from hrothgar.tagging.config import TaggingConfig
from hrothgar.utils import SaveLoadModel


class TagPredictor(SaveLoadModel):
    """A small MLP mapping a gap feature vector to one scalar per tag (in [0, 1])."""

    def __init__(self, config: TaggingConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.num_tags = len(config.tag_names)

        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.SiLU())
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, self.num_tags))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``(B, num_tags)`` predictions in [0, 1]."""
        return self.net(features)
