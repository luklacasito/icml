"""Retained benchmark models, extracted from the recovered execution source.

Module/parameter order and initialization are preserved; only unused muP
construction has been removed. See README.md for provenance and limitations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn

ActivationName = Literal["relu", "gelu"]


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int = 3 * 32 * 32
    width: int = 256
    output_dim: int = 10
    depth: int = 6
    activation: ActivationName = "relu"
    sigma_w_sq: float = 1.98
    sigma_b_sq: float = 0.02


class CriticalMLP(nn.Module):
    """Plain depth-dependent-dropout MLP used by the paper experiments.

    ``depth`` is the number of hidden affine/nonlinear/dropout blocks.  The
    classifier is a separate readout, matching the existing notebooks.
    """

    def __init__(
        self,
        config: MLPConfig,
        dropout_layers: list[float] | tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(dropout_layers) != config.depth:
            raise ValueError(
                f"Expected {config.depth} dropout probabilities, got {len(dropout_layers)}"
            )

        self.config = config
        hidden: list[nn.Linear] = [nn.Linear(config.input_dim, config.width)]
        hidden.extend(nn.Linear(config.width, config.width) for _ in range(config.depth - 1))
        self.hidden = nn.ModuleList(hidden)
        self.dropouts = nn.ModuleList(nn.Dropout(float(p)) for p in dropout_layers)
        self.activation = {"relu": nn.ReLU, "gelu": nn.GELU}[config.activation]()

        self.readout = nn.Linear(config.width, config.output_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        for layer, dropout in zip(self.hidden, self.dropouts, strict=True):
            x = dropout(self.activation(layer(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.forward_features(x))


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 4,
        *,
        in_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.projection = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dimension, 3 * dimension, bias=False)
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dimension = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, dimension // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, tokens, dimension)
        return self.projection(attended)


class _TransformerMLP(nn.Module):
    def __init__(self, dimension: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dimension * ratio)
        self.first = nn.Linear(dimension, hidden)
        self.second = nn.Linear(hidden, dimension)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.activation(self.first(x)))


class _TransformerBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, ratio: float, dropout: float) -> None:
        super().__init__()
        self.first_norm = nn.LayerNorm(dimension)
        self.second_norm = nn.LayerNorm(dimension)
        self.attention = _Attention(dimension, heads)
        self.mlp = _TransformerMLP(dimension, ratio)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.first_norm(x)))
        return x + self.dropout(self.mlp(self.second_norm(x)))


class TinyViT(nn.Module):
    def __init__(
        self,
        dropout_layers: list[float] | tuple[float, ...],
        *,
        dimension: int = 128,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        output_dim: int = 100,
        patch_size: int = 4,
        in_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.patch = _PatchEmbed(
            dimension, patch_size, in_channels=in_channels, image_size=image_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position = nn.Parameter(torch.zeros(1, self.patch.num_patches + 1, dimension))
        self.blocks = nn.ModuleList(
            _TransformerBlock(dimension, heads, mlp_ratio, float(dropout))
            for dropout in dropout_layers
        )
        self.norm = nn.LayerNorm(dimension)
        self.readout = nn.Linear(dimension, output_dim)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1) + self.position
        for block in self.blocks:
            x = block(x)
        return self.readout(self.norm(x)[:, 0])


class SequenceTransformer(nn.Module):
    """TinyViT's block stack over a generic token sequence.

    * limit order book -- ``(batch, snapshots, book_features)`` continuous,
    * tabular -- ``(batch, features, 1)`` continuous, one token per feature,
      which is the numerical-embedding half of an FT-Transformer.

    A linear projection maps each continuous token to the model dimension.
    """

    def __init__(
        self,
        dropout_layers: list[float] | tuple[float, ...],
        *,
        sequence_length: int,
        dimension: int = 128,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        output_dim: int = 2,
        input_features: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(input_features, dimension)

        self.sequence_length = sequence_length
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position = nn.Parameter(torch.zeros(1, sequence_length + 1, dimension))
        self.blocks = nn.ModuleList(
            _TransformerBlock(dimension, heads, mlp_ratio, float(dropout))
            for dropout in dropout_layers
        )
        self.norm = nn.LayerNorm(dimension)
        self.readout = nn.Linear(dimension, output_dim)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.apply(TinyViT._initialize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tabular inputs arrive flat; one token per feature.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.embed(x)
        if x.shape[1] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {x.shape[1]}")
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1) + self.position
        for block in self.blocks:
            x = block(x)
        return self.readout(self.norm(x)[:, 0])


def _initialize_sp(model: CriticalMLP) -> None:
    config = model.config
    for layer in model.hidden:
        fan_in = layer.weight.shape[1]
        nn.init.normal_(layer.weight, mean=0.0, std=(config.sigma_w_sq / fan_in) ** 0.5)
        if config.sigma_b_sq:
            nn.init.normal_(layer.bias, mean=0.0, std=config.sigma_b_sq**0.5)
        else:
            nn.init.zeros_(layer.bias)
    fan_in = model.readout.weight.shape[1]
    nn.init.normal_(
        model.readout.weight,
        mean=0.0,
        std=(config.sigma_w_sq / fan_in) ** 0.5,
    )
    if config.sigma_b_sq:
        nn.init.normal_(model.readout.bias, mean=0.0, std=config.sigma_b_sq**0.5)
    else:
        nn.init.zeros_(model.readout.bias)


def build_model(spec: dict, dropout_layers: list[float]) -> nn.Module:
    from .data import BENCHMARK_SPECS

    data = BENCHMARK_SPECS[spec["dataset"]]
    if len(dropout_layers) != spec["depth"]:
        raise ValueError("Dropout profile length differs from depth")
    if any(not 0 <= value < 1 for value in dropout_layers):
        raise ValueError("Dropout probabilities must lie in [0, 1)")
    if spec["model_kind"] == "mlp":
        config = MLPConfig(
            input_dim=data.mlp_input_dim,
            output_dim=data.classes,
            **{
                key: spec[key]
                for key in ("width", "depth", "activation", "sigma_w_sq", "sigma_b_sq")
            },
        )
        model = CriticalMLP(config, dropout_layers)
        _initialize_sp(model)
        return model
    if spec["model_kind"] != "transformer":
        raise ValueError("Unknown model kind")
    common = dict(
        dimension=spec["width"],
        heads=spec["heads"],
        mlp_ratio=spec["mlp_ratio"],
        output_dim=data.classes,
    )
    if data.image_channels is not None:
        return TinyViT(
            dropout_layers,
            patch_size=data.patch_size,
            in_channels=data.image_channels,
            image_size=data.image_size,
            **common,
        )
    return SequenceTransformer(
        dropout_layers,
        sequence_length=data.sequence_length,
        input_features=data.input_features,
        **common,
    )
