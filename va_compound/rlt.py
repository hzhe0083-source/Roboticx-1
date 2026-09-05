"""Core modules for RL Token (RLT).

The encoder/decoder follows Eq. 1-2 of ``RL Token: Bootstrapping Online RL
with Vision-Language-Action Models``.  The frozen VLA token sequence is
compressed through learned ``<rl>`` token(s), then reconstructed
autoregressively during the representation-training stage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RLTokenConfig:
    token_dim: int
    num_tokens: int = 1
    heads: int = 8
    encoder_layers: int = 3
    decoder_layers: int = 3
    ff_dim: int = 2048
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "token_dim": self.token_dim,
            "num_tokens": self.num_tokens,
            "heads": self.heads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "ff_dim": self.ff_dim,
        }
        if any(value < 1 for value in positive.values()):
            raise ValueError(f"RL-token dimensions must be positive: {positive}")
        if self.token_dim % self.heads:
            raise ValueError("token_dim must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


class RLTokenModule(nn.Module):
    """Learned RL-token bottleneck with an optional training-only decoder."""

    def __init__(self, config: RLTokenConfig, *, with_decoder: bool = True) -> None:
        super().__init__()
        self.config = config
        self.rl_tokens = nn.Parameter(
            torch.randn(1, config.num_tokens, config.token_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_layers
        )
        self.decoder: nn.TransformerDecoder | None = None
        self.output_projection: nn.Linear | None = None
        if with_decoder:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.token_dim,
                nhead=config.heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=config.decoder_layers
            )
            self.output_projection = nn.Linear(config.token_dim, config.token_dim)

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.token_dim:
            raise ValueError(
                "VLA tokens must have shape [batch, sequence, "
                f"{self.config.token_dim}], got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] < self.config.num_tokens:
            raise ValueError("VLA token sequence is shorter than the RL-token prefix")

    def encode_multi(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        self._validate_tokens(vla_tokens)
        frozen = vla_tokens.detach()
        special = self.rl_tokens.expand(frozen.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((frozen, special), dim=1))
        return encoded[:, -self.config.num_tokens :]

    def encode(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """Return the compact state vector used by the actor and critic."""
        return self.encode_multi(vla_tokens).mean(dim=1)

    def decode(
        self, rl_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        self._validate_tokens(teacher_tokens)
        if self.decoder is None or self.output_projection is None:
            raise RuntimeError("this inference-only RL-token module has no decoder")
        if rl_tokens.ndim == 2:
            rl_tokens = rl_tokens[:, None]
        expected = (
            teacher_tokens.shape[0],
            self.config.num_tokens,
            self.config.token_dim,
        )
        if tuple(rl_tokens.shape) != expected:
            raise ValueError(
                f"RL tokens must have shape {expected}, got {tuple(rl_tokens.shape)}"
            )
        length = teacher_tokens.shape[1]
        prefix = teacher_tokens.detach()[:, : length - self.config.num_tokens]
        decoder_input = torch.cat((rl_tokens, prefix), dim=1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            length, device=teacher_tokens.device, dtype=teacher_tokens.dtype
        )
        decoded = self.decoder(
            tgt=decoder_input, memory=rl_tokens, tgt_mask=causal_mask
        )
        return self.output_projection(decoded)

    def reconstruction_loss(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """Autoregressive stop-gradient reconstruction objective from paper Eq. 2."""
        target = vla_tokens.detach()
        return F.mse_loss(self.decode(self.encode_multi(target), target), target)

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Small deployment state without the training-only decoder."""
        return {
            key: value
            for key, value in self.state_dict().items()
            if key == "rl_tokens" or key.startswith("encoder.")
        }

    def load_encoder_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.encoder_state_dict())
        if set(state) != expected:
            raise ValueError(
                "RL-token encoder keys differ: "
                f"missing={sorted(expected - set(state))[:8]} "
                f"unexpected={sorted(set(state) - expected)[:8]}"
            )
        missing, unexpected = self.load_state_dict(state, strict=False)
        decoder_missing = {
            key
            for key in missing
            if key.startswith("decoder.") or key.startswith("output_projection.")
        }
        if set(missing) != decoder_missing or unexpected:
            raise RuntimeError(
                f"RL-token encoder load failed: missing={missing} unexpected={unexpected}"
            )
