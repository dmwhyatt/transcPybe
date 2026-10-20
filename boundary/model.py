"""The boundary network: a small convolutional front end over log-mel, then a
bidirectional GRU over time with transcPybe's f0 features alongside.

Time is never pooled, so every 10 ms frame gets its own three outputs (onset,
offset, sounding) as logits. Frequency is pooled by the convolutions, which is
where consonants and vowel changes show: broadband noise, formant moves.
About 1M parameters at the defaults.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class NetConfig:
    n_mels: int = 80
    n_aux: int = 5
    channels: tuple = (32, 64, 96)
    hidden: int = 128
    gru_layers: int = 2
    dropout: float = 0.2
    use_aux: bool = True
    n_out: int = 3


class BoundaryNet(nn.Module):
    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        layers, c_in, f = [], 1, cfg.n_mels
        for c in cfg.channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.GELU(),
                       nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.GELU(),
                       nn.MaxPool2d((1, 2)), nn.Dropout(cfg.dropout)]
            c_in, f = c, f // 2
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(c_in * f + (cfg.n_aux if cfg.use_aux else 0), cfg.hidden)
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, num_layers=cfg.gru_layers, batch_first=True,
                          bidirectional=True, dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0)
        self.head = nn.Linear(2 * cfg.hidden, cfg.n_out)

    def forward(self, mel: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """mel (B, T, n_mels), aux (B, T, n_aux) -> logits (B, T, n_out)."""
        x = self.conv(mel.unsqueeze(1))                  # (B, C, T, F')
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        if self.cfg.use_aux:
            x = torch.cat([x, aux], dim=-1)
        x, _ = self.gru(torch.nn.functional.gelu(self.proj(x)))
        return self.head(x)


def save(path, model: BoundaryNet, **extra) -> None:
    torch.save(dict(state=model.state_dict(), cfg=asdict(model.cfg), **extra), path)


def load(path, device="cpu") -> BoundaryNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    cfg["channels"] = tuple(cfg["channels"])
    model = BoundaryNet(NetConfig(**cfg))
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval()
