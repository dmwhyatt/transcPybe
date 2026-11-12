"""The boundary network: a small convolutional front end over log-mel, then a
bidirectional GRU over time with transcPybe's f0 features alongside.

Time is never pooled, so every 10 ms frame gets its own outputs as logits. The
shipped networks have five: where a syllable starts sounding (its articulation),
offset, sounding, re-sung onset, and where the syllable's vowel starts.
Frequency is pooled by the convolutions, which is where consonants and vowel
changes show: broadband noise, formant moves. About 0.8M parameters.

Copied from the training repository's ``boundary/model.py`` (boundary 56d97f6),
less the training-only ``save``; ``forward`` and ``load`` are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    n_out: int = 4


class BoundaryNet(nn.Module):
    def __init__(self, cfg: Optional[NetConfig] = None):
        super().__init__()
        cfg = cfg if cfg is not None else NetConfig()
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

    def forward(self, mel: torch.Tensor, aux: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """mel (B, T, n_mels), aux (B, T, n_aux) -> logits (B, T, n_out).

        ``lengths`` (B,) gives each sequence's real length in a zero-padded batch,
        so a padded batch gives what each sequence would give alone: padded frames
        are zeroed before every convolution (bias and BatchNorm would otherwise
        make them non-zero and leak into the last real frames), and the GRU runs
        over the real frames only, so its backward direction starts at each
        sequence's own end.
        """
        x = mel.unsqueeze(1)                             # (B, 1, T, F)
        t = x.shape[2]
        keep = None
        if lengths is not None:
            keep = (torch.arange(t, device=x.device)[None, :] < lengths.to(x.device)[:, None])
            keep = keep.to(x.dtype)[:, None, :, None]
        for layer in self.conv:
            if keep is not None and isinstance(layer, nn.Conv2d):
                x = x * keep
            x = layer(x)                                 # ends (B, C, T, F')
        b, c, _, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        if self.cfg.use_aux:
            x = torch.cat([x, aux], dim=-1)
        x = nn.functional.gelu(self.proj(x))
        if lengths is None:
            x, _ = self.gru(x)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=t)
        return self.head(x)


def load(path, device="cpu") -> BoundaryNet:
    """A network from a checkpoint: a path, or a binary file object."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    cfg["channels"] = tuple(cfg["channels"])
    model = BoundaryNet(NetConfig(**cfg))
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval()
