"""Audio loading for transcription."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_audio(path: str | Path, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at ``sample_rate``.

    Multi-channel input is averaged to mono. Any format libsndfile or audioread
    can open is accepted (wav, flac, ogg, mp3, m4a, ...).

    Returns
    -------
    (audio, sample_rate)
        ``audio`` is 1-D float32 in roughly [-1, 1].
    """
    import librosa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    return audio, sr


def frame_rms_db(audio: np.ndarray, hop_length: int, n_frames: int) -> np.ndarray:
    """Per-frame RMS in dBFS, aligned to a centred frame grid of ``n_frames``.

    The window is 4x the hop, which smooths the level enough that a single
    glottal period does not swing the gate.
    """
    import librosa

    frame_length = max(hop_length * 4, 256)
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]

    # librosa's frame count can differ by one from the f0 backend's; align.
    if rms.shape[0] < n_frames:
        rms = np.pad(rms, (0, n_frames - rms.shape[0]), mode="edge")
    elif rms.shape[0] > n_frames:
        rms = rms[:n_frames]

    return 20.0 * np.log10(np.maximum(rms, 1e-10))
