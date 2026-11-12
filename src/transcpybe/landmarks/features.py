"""What the networks see, frame by frame, on transcPybe's 10 ms f0 grid (frame k centred at k*10 ms).

- ``mel`` (T, 80): log-mel of the audio at 16 kHz, n_fft 1024, floored 80 dB below the recording's loudest bin,
  then standardised per recording.
- ``aux`` (T, 5): transcPybe's own f0 track: pitch relative to C4 in octaves (0 where unpitched), a pitched flag,
  periodicity, voicing, and level in dB relative to the recording's 95th percentile, divided by 40 and clipped to
  [-2, 0.5].

Copied from the training repository's ``boundary/features.py`` (boundary 56d97f6). ``frame_features`` is unchanged;
``check_tracker`` also accepts the trackers that ship with a voicing map.
"""
from __future__ import annotations

import warnings

import numpy as np

HOP_S = 0.01            # one frame per transcPybe f0 frame: 160 samples at 16 kHz
FEATURE_SR = 16000      # what transcPybe analyses at
N_MELS = 80
MEL_TOP_DB = 80.0
LEVEL_RANGE = (-2.0, 0.5)     # 80 dB below the 95th percentile, 20 dB above it

TRAINED_ON_TRACKER = "crepe-full"
"""The pitch tracker the networks were trained on, as ``F0Track.backend`` names it."""

MAPPED_TRACKERS = ("pesto", "pyin")
"""Trackers with a voicing map (``voicing``): the presets were scored with each of them as well."""


def check_tracker(track) -> None:
    """Warn when ``track`` comes from a pitch tracker the presets were never scored with.

    Three of the five aux features (pitch, periodicity, voicing) are that
    tracker's output, and periodicity in particular means a different thing for
    each one, so a network fed another tracker's track is outside what it was
    trained and checked on. A track that names no tracker (built by hand, or
    pickled before tracks recorded one) is taken to be CREPE's.
    """
    backend = getattr(track, "backend", "") or TRAINED_ON_TRACKER
    if backend != TRAINED_ON_TRACKER and backend not in MAPPED_TRACKERS:
        warnings.warn(f"the boundary networks were trained on {TRAINED_ON_TRACKER} pitch tracks, and the presets "
                      f"scored with it, PESTO and pYIN; this one is from {backend}, so its results are unchecked",
                      stacklevel=3)


def frame_features(audio: np.ndarray, track) -> tuple:
    """(mel, aux) for 16 kHz mono ``audio`` and its transcPybe ``F0Track``, trimmed to the same frames."""
    import librosa

    check_tracker(track)

    hop = int(round(HOP_S * FEATURE_SR))
    mel = librosa.feature.melspectrogram(y=audio, sr=FEATURE_SR, n_fft=1024, hop_length=hop,
                                         n_mels=N_MELS, fmin=40, fmax=8000)
    mel = librosa.power_to_db(mel, ref=np.max, top_db=MEL_TOP_DB).T
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)          # per recording: gain-invariant
    f0 = np.asarray(track.f0_hz, dtype=np.float64)
    pitched = np.isfinite(f0) & (f0 > 0)
    rel = np.zeros_like(f0)
    rel[pitched] = np.log2(f0[pitched] / 261.63)
    level = np.asarray(track.loudness_db, dtype=np.float64)
    level = np.clip((level - np.percentile(level, 95)) / 40.0, *LEVEL_RANGE)
    aux = np.stack([rel, pitched.astype(float), track.periodicity, track.voicing, level], axis=1)
    n = min(len(mel), len(aux))
    return mel[:n].astype(np.float32), aux[:n].astype(np.float32)
