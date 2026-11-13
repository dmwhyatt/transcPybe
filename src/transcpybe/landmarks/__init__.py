"""transcPybe v2: note starts from three small neural networks, notes from transcPybe's decoder.

The networks (``model.BoundaryNet``: a convolutional front end over log-mel and a bidirectional GRU, about 0.8M
parameters each) were trained with three seeds on the same data. Each reads the audio's log-mel and the pitch
tracker's output, and marks, every 10 ms, where a syllable starts sounding and where its vowel starts. Their outputs
are averaged, a note start is placed between each such pair (``infer.landmark_track``), and transcPybe's note decoder
makes the notes from the pitch track with those starts.

Two presets, each fixed on tuning sets and then checked once on test recordings nothing had been tuned on:

- ``"acappella"``, for solo singing with nothing else in the recording: checked on SSVD's 127 test recordings;
- ``"pop"``, for a lead vocal separated from a pop mix: checked on 82 MIR-ST500 test songs.

``TranscriptionConfig(preset=...)`` selects one, ``"pop"`` by default; ``"rules"`` is transcPybe v1, with no networks.

The networks were trained on CREPE full's pitch tracks. PESTO and pYIN work too: their confidence runs lower than
CREPE's, so the decoder gets their track through a voicing map (``voicing``), and the networks get it as it is.

The networks run on the CPU, as they did when the presets were scored. Their weights and the voicing maps (``weights/``) are licensed CC BY-NC-SA 4.0, not MIT: see
``weights/README.md``.
"""
from __future__ import annotations

import dataclasses
import functools
import json
from dataclasses import dataclass
from importlib import resources

from .features import FEATURE_SR, HOP_S, MAPPED_TRACKERS, TRAINED_ON_TRACKER

SEEDS = (0, 1, 2)

# The decoder settings the networks were scored with: B-plain (the settings frozen on RWC-P's development half as
# B-full, without its timbre and voicing-break splits), the sounding merge at 0.99, and note starts snapped to an
# onset up to 150 ms either way.
B_PLAIN = dict(min_note_duration_s=0.12, note_level_floor_db=15.0, fragment_max_duration_s=0.18,
               fragment_dominance_ratio=1.5, p_self=0.99, timbre_split_ratio=0.0, onset_delta_ratio=0.6,
               onset_split_strength_ratio=0.5, onset_split_min_gap_s=0.15, split_on_voicing_breaks=False)
LANDMARK_DECODER = dict(B_PLAIN, sustain_merge_above=0.99, onset_snap_window_s=0.15, onset_snap_direction="both")


@dataclass(frozen=True)
class Preset:
    """How the networks' outputs become note starts, and where a note's release is trimmed."""
    fraction: float
    """Where a note starts between a syllable's articulation (0) and its vowel (1)."""
    threshold: float
    """The peak height an output must reach to count as an onset."""
    lone: str
    """What an articulation or vowel onset with no partner does (``infer.landmark_onsets``)."""
    release_drop_db: float
    """The decoder's ``release_drop_db``: a note ends where it falls this far below its own level."""
    use: str
    """The recordings it is for."""


PRESETS = {
    "acappella": Preset(fraction=0.875, threshold=0.6, lone="drop", release_drop_db=15.0,
                        use="solo singing with nothing else in the recording"),
    "pop": Preset(fraction=0.5, threshold=0.5, lone="snap", release_drop_db=20.0,
                  use="a lead vocal separated from a pop mix"),
}
DECODER_FIELDS = tuple(LANDMARK_DECODER) + ("release_drop_db",)
"""The config fields a preset sets; the rest of the config is used as given."""


def preset_config(config):
    """``config`` with its preset's decoder settings. Unchanged for ``preset="rules"``."""
    if config.preset == "rules":
        return config
    return dataclasses.replace(config, **LANDMARK_DECODER, release_drop_db=PRESETS[config.preset].release_drop_db)


def _weights():
    return resources.files(__name__) / "weights"


def _needs_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError("the acappella and pop presets run three small neural networks, which need PyTorch: "
                          "pip install torch") from None


@functools.lru_cache(maxsize=1)
def networks() -> tuple:
    """The three networks, on the CPU, loaded once per process."""
    _needs_torch()
    from .model import load

    nets = []
    for s in SEEDS:
        with (_weights() / f"seed{s}.pt").open("rb") as f:
            nets.append(load(f, "cpu"))
    return tuple(nets)


@functools.lru_cache(maxsize=None)
def voicing_map(backend: str):
    """``backend``'s voicing map (as ``F0Track.backend`` names it), or None if the decoder takes its track as it is."""
    if backend not in MAPPED_TRACKERS:
        return None
    return json.loads((_weights() / f"voicing-{backend}.json").read_text())["map"]


def landmark_notes(audio, track, config):
    """Notes for ``audio`` (16 kHz mono) from its pitch ``track``, with the preset ``config.preset`` names.

    Returns what ``transcpybe.core.decode_notes`` returns: (notes, the networks' onset track, raw segments, tuning
    offset). This is the sequence the presets were scored with. ``config.onset_position``, if set, replaces the
    preset's ``fraction``: where a note starts between its syllable's start (0) and its vowel's (1).
    """
    from ..core import decode_notes
    from .features import frame_features
    from .infer import averaged_posteriors, landmark_track
    from .voicing import apply

    p = PRESETS[config.preset]
    fraction = p.fraction if config.onset_position is None else config.onset_position
    mel, aux = frame_features(audio, track)                   # the networks always get the tracker's own track
    post = averaged_posteriors(networks(), mel, aux, "cpu")
    onsets = landmark_track(post, fraction, p.threshold, sounding=post[:, 2], lone=p.lone)
    m = voicing_map(getattr(track, "backend", "") or TRAINED_ON_TRACKER)
    decode_with = apply(track, m) if m is not None else track
    return decode_notes(decode_with, preset_config(config), audio=audio, onsets=onsets)


__all__ = ["PRESETS", "Preset", "LANDMARK_DECODER", "DECODER_FIELDS", "preset_config", "networks", "voicing_map",
           "landmark_notes", "FEATURE_SR", "HOP_S"]
