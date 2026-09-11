"""Absorbing spurious short segments into the note they belong to.

The pitch HMM reliably finds the body of each note, but it tends to shed a
short fragment at a note's edge, one or two semitones away from the note
proper. The commonest cause is wide vibrato: as a sustained note decays, the
final downward swing of the vibrato dips far enough that the decoder spends a
few frames in the semitone below, and that tail is then emitted as a note of
its own. Consonants and breath at a note's edge do the same thing.

These fragments are recognisable without guessing at their cause: they are very
short, they sit immediately against a neighbour many times their length, and
they are within a semitone or two of it. Nothing sung looks like that, so they
are folded back into the neighbour they came from.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .config import TranscriptionConfig
from .segment import Segment


def _neighbour_score(fragment: Segment, neighbour: Optional[Segment],
                     config: TranscriptionConfig) -> Optional[int]:
    """Length of ``neighbour`` if it can absorb ``fragment``, else None."""
    if neighbour is None:
        return None

    # Contiguity is required rather than merely "close": absorbing across a gap
    # would stretch a note over a rest it was not sounding through.
    contiguous = (
        neighbour.end_frame == fragment.start_frame
        or fragment.end_frame == neighbour.start_frame
    )
    if not contiguous:
        return None
    if abs(neighbour.pitch - fragment.pitch) > config.fragment_max_interval_semitones:
        return None
    if neighbour.n_frames < config.fragment_dominance_ratio * fragment.n_frames:
        return None
    return neighbour.n_frames


def absorb_short_fragments(
    segments: List[Segment], config: TranscriptionConfig
) -> List[Segment]:
    """Fold short edge fragments into the dominant neighbour beside them."""
    if not config.absorb_fragments or len(segments) < 2:
        return segments

    hop_s = config.hop_length / config.sample_rate
    max_frames = int(round(config.fragment_max_duration_s / hop_s))

    working = list(segments)
    # Repeat to a fixed point: absorbing one fragment can expose another that is
    # now adjacent to a long enough neighbour.
    for _ in range(len(segments)):
        target: Optional[Tuple[int, int]] = None      # (fragment index, neighbour index)
        best_length = -1

        for i, fragment in enumerate(working):
            if fragment.n_frames > max_frames:
                continue
            previous = working[i - 1] if i > 0 else None
            following = working[i + 1] if i + 1 < len(working) else None

            for offset, neighbour in ((-1, previous), (1, following)):
                length = _neighbour_score(fragment, neighbour, config)
                if length is not None and length > best_length:
                    best_length, target = length, (i, i + offset)

        if target is None:
            break

        i, j = target
        fragment, neighbour = working[i], working[j]
        merged = Segment(
            start_frame=min(fragment.start_frame, neighbour.start_frame),
            end_frame=max(fragment.end_frame, neighbour.end_frame),
            pitch=neighbour.pitch,
        )
        working[j] = merged
        working.pop(i)

    return working
