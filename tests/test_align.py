"""Warping a notated reference onto the recording it describes.

A free warp aligns anything to anything, so most of what these pin is the
refusal: an alignment that is not a tempo, a reference that belongs to another
song, a warp that crushes a note to nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from transcpybe.benchmark.align import align_reference
from transcpybe.benchmark.metrics import score_item

MELODY = [60, 62, 64, 65, 67, 65, 64, 62, 60, 59, 60, 62, 64, 65, 67, 69, 67, 65]


def performance(n=200, duration=0.45, gap=0.15, start=3.0):
    notes, t = [], start
    for i in range(n):
        notes.append((t, t + duration, float(MELODY[i % len(MELODY)])))
        t += duration + gap
    return notes


def contour(notes, hop=0.01):
    end = max(e for _, e, _ in notes) + 2.0
    times = np.arange(0.0, end, hop)
    f0 = np.zeros(times.shape)
    voicing = np.zeros(times.shape)
    for start, stop, pitch in notes:
        mask = (times >= start) & (times < stop)
        f0[mask] = 440.0 * 2 ** ((pitch - 69) / 12)
        voicing[mask] = 0.9
    return f0, voicing, times


def notated(notes, scale=1.0, offset=0.0, transpose=0):
    """The reference as a score would hold it: one tempo, its own start."""
    return [(s * scale + offset, e * scale + offset, p + transpose)
            for s, e, p in notes]


@pytest.mark.slow
@pytest.mark.parametrize("scale,offset", [(0.985, 0.4), (1.03, -0.8), (1.0, 0.0)])
def test_alignment_recovers_a_known_warp(scale, offset):
    truth = performance()
    f0, voicing, times = contour(truth)
    reference = notated(truth, scale, offset)

    before = score_item("raw", truth, reference).conp.f
    warp = align_reference(reference, f0, voicing, times)
    after = score_item("aligned", truth, warp.warp_notes(reference)).conp.f

    assert warp.ok, warp.reason
    assert after > 0.95
    if scale != 1.0:
        assert before < 0.3, "precondition: the unwarped reference scores badly"


@pytest.mark.slow
def test_a_reference_from_another_song_is_refused():
    """A long recording holds frames at every pitch, so a wrong melody also
    finds a cheap path. What gives it away is that the path is not a tempo."""
    truth = performance()
    f0, voicing, times = contour(truth)
    rng = np.random.default_rng(1)
    other = [(s, e, float(60 + rng.integers(0, 12))) for s, e, _ in performance()]

    warp = align_reference(other, f0, voicing, times)

    assert not warp.ok
    assert "not a tempo" in warp.reason
    assert warp.steady_fraction < 0.85


@pytest.mark.slow
def test_a_transposed_reference_is_refused_rather_than_mis_warped():
    """The aligner needs the reference in the recording's key, and says so."""
    truth = performance()
    f0, voicing, times = contour(truth)

    warp = align_reference(notated(truth, 0.985, 0.4, transpose=7), f0, voicing, times)

    assert not warp.ok


@pytest.mark.slow
def test_warping_never_produces_a_note_of_no_length():
    """mir_eval rejects a zero-length interval, and a hard compression makes one."""
    truth = performance(n=60, duration=0.06, gap=0.02)
    f0, voicing, times = contour(truth)
    reference = notated(truth, 0.9, 0.2)

    warp = align_reference(reference, f0, voicing, times)
    warped = warp.warp_notes(reference)

    assert all(end > start for start, end, _ in warped)


def test_a_rejected_alignment_leaves_the_reference_alone():
    from transcpybe.benchmark.align import Alignment

    notes = [(0.0, 1.0, 60.0), (1.0, 2.0, 62.0)]

    assert Alignment(False, "nope").warp_notes(notes) == notes


def test_local_slope_measures_the_rate_not_the_step():
    """Frame to frame a path steps diagonally or sideways, so its instantaneous
    slope is only ever zero or one and says nothing."""
    from transcpybe.benchmark.align import _local_slopes

    source = np.arange(0.0, 60.0, 0.05)
    steady = _local_slopes(source, source * 1.02)
    lurching = _local_slopes(source, np.sort(source + np.sin(source) * 8.0))

    assert np.allclose(steady, 1.02, atol=0.02)
    assert np.std(lurching) > np.std(steady) * 5
