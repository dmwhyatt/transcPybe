"""A note's pitch after its start has moved: measure_after_snap and cut_settle_s.

In both cases the pitch tracker still reports the previous note's pitch for a while after the attack, so measuring
from the attack takes in the old pitch. Each test builds a case where the default measurement rounds to the old
pitch and the option recovers the new one.
"""
import dataclasses

import numpy as np

from transcpybe.config import TranscriptionConfig
from transcpybe.f0 import F0Track
from transcpybe.notes import segments_to_notes
from transcpybe.onsets import OnsetTrack, snap_segment_starts, split_segments_on_onsets
from transcpybe.segment import Segment

HZ = {60: 261.6256, 64: 329.6276}


def _track(pitches):
    """One frame per entry: a MIDI pitch, or None for silence."""
    n = len(pitches)
    voiced = np.array([p is not None for p in pitches])
    return F0Track(times=np.arange(n) * 0.01, f0_hz=np.array([HZ[p] if p else np.nan for p in pitches]),
                   periodicity=np.where(voiced, 0.95, 0.0), loudness_db=np.where(voiced, -20.0, -60.0),
                   loudness_gate=voiced.astype(float), voicing=np.where(voiced, 0.95, 0.0))


def _onsets(times, n):
    return OnsetTrack(np.array(times), np.ones(len(times)), np.zeros(n), np.arange(n) * 0.01)


def test_a_start_snapped_earlier_is_measured_from_where_its_pitch_crossed_over():
    # the decoder put the boundary at frame 50, where the pitch crossed over; the attack is at frame 28
    track = _track([60] * 50 + [64] * 14 + [None] * 16)
    segments = [Segment(0, 50, 60), Segment(50, 64, 64)]
    base = TranscriptionConfig(device="cpu", onset_snap_window_s=0.3)
    for option, expected in ((False, 60), (True, 64)):
        cfg = dataclasses.replace(base, measure_after_snap=option)
        snapped = snap_segment_starts(segments, _onsets([0.28], 80), track.times, cfg, track.voicing)
        assert [(s.start_frame, s.end_frame) for s in snapped] == [(0, 28), (28, 64)]
        notes = segments_to_notes(snapped, track, cfg)
        assert notes[1].onset == 0.28 and notes[1].pitch == expected


def test_a_part_after_a_cut_is_measured_once_its_pitch_has_settled():
    # one decoded note, cut at frame 30; the tracker holds the old pitch until frame 40
    track = _track([60] * 41 + [64] * 7 + [None] * 12)
    for settle, expected in ((0.0, 60), (0.10, 64)):
        cfg = TranscriptionConfig(device="cpu", cut_settle_s=settle)
        parts = split_segments_on_onsets([Segment(0, 48, 60)], _onsets([0.30], 60), track.times, cfg,
                                         track.loudness_db, track.periodicity, track.voicing)
        assert [(s.start_frame, s.end_frame) for s in parts] == [(0, 30), (30, 48)]
        notes = segments_to_notes(parts, track, cfg)
        assert notes[1].onset == 0.30 and notes[1].pitch == expected
