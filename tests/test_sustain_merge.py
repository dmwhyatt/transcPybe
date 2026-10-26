"""Rejoining touching notes that a sounding track hears as one note: sustain_merge_above.

A scoop into a note leaves the decoder with two notes, a short one at the wrong pitch and then the rest. A boundary
network's sounding output stays high through a scoop and dips between real notes; where it stays high, the two
notes are merged and measured as one.
"""
import dataclasses

import numpy as np

from transcpybe.config import TranscriptionConfig
from transcpybe.core import decode_notes
from transcpybe.f0 import F0Track
from transcpybe.onsets import OnsetTrack, merge_sustained_boundaries, refine_segments
from transcpybe.segment import Segment

TIMES = np.arange(80) * 0.01
MERGE = TranscriptionConfig(device="cpu", sustain_merge_above=0.98)


def _sounding(values, times=TIMES, onset_times=()):
    n = len(times)
    return OnsetTrack(np.array(onset_times, float), np.ones(len(onset_times)), np.zeros(n), times,
                      sounding=np.asarray(values, float))


def _spans(segments):
    return [(s.start_frame, s.end_frame) for s in segments]


def test_touching_notes_merge_only_where_the_sounding_track_stays_high_around_the_join():
    segments = [Segment(0, 15, 62), Segment(15, 60, 64)]
    assert _spans(merge_sustained_boundaries(segments, _sounding(np.full(80, 0.99)), TIMES, MERGE)) == [(0, 60)]
    dips = np.full(80, 0.99)
    dips[14] = 0.5                                        # 10 ms before the join
    assert merge_sustained_boundaries(segments, _sounding(dips), TIMES, MERGE) == segments
    far = np.full(80, 0.99)
    far[19] = 0.5                                         # 40 ms after it: outside the window
    assert _spans(merge_sustained_boundaries(segments, _sounding(far), TIMES, MERGE)) == [(0, 60)]
    rest = [Segment(0, 15, 62), Segment(19, 60, 64)]      # a 40 ms gap is a rest, not a join
    assert merge_sustained_boundaries(rest, _sounding(np.full(80, 0.99)), TIMES, MERGE) == rest


def test_a_note_cut_in_three_merges_back_to_one_and_keeps_the_first_measuring_point():
    segments = [Segment(0, 10, 62, measure_from=4), Segment(10, 30, 64), Segment(31, 50, 64)]
    merged = merge_sustained_boundaries(segments, _sounding(np.full(80, 0.99)), TIMES, MERGE)
    assert _spans(merged) == [(0, 50)] and merged[0].measure_from == 4


def test_nothing_merges_with_the_option_off_or_without_a_sounding_track():
    segments = [Segment(0, 15, 62), Segment(15, 60, 64)]
    off = TranscriptionConfig(device="cpu")
    assert merge_sustained_boundaries(segments, _sounding(np.full(80, 0.99)), TIMES, off) == segments
    bare = OnsetTrack(np.zeros(0), np.zeros(0), np.zeros(80), TIMES)
    assert merge_sustained_boundaries(segments, bare, TIMES, MERGE) == segments


def test_the_sounding_track_survives_the_voicing_break_and_timbre_rebuilds():
    # the pitch plan's settle option was lost in a rebuild like these on 26 October
    n = 100
    times = np.arange(n) * 0.01
    sounding = np.linspace(0.0, 1.0, n)
    voicing = np.full(n, 0.95)
    voicing[50:54] = 0.0                                  # a short break: an unvoiced consonant
    cfg = TranscriptionConfig(device="cpu", split_on_voicing_breaks=True, timbre_split_ratio=2.0)
    audio = np.sin(2 * np.pi * 220 * np.arange(cfg.sample_rate) / cfg.sample_rate).astype(np.float32)
    onsets = _sounding(sounding, times, onset_times=[0.3])
    _, rebuilt = refine_segments([Segment(0, n, 60)], audio, times, cfg, onsets=onsets, voicing=voicing)
    assert rebuilt is not onsets and len(rebuilt) > len(onsets)
    np.testing.assert_array_equal(rebuilt.sounding, sounding)


def _track(pitches):
    """One frame per entry: a MIDI pitch, or None for silence."""
    hz = np.array([440.0 * 2 ** ((p - 69) / 12) if p else np.nan for p in pitches])
    voiced = np.isfinite(hz)
    return F0Track(times=np.arange(len(pitches)) * 0.01, f0_hz=hz, periodicity=np.where(voiced, 0.95, 0.0),
                   loudness_db=np.where(voiced, -20.0, -60.0), loudness_gate=voiced.astype(float),
                   voicing=np.where(voiced, 0.95, 0.0))


def test_the_decoder_rejoins_a_scoop_only_with_the_option_on():
    track = _track([None] * 10 + [61] * 20 + [64] * 45 + [None] * 15)
    sounding = np.where(np.arange(90) < 10, 0.0, 0.99)
    sounding[75:] = 0.0

    def notes(cfg, values):
        tr = OnsetTrack(np.zeros(0), np.zeros(0), np.zeros(90), track.times,
                        sounding=None if values is None else values)
        return [(round(n.onset, 2), n.pitch) for n in decode_notes(track, cfg, onsets=tr)[0]]

    base = TranscriptionConfig(device="cpu")
    alone = notes(base, None)
    assert [p for _, p in alone] == [61, 64]              # the decoder alone leaves the scoop as a note
    assert notes(base, sounding) == alone                 # a sounding track changes nothing while the option is off
    assert notes(dataclasses.replace(base, sustain_merge_above=0.98), sounding) == [(0.1, 64)]
