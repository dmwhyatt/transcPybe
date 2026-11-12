"""Short notes at a syllable a model is sure of: landmark_min_note_s.

A boundary network's paired landmarks (a consonant, then the vowel after it) mark a new syllable. With
``landmark_min_note_s`` above zero, a note starting at one of the onset track's ``protected_times`` escapes the three
short-note rules down to that length: the onset-split gap, fragment folding and the minimum note length. Anything
else, an ornament inside a syllable included, is held to the ordinary rules.
"""
import dataclasses

import numpy as np

from transcpybe.config import TranscriptionConfig
from transcpybe.core import decode_notes
from transcpybe.f0 import F0Track
from transcpybe.fragments import absorb_short_fragments
from transcpybe.notes import Note, drop_short_notes
from transcpybe.onsets import OnsetTrack, protected_frames, split_segments_on_onsets
from transcpybe.segment import Segment

TIMES = np.arange(100) * 0.01
# preset B's short-note rules, which the a cappella decoder inherits
B = TranscriptionConfig(device="cpu", min_note_duration_s=0.12, onset_split_min_gap_s=0.15,
                        fragment_max_duration_s=0.18, fragment_dominance_ratio=1.5)
ON = dataclasses.replace(B, landmark_min_note_s=0.06)


def _onsets(times, protected=None):
    times = np.asarray(times, float)
    return OnsetTrack(times, np.ones(times.size), np.zeros(TIMES.size), TIMES, protected_times=protected)


def _spans(segments):
    return [(s.start_frame, s.end_frame) for s in segments]


def _note(onset, offset):
    return Note(onset=onset, offset=offset, pitch=60, velocity=80, confidence=0.9, loudness_db=-20.0,
                pitch_range_semitones=0.0, pitch_hz=261.6, pitch_midi_float=60.0, hmm_pitch=60,
                start_frame=int(round(onset * 100)), end_frame=int(round(offset * 100)))


def test_a_cut_near_a_segment_end_needs_a_protected_start_and_the_option():
    segment = [Segment(0, 60, 60)]
    late = [0.5]                                          # 10 frames from the end: under the 15-frame split gap
    assert _spans(split_segments_on_onsets(segment, _onsets(late), TIMES, ON)) == [(0, 60)]
    assert _spans(split_segments_on_onsets(segment, _onsets(late, late), TIMES, B)) == [(0, 60)]
    assert _spans(split_segments_on_onsets(segment, _onsets(late, late), TIMES, ON)) == [(0, 50), (50, 60)]
    longer = dataclasses.replace(B, landmark_min_note_s=0.11)
    assert _spans(split_segments_on_onsets(segment, _onsets(late, late), TIMES, longer)) == [(0, 60)]


def test_a_segment_shorter_than_two_split_gaps_can_be_cut_only_at_a_protected_start():
    segment = [Segment(0, 25, 60)]                        # 0.25 s, under 2 x 0.15 s
    assert _spans(split_segments_on_onsets(segment, _onsets([0.12]), TIMES, ON)) == [(0, 25)]
    assert _spans(split_segments_on_onsets(segment, _onsets([0.12], [0.12]), TIMES, ON)) == [(0, 12), (12, 25)]


def test_an_unprotected_cut_keeps_the_ordinary_gap_beside_a_protected_one():
    segment = [Segment(0, 90, 60)]
    onsets = _onsets([0.08, 0.45], [0.08])                # 0.45 is far enough from both ends anyway
    assert _spans(split_segments_on_onsets(segment, onsets, TIMES, ON)) == [(0, 8), (8, 45), (45, 90)]
    onsets = _onsets([0.08, 0.80], [0.08])                # 0.80 is 10 frames from the end and not protected
    assert _spans(split_segments_on_onsets(segment, onsets, TIMES, ON)) == [(0, 8), (8, 90)]


def test_a_fragment_is_kept_only_with_a_protected_start_in_reach_and_enough_length():
    segments = [Segment(0, 40, 60), Segment(40, 52, 62)]  # 12 frames: a fragment, folded into its neighbour
    assert _spans(absorb_short_fragments(segments, B)) == [(0, 52)]
    assert _spans(absorb_short_fragments(segments, B, np.array([38]))) == [(0, 52)]   # option off
    assert _spans(absorb_short_fragments(segments, ON, np.array([38]))) == [(0, 40), (40, 52)]
    assert _spans(absorb_short_fragments(segments, ON, np.array([20]))) == [(0, 52)]  # 20 frames away: out of reach
    longer = dataclasses.replace(B, landmark_min_note_s=0.15)
    assert _spans(absorb_short_fragments(segments, longer, np.array([38]))) == [(0, 52)]


def test_protected_frames_are_empty_while_the_option_is_off():
    onsets = _onsets([0.2, 0.5], [0.5])
    assert protected_frames(onsets, TIMES, B).size == 0
    assert protected_frames(onsets, TIMES, ON).tolist() == [50]
    assert protected_frames(_onsets([0.2]), TIMES, ON).size == 0


def test_a_short_note_survives_only_at_a_protected_start():
    notes = [_note(0.10, 0.18), _note(0.50, 0.58), _note(0.80, 0.83)]     # 80, 80 and 30 ms
    onsets = _onsets([0.10, 0.51, 0.80], [0.10, 0.51, 0.80])
    assert drop_short_notes(notes, B, onsets) == []
    kept = drop_short_notes(notes, ON, onsets)
    assert [n.onset for n in kept] == [0.10, 0.50]        # one frame from 0.51 still counts; 30 ms is too short
    assert drop_short_notes(notes, ON, _onsets([0.10, 0.51])) == []        # starts not protected


def _track(pitches):
    hz = np.array([440.0 * 2 ** ((p - 69) / 12) if p else np.nan for p in pitches])
    voiced = np.isfinite(hz)
    return F0Track(times=np.arange(len(pitches)) * 0.01, f0_hz=hz, periodicity=np.where(voiced, 0.95, 0.0),
                   loudness_db=np.where(voiced, -20.0, -60.0), loudness_gate=voiced.astype(float),
                   voicing=np.where(voiced, 0.95, 0.0))


def test_the_decoder_keeps_a_short_syllable_only_with_the_option_on():
    track = _track([None] * 10 + [60] * 40 + [62] * 9 + [64] * 40 + [None] * 11)
    times = track.times
    starts = [0.10, 0.50, 0.59]

    def notes(cfg, protected):
        onsets = OnsetTrack(np.array(starts), np.ones(3), np.zeros(times.size), times, protected_times=protected)
        return [(round(n.onset, 2), n.pitch) for n in decode_notes(track, cfg, onsets=onsets)[0]]

    plain = notes(B, None)
    assert 62 not in [p for _, p in plain]                # the 90 ms note on 62 is lost
    assert notes(B, starts) == plain                      # protected times change nothing while the option is off
    assert notes(ON, None) == plain                       # nor does the option without protected times
    assert (0.5, 62) in notes(ON, starts)
