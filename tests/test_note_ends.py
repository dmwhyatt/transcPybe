"""Placing note ends at end times found elsewhere, such as a boundary network's offset peaks: end_snap_window_s."""
import dataclasses

import numpy as np

from transcpybe.config import TranscriptionConfig
from transcpybe.core import decode_notes
from transcpybe.f0 import F0Track
from transcpybe.notes import Note, snap_note_ends
from transcpybe.onsets import OnsetTrack, refine_segments
from transcpybe.segment import Segment

SNAP = TranscriptionConfig(device="cpu", end_snap_window_s=0.2)


def _note(onset, offset, pitch=60):
    return Note(onset=onset, offset=offset, pitch=pitch, velocity=80, confidence=0.9, loudness_db=-20.0,
                pitch_range_semitones=0.1, pitch_hz=261.6, pitch_midi_float=float(pitch), hmm_pitch=pitch,
                start_frame=int(round(onset / 0.01)), end_frame=int(round(offset / 0.01)))


def _ends(times, n=300):
    return OnsetTrack(np.zeros(0), np.zeros(0), np.zeros(n), np.arange(n) * 0.01, end_times=np.array(times, float))


def _spans(notes):
    return [(round(n.onset, 2), round(n.offset, 2)) for n in notes]


def test_a_late_end_moves_back_and_an_early_one_forward_to_the_nearest_end_time():
    notes = [_note(0.0, 0.8), _note(0.8, 1.2), _note(1.6, 1.9)]
    snapped = snap_note_ends(notes, _ends([0.55, 0.7, 2.05]), SNAP)
    # the first note ran on into the second: its end moves back to 0.7, the nearer of the two in reach;
    # the second has nothing within 0.2 s; the third stopped early before a rest
    assert _spans(snapped) == [(0.0, 0.7), (0.8, 1.2), (1.6, 2.05)]
    assert snapped[0].end_frame == 70 and snapped[0].pitch == notes[0].pitch


def test_an_end_never_passes_the_next_onset_nor_comes_too_close_to_its_own():
    # 0.58 is within both notes' windows, but past the first one's next onset (0.55), and nearer the second
    # one's own onset than the shortest note allowed (0.055 s)
    notes = [_note(0.0, 0.5), _note(0.55, 0.75)]
    snapped = snap_note_ends(notes, _ends([0.58]), SNAP)
    assert _spans(snapped) == [(0.0, 0.5), (0.55, 0.75)]
    assert _spans(snap_note_ends(notes, _ends([0.62]), SNAP)) == [(0.0, 0.5), (0.55, 0.62)]


def test_nothing_moves_with_the_option_off_or_without_end_times():
    notes = [_note(0.0, 0.8)]
    assert snap_note_ends(notes, _ends([0.7]), TranscriptionConfig(device="cpu")) == notes
    assert snap_note_ends(notes, _ends([]), SNAP) == notes
    assert snap_note_ends(notes, None, SNAP) == notes


def test_end_times_survive_the_voicing_break_and_timbre_rebuilds():
    n = 100
    times = np.arange(n) * 0.01
    voicing = np.full(n, 0.95)
    voicing[50:54] = 0.0
    cfg = TranscriptionConfig(device="cpu", split_on_voicing_breaks=True, timbre_split_ratio=2.0)
    audio = np.sin(2 * np.pi * 220 * np.arange(cfg.sample_rate) / cfg.sample_rate).astype(np.float32)
    onsets = OnsetTrack(np.array([0.3]), np.ones(1), np.zeros(n), times, end_times=np.array([0.9, 0.45]))
    _, rebuilt = refine_segments([Segment(0, n, 60)], audio, times, cfg, onsets=onsets, voicing=voicing)
    assert rebuilt is not onsets and rebuilt.end_times.tolist() == [0.45, 0.9]


def test_the_decoder_places_a_held_note_end_only_with_the_option_on():
    n = 120
    voiced = np.r_[np.zeros(10), np.ones(90), np.zeros(20)]
    track = F0Track(times=np.arange(n) * 0.01, f0_hz=np.where(voiced > 0, 220.0, np.nan),
                    periodicity=0.95 * voiced, loudness_db=np.where(voiced > 0, -20.0, -60.0),
                    loudness_gate=voiced, voicing=0.95 * voiced)
    base = TranscriptionConfig(device="cpu")

    def notes(cfg):
        tr = OnsetTrack(np.zeros(0), np.zeros(0), np.zeros(n), track.times, end_times=np.array([0.85]))
        return [(round(x.onset, 2), round(x.offset, 2), x.pitch) for x in decode_notes(track, cfg, onsets=tr)[0]]

    alone = notes(base)
    assert len(alone) == 1 and alone[0][1] >= 0.95
    assert notes(dataclasses.replace(base, end_snap_window_s=0.2)) == [(alone[0][0], 0.85, alone[0][2])]
