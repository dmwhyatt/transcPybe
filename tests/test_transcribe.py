"""Tests for the vocal melody transcriber.

End-to-end tests use the pYIN backend, which needs no torch and runs quickly.
The CREPE backend is exercised by a single test marked ``slow``; run the full
synthetic benchmark with ``python -m transcpybe --self-test`` to
evaluate it properly.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from transcpybe import (
    TranscriptionConfig,
    format_note_table,
    hz_to_midi,
    midi_to_hz,
    note_name,
    transcribe_audio,
    write_csv,
    write_json,
    write_midi,
)
from transcpybe.evaluate import score
from transcpybe.f0 import F0Track, estimate_f0
from transcpybe.fragments import absorb_short_fragments
from transcpybe.notes import Note, weighted_median, weighted_quantile
from transcpybe.segment import (
    Segment,
    build_transition_matrix,
    path_to_segments,
    pitch_range,
    segment_notes,
)
from transcpybe.synth import ALL_CASES, case_legato, case_scale


def pyin_config(**overrides) -> TranscriptionConfig:
    config = TranscriptionConfig(f0_backend="pyin", **overrides)
    config.validate()
    return config


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_default_config_is_valid():
    TranscriptionConfig().validate()


def test_hop_length_follows_sample_rate_and_hop_ms():
    assert TranscriptionConfig(sample_rate=16000, hop_ms=10.0).hop_length == 160
    assert TranscriptionConfig(sample_rate=44100, hop_ms=10.0).hop_length == 441


@pytest.mark.parametrize("overrides", [
    {"f0_backend": "nonsense"},
    {"crepe_model": "enormous"},
    {"fmin_hz": 500.0, "fmax_hz": 100.0},
    {"f0_median_filter_frames": 4},
    {"p_self": 1.5},
    {"p_self": 0.99, "p_voiced_to_unvoiced": 0.5},
    {"pitch_sigma_semitones": 0.0},
])
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        TranscriptionConfig(**overrides).validate()


def test_config_round_trips_through_dict():
    config = TranscriptionConfig(p_self=0.97, midi_program=53)
    assert TranscriptionConfig.from_dict(config.to_dict()) == config


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown config keys"):
        TranscriptionConfig.from_dict({"not_a_real_key": 1})


# --------------------------------------------------------------------------
# Pitch conversion
# --------------------------------------------------------------------------

def test_a440_is_midi_69():
    assert hz_to_midi(np.array([440.0]))[0] == pytest.approx(69.0)
    assert midi_to_hz(69.0) == pytest.approx(440.0)


def test_hz_to_midi_round_trips():
    midi = np.arange(21, 108, dtype=float)
    assert np.allclose(hz_to_midi(np.asarray(midi_to_hz(midi))), midi)


def test_hz_to_midi_rejects_non_positive_and_propagates_nan():
    out = hz_to_midi(np.array([0.0, -5.0, np.nan, np.inf]))
    assert np.isnan(out).all()


def test_note_name_uses_scientific_pitch_notation():
    assert note_name(60) == "C4"
    assert note_name(69) == "A4"
    assert note_name(21) == "A0"


# --------------------------------------------------------------------------
# Weighted statistics
# --------------------------------------------------------------------------

def test_weighted_median_matches_plain_median_with_equal_weights():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert weighted_median(values, np.ones(5)) == 3.0


def test_weighted_median_ignores_low_weight_outliers():
    values = np.array([60.0, 60.1, 72.0])
    assert weighted_median(values, np.array([1.0, 1.0, 0.01])) == pytest.approx(60.1)


def test_weighted_quantile_spans_the_range():
    values = np.arange(10.0)
    weights = np.ones(10)
    assert weighted_quantile(values, weights, 0.0) <= 1.0
    assert weighted_quantile(values, weights, 1.0) == 9.0


def test_weighted_statistics_return_nan_when_nothing_is_usable():
    assert np.isnan(weighted_median(np.array([np.nan, np.nan]), np.ones(2)))
    assert np.isnan(weighted_median(np.array([1.0, 2.0]), np.zeros(2)))


# --------------------------------------------------------------------------
# HMM
# --------------------------------------------------------------------------

def test_transition_matrix_rows_are_probability_distributions():
    config = TranscriptionConfig()
    log_a = build_transition_matrix(24, config)
    assert log_a.shape == (25, 25)
    assert np.allclose(np.exp(log_a).sum(axis=1), 1.0)


def test_transition_matrix_prefers_staying_put():
    log_a = build_transition_matrix(24, TranscriptionConfig())
    for state in range(1, 25):
        assert log_a[state, state] > log_a[state, 1:].max(initial=-np.inf, where=(
            np.arange(1, 25) != state))


def test_transition_matrix_prefers_small_intervals():
    log_a = build_transition_matrix(24, TranscriptionConfig())
    # From the middle state, a semitone step must beat an octave leap.
    assert log_a[12, 13] > log_a[12, 24]


def test_pitch_range_respects_configured_frequencies():
    lo, hi = pitch_range(TranscriptionConfig(fmin_hz=55.0, fmax_hz=1100.0))
    assert lo == 33 and hi == 85


def test_path_to_segments_groups_runs_and_drops_silence():
    path = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0])
    segments = path_to_segments(path, pitch_lo=60)
    assert [(s.start_frame, s.end_frame, s.pitch) for s in segments] == [
        (2, 5, 60), (6, 8, 61),
    ]


def test_viterbi_recovers_notes_from_a_vibrato_contour():
    """Three notes with +/-50 cent vibrato must decode as exactly three notes."""
    config = TranscriptionConfig()
    rng = np.random.default_rng(0)
    spans = [(30, 80, 60), (90, 140, 62), (150, 200, 60)]

    midi = np.full(230, np.nan)
    voicing = np.zeros(230)
    for start, end, pitch in spans:
        t = np.arange(end - start) / 100.0
        midi[start:end] = pitch + 0.5 * np.sin(2 * np.pi * 5.5 * t) + rng.normal(0, 0.05, end - start)
        voicing[start:end] = 0.95

    f0 = midi_to_hz(midi)
    track = F0Track(
        times=np.arange(230) / 100.0, f0_hz=f0, periodicity=voicing,
        loudness_db=np.where(voicing > 0, -20.0, -80.0),
        loudness_gate=np.where(voicing > 0, 1.0, 0.0), voicing=voicing,
    )

    segments = segment_notes(track, config)
    assert [s.pitch for s in segments] == [60, 62, 60]
    for segment, (start, end, _) in zip(segments, spans):
        assert abs(segment.start_frame - start) <= 2
        assert abs(segment.end_frame - end) <= 2


def _hand_track(midi, loudness_db=None):
    """An F0Track built from a fractional-MIDI contour; NaN is unvoiced."""
    midi = np.asarray(midi, dtype=float)
    voiced = np.isfinite(midi)
    voicing = np.where(voiced, 0.95, 0.0)
    loudness = (np.where(voiced, -20.0, -80.0) if loudness_db is None
                else np.asarray(loudness_db, dtype=float))
    return F0Track(
        times=np.arange(midi.shape[0]) / 100.0,
        f0_hz=np.where(voiced, midi_to_hz(np.nan_to_num(midi, nan=60.0)), np.nan),
        periodicity=voicing, loudness_db=loudness,
        loudness_gate=np.where(voiced, 1.0, 0.0), voicing=voicing,
    )


def test_a_note_sung_between_two_semitones_is_one_note():
    """Vibrato around x.5 flipped a one-state-per-semitone decoder.

    A note at 61.48 is equally well explained by 61 and 62, and +/-80 cents of
    vibrato hands it to each in turn several times a second. A singer with no
    reference pitch puts notes there routinely. On a third-of-a-semitone grid
    it has a state of its own.
    """
    rng = np.random.default_rng(2)
    t = np.arange(150) / 100.0
    midi = np.full(200, np.nan)
    midi[25:175] = (61.48 + 0.8 * np.sin(2 * np.pi * 5.8 * t)
                    + rng.normal(0.0, 0.05, 150))
    track = _hand_track(midi)

    coarse = segment_notes(track, TranscriptionConfig(pitch_states_per_semitone=1))
    fine = segment_notes(track, TranscriptionConfig())

    assert len(coarse) >= 5
    assert len(fine) == 1
    assert abs(fine[0].start_frame - 25) <= 2 and abs(fine[0].end_frame - 175) <= 2


def test_fine_grid_still_splits_a_narrow_semitone_step():
    """A finer grid must not make the decoder hold a compromise state.

    Two notes a semitone apart, sung 0.85 apart. A fine grid has states
    between them; if a change to any one sub-state cost more than a change to
    a semitone did, holding one of those in-between states across both notes
    would be cheaper than changing, and the step would vanish.
    """
    midi = np.full(160, np.nan)
    midi[20:60] = 64.1
    midi[60:140] = 64.95
    segments = segment_notes(_hand_track(midi), TranscriptionConfig())
    assert [s.pitch for s in segments] == [64, 65]
    assert abs(segments[1].start_frame - 60) <= 2


def test_fine_grid_does_not_split_a_drifting_note():
    """A note that drifts by a fifth of a semitone is still one note."""
    midi = np.full(160, np.nan)
    midi[20:140] = np.linspace(62.0, 62.2, 120)
    assert len(segment_notes(_hand_track(midi), TranscriptionConfig())) == 1


def test_one_state_per_semitone_is_the_old_decoder():
    config = TranscriptionConfig(pitch_states_per_semitone=1)
    log_a = build_transition_matrix(10, config)
    assert np.allclose(np.exp(log_a).sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# Glides
# --------------------------------------------------------------------------

def test_glide_backtrack_stops_at_a_note_held_off_its_semitone():
    """The walk must not march back through a held note.

    It used to stop only on arriving within 0.35 of the previous note's
    *integer* pitch. A note held at 60.4 never does, and sits inside the
    corridor up to 62, so the next onset was pulled 300 ms early and the held
    note cut short by the same amount.
    """
    from transcpybe.glide import backtrack_glide_onsets

    midi = np.full(160, np.nan)
    midi[10:90] = 60.4
    midi[90:150] = 62.0
    track = _hand_track(midi)
    segments = [Segment(10, 90, 60), Segment(90, 150, 62)]

    out = backtrack_glide_onsets(segments, track, TranscriptionConfig())
    assert abs(out[1].start_frame - 90) <= 2
    assert abs(out[0].end_frame - 90) <= 2


def test_glide_backtrack_still_finds_the_start_of_a_glide():
    from transcpybe.glide import backtrack_glide_onsets

    midi = np.full(160, np.nan)
    midi[10:80] = 60.0
    midi[80:95] = np.linspace(60.0, 62.0, 15)    # a 150 ms slide
    midi[95:150] = 62.0
    track = _hand_track(midi)
    segments = [Segment(10, 90, 60), Segment(90, 150, 62)]

    out = backtrack_glide_onsets(segments, track, TranscriptionConfig())
    assert out[1].start_frame <= 84


def test_a_short_note_between_rests_is_not_a_passing_tone():
    """A glide crossing touches both notes it joins; this one touches neither."""
    from transcpybe.glide import absorb_passing_tones

    segments = [Segment(0, 30, 60), Segment(45, 56, 62), Segment(71, 120, 64)]
    assert len(absorb_passing_tones(list(segments), TranscriptionConfig())) == 3
    # The same note with no rests around it is a glide crossing.
    contiguous = [Segment(0, 30, 60), Segment(30, 41, 62), Segment(41, 90, 64)]
    assert len(absorb_passing_tones(list(contiguous), TranscriptionConfig())) == 2




# --------------------------------------------------------------------------
# Fragment absorption
# --------------------------------------------------------------------------

def test_short_edge_fragment_is_absorbed_into_its_long_neighbour():
    config = TranscriptionConfig()
    segments = [Segment(0, 124, 69), Segment(124, 131, 68)]
    out = absorb_short_fragments(segments, config)
    assert len(out) == 1
    assert out[0].pitch == 69 and out[0].start_frame == 0 and out[0].end_frame == 131


def test_two_short_notes_side_by_side_are_not_merged():
    """Neither dominates the other, so a real melisma survives."""
    config = TranscriptionConfig()
    segments = [Segment(0, 14, 67), Segment(14, 28, 69)]
    assert len(absorb_short_fragments(segments, config)) == 2


def test_fragment_across_a_gap_is_not_absorbed():
    """Absorbing over a rest would falsify the neighbour's offset."""
    config = TranscriptionConfig()
    segments = [Segment(0, 124, 69), Segment(160, 167, 68)]
    assert len(absorb_short_fragments(segments, config)) == 2


def test_distant_pitch_is_not_absorbed():
    config = TranscriptionConfig()
    segments = [Segment(0, 124, 69), Segment(124, 131, 50)]
    assert len(absorb_short_fragments(segments, config)) == 2


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_scale_transcribes_exactly():
    audio, ground_truth = case_scale()
    result = transcribe_audio(audio, pyin_config())
    assert score(result.notes, ground_truth).note_f1 == 1.0


def test_legato_transcribes_exactly():
    audio, ground_truth = case_legato()
    result = transcribe_audio(audio, pyin_config())
    assert score(result.notes, ground_truth).note_f1 == 1.0


def test_pitches_are_integers_in_midi_range():
    audio, _ = case_scale()
    for note in transcribe_audio(audio, pyin_config()).notes:
        assert isinstance(note.pitch, int)
        assert 0 <= note.pitch <= 127
        assert 1 <= note.velocity <= 127


def test_notes_are_ordered_and_non_overlapping():
    audio, _ = case_scale()
    notes = transcribe_audio(audio, pyin_config()).notes
    for earlier, later in zip(notes, notes[1:]):
        assert earlier.onset <= later.onset
        assert earlier.offset <= later.onset + 1e-9


def test_onsets_are_floats_not_quantised():
    audio, _ = case_scale()
    onsets = [n.onset for n in transcribe_audio(audio, pyin_config()).notes]
    assert all(isinstance(o, float) for o in onsets)


def test_silence_produces_no_notes():
    audio = np.zeros(16000 * 2, dtype=np.float32)
    assert transcribe_audio(audio, pyin_config()).notes == []


def test_white_noise_produces_no_confident_melody():
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.1, 16000 * 2).astype(np.float32)
    notes = transcribe_audio(audio, pyin_config()).notes
    assert len(notes) <= 3


def test_very_short_input_does_not_crash():
    audio = np.zeros(800, dtype=np.float32)
    transcribe_audio(audio, pyin_config())


def test_quiet_note_rejection_can_be_disabled():
    audio, _ = case_scale()
    config = pyin_config(reject_quiet_notes=False)
    assert len(transcribe_audio(audio, config).notes) >= 15


@pytest.mark.slow
def test_crepe_backend_transcribes_the_scale():
    pytest.importorskip("torchcrepe")
    audio, ground_truth = case_scale()
    config = TranscriptionConfig(f0_backend="crepe", crepe_model="tiny")
    assert score(transcribe_audio(audio, config).notes, ground_truth).note_f1 > 0.9


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def test_midi_round_trips_with_identical_pitches_and_times(tmp_path):
    pretty_midi = pytest.importorskip("pretty_midi")
    audio, _ = case_scale()
    config = pyin_config()
    result = transcribe_audio(audio, config)

    path = write_midi(result.notes, tmp_path / "out.mid", config)
    reloaded = pretty_midi.PrettyMIDI(str(path)).instruments[0].notes

    assert len(reloaded) == len(result.notes)
    for written, original in zip(reloaded, result.notes):
        assert written.pitch == original.pitch
        assert written.start == pytest.approx(original.onset, abs=0.002)


def test_json_and_csv_carry_every_note(tmp_path):
    import csv
    import json

    audio, _ = case_scale()
    config = pyin_config()
    result = transcribe_audio(audio, config)

    payload = json.loads(write_json(result.notes, tmp_path / "n.json", config).read_text())
    assert payload["n_notes"] == len(result.notes)
    assert payload["config"]["f0_backend"] == "pyin"

    rows = list(csv.DictReader(write_csv(result.notes, tmp_path / "n.csv").open()))
    assert len(rows) == len(result.notes)
    assert int(rows[0]["pitch"]) == result.notes[0].pitch


def test_empty_note_list_still_writes_readable_files(tmp_path):
    """A track with no detected melody must still produce parseable output.

    pretty_midi omits an instrument that carries no events, so the resulting
    file has no instrument track at all. That is valid MIDI and is the
    behaviour downstream consumers see, so it is what is asserted here.
    """
    pretty_midi = pytest.importorskip("pretty_midi")
    import json

    config = TranscriptionConfig()
    midi_path = write_midi([], tmp_path / "e.mid", config)
    json_path = write_json([], tmp_path / "e.json", config)
    csv_path = write_csv([], tmp_path / "e.csv")

    assert midi_path.exists() and json_path.exists() and csv_path.exists()
    reloaded = pretty_midi.PrettyMIDI(str(midi_path))
    assert sum(len(i.notes) for i in reloaded.instruments) == 0
    assert json.loads(json_path.read_text())["n_notes"] == 0


def test_note_table_renders_without_error():
    audio, _ = case_scale()
    table = format_note_table(transcribe_audio(audio, pyin_config()).notes)
    assert "pitch" in table and "C4" in table


# --------------------------------------------------------------------------
# Synthetic suite integrity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ALL_CASES))
def test_synthetic_cases_are_well_formed(name):
    audio, ground_truth = ALL_CASES[name]()
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert len(ground_truth) > 0
    for onset, offset, pitch in ground_truth:
        assert offset > onset
        assert 0 <= pitch <= 127
    for earlier, later in zip(ground_truth, ground_truth[1:]):
        assert later[0] >= earlier[0]


# ---------------------------------------------------------------------------
# Re-articulated notes.
#
# A singer moving to a new syllable without stopping phonation leaves the pitch
# unchanged and the sound continuous, so only the onset detector can find the
# boundary. Eight such notes were decoded as one: the split gate compared onset
# peak strengths against a percentile of the onset *envelope*, which counts the
# loud frames between the peaks, so the bar sat above most real onsets.
# ---------------------------------------------------------------------------


def test_rearticulated_notes_are_separated():
    """Same pitch, no silence between notes: the pitch decoder cannot see these."""
    from transcpybe.synth import case_rearticulated

    audio, ground_truth = case_rearticulated()
    notes = transcribe_audio(audio, pyin_config()).notes

    assert len(notes) == len(ground_truth) == 8
    assert all(abs(n.pitch - 62) <= 0 for n in notes)


def test_fast_rearticulation_is_separated():
    """170 ms syllables, which the old 100 ms gap guard could not split apart."""
    from transcpybe.synth import case_rearticulated_fast

    audio, ground_truth = case_rearticulated_fast()
    notes = transcribe_audio(audio, pyin_config()).notes

    assert len(notes) == len(ground_truth) == 10


def test_sustained_notes_still_do_not_fragment():
    """The guard the looser gate must not cost: vibrato is not a series of onsets."""
    from transcpybe.synth import case_vibrato, case_heavy_vibrato

    for make_case in (case_vibrato, case_heavy_vibrato):
        audio, ground_truth = make_case()
        notes = transcribe_audio(audio, pyin_config()).notes
        assert len(notes) == len(ground_truth)


def test_split_threshold_comes_from_onsets_not_the_envelope():
    """Pins the bug directly, without going through the whole pipeline."""
    import numpy as np
    from transcpybe.onsets import OnsetTrack, split_segments_on_onsets
    from transcpybe.segment import Segment

    config = pyin_config()
    frame_times = np.arange(0, 3.0, config.hop_ms / 1000.0)
    # One continuous segment carrying four evenly spaced onsets of ordinary
    # strength, sitting on an envelope whose between-peak frames are louder
    # still -- which is what continuous singing looks like.
    envelope = np.full(frame_times.shape[0], 8.0)
    times = np.array([0.6, 1.2, 1.8, 2.4])
    strengths = np.array([10.0, 9.0, 11.0, 9.5])
    onsets = OnsetTrack(times, strengths, envelope, frame_times)
    segments = [Segment(start_frame=0, end_frame=frame_times.shape[0] - 1, pitch=62)]

    out = split_segments_on_onsets(segments, onsets, frame_times, config)

    assert len(out) == 5, "four onsets inside one segment must yield five notes"


# ---------------------------------------------------------------------------
# Note ends.
#
# Nothing in the pipeline used to touch them. Every other refinement stage acts
# on note starts, and the end was wherever the Viterbi path left the pitch
# state -- which the adaptive gate holds open until a frame is 45 dB below the
# whole performance. A note with a release was reported for as long as it
# remained audible rather than for as long as it was a note.
# ---------------------------------------------------------------------------


def test_release_is_not_counted_as_note():
    """Sound continuing past the annotation must not extend the note.

    Opt-in, because the trim is off by default: it ends notes at the top of the
    decay, and vocadito's annotators mark them later -- enabling it cost 0.161
    COnPOff there, on all forty recordings. What this test pins is that the
    mechanism works, not that the default should use it.
    """
    from transcpybe.synth import case_release_tails

    audio, ground_truth = case_release_tails()
    notes = transcribe_audio(audio, pyin_config(trim_note_releases=True)).notes

    assert len(notes) == len(ground_truth)
    for note, (_, true_end, _) in zip(notes, ground_truth):
        # The tail is 180 ms; tolerance is 20% of the note or 50 ms, whichever
        # is larger. Untrimmed, every one of these overshoots by the full tail.
        assert note.offset - true_end < 0.11


def test_breath_after_a_note_is_not_part_of_it():
    from transcpybe.synth import case_breath_ends

    audio, ground_truth = case_breath_ends()
    notes = transcribe_audio(audio, pyin_config(trim_note_releases=True)).notes

    assert len(notes) == len(ground_truth)
    for note, (_, true_end, _) in zip(notes, ground_truth):
        assert abs(note.offset - true_end) <= 0.10


def test_release_trim_only_ever_shortens():
    """A decoder that already ended a note early must be left alone."""
    from transcpybe.notes import trim_note_releases

    config = pyin_config(trim_note_releases=True)
    audio, _ = case_scale()
    result = transcribe_audio(audio, pyin_config(trim_note_releases=False))
    untrimmed = result.notes
    trimmed = trim_note_releases(list(untrimmed), result.f0_track, config)

    assert len(trimmed) == len(untrimmed)
    for before, after in zip(untrimmed, trimmed):
        assert after.offset <= before.offset
        assert after.onset == before.onset
        assert after.duration >= config.min_note_duration_s


def test_release_trim_is_off_by_default():
    """The default must leave note ends where the decoder put them.

    Trimming them scored better on every synthetic case and worse on every one
    of vocadito's forty recordings, so the default follows the real corpus.
    """
    from transcpybe.synth import case_release_tails

    assert TranscriptionConfig().trim_note_releases is False

    audio, ground_truth = case_release_tails()
    kept = transcribe_audio(audio, pyin_config()).notes

    overshoot = [n.offset - end for n, (_, end, _) in zip(kept, ground_truth)]
    assert min(overshoot) > 0.11, "untrimmed, these run long -- and that is the default"


# ---------------------------------------------------------------------------
# Onset envelope.
#
# Three librosa defaults were being inherited. At 16 kHz n_fft=2048 is a 128 ms
# window; median aggregation across 128 mel bands only fires when half the
# spectrum rises at once; and a 3-band max filter is too narrow to be the
# vibrato defence the other two were relying on.
# ---------------------------------------------------------------------------


def test_vowel_rearticulation_is_found_without_a_consonant():
    """The commoner case: a syllable boundary with no transient on it."""
    from transcpybe.synth import case_soft_rearticulation

    audio, ground_truth = case_soft_rearticulation()
    notes = transcribe_audio(audio, pyin_config()).notes

    assert len(notes) == len(ground_truth) == 8


def test_repeated_pitches_in_a_legato_phrase_are_separated():
    from transcpybe.synth import case_soft_legato_repeats

    audio, ground_truth = case_soft_legato_repeats()
    notes = transcribe_audio(audio, pyin_config()).notes

    assert len(notes) == len(ground_truth) == 8


def test_onset_window_is_set_rather_than_inherited():
    """librosa's 2048 default is 128 ms here, and smears the attack away."""
    from transcpybe.onsets import detect_onsets

    config = pyin_config()
    assert config.onset_n_fft == 512

    audio, ground_truth = case_scale()
    sharp = detect_onsets(audio, config)
    blunt = detect_onsets(audio, pyin_config(onset_n_fft=2048,
                                             onset_aggregate="median",
                                             onset_max_size=3))

    assert len(sharp) >= len(ground_truth)
    assert len(blunt) < len(ground_truth)


def test_peak_threshold_scales_with_the_envelope():
    """An absolute delta cannot serve an envelope whose scale spans decades."""
    from transcpybe.onsets import detect_onsets
    from transcpybe.synth import case_breathy

    audio, ground_truth = case_breathy()
    strict = detect_onsets(audio, pyin_config(onset_delta_ratio=3.0))
    loose = detect_onsets(audio, pyin_config(onset_delta_ratio=1e-9))

    assert len(loose) > 3 * len(ground_truth)
    assert len(strict) < len(loose)


def test_vibrato_suppression_buys_onset_recall():
    """What the max filter is for, once the peak threshold is relative.

    A narrow filter lets a wobbling sustain register as flux. That does not
    show up as spurious splits, because the threshold scales with the
    envelope's median and the extra flux simply raises it -- it shows up as
    genuine onsets falling underneath the raised bar.
    """
    from transcpybe.synth import case_heavy_vibrato, case_rearticulated

    audio, ground_truth = case_heavy_vibrato()
    assert len(transcribe_audio(audio, pyin_config()).notes) == len(ground_truth)

    audio, ground_truth = case_rearticulated()
    assert len(transcribe_audio(audio, pyin_config()).notes) == len(ground_truth)
    narrow = transcribe_audio(audio, pyin_config(onset_max_size=1)).notes
    assert len(narrow) < len(ground_truth), "a 1-band filter should lose onsets"


# --------------------------------------------------------------------------
# Tuning reference
# --------------------------------------------------------------------------

def _detuned_phrase(offset_cents, scatter_cents=0.0, seed=5):
    """A phrase sung ``offset_cents`` away from the grid, plus scatter."""
    from transcpybe.synth import _sequence, synth_phrase

    pitches = [62, 64, 66, 67, 69, 67, 66, 64, 62, 60, 62, 64]
    rng = np.random.default_rng(seed)
    detune = [offset_cents + rng.normal(0.0, scatter_cents) for _ in pitches]
    return synth_phrase(_sequence(pitches, 0.5, 0.15), 16000,
                        vibrato_cents=20.0, detune_cents=detune, seed=seed)


def test_tuning_offset_is_found_and_reported():
    """A phrase sung a third of a semitone flat is recognised as such."""
    audio, _ = _detuned_phrase(-33.0)
    result = transcribe_audio(audio, pyin_config())

    assert result.tuning_offset == pytest.approx(-0.33, abs=0.08)


def test_a_phrase_on_the_grid_is_left_alone():
    """No offset means no correction, and no second decoding pass."""
    audio, _ = _detuned_phrase(0.0)
    result = transcribe_audio(audio, pyin_config())

    assert result.tuning_offset == 0.0


def test_tuning_correction_recovers_notes_lost_to_rounding():
    """The point of the whole exercise: the notes the singer meant.

    A third of a semitone off the grid, with ordinary intonation scatter on
    top, is enough to round a share of notes to the neighbouring semitone.
    """
    audio, truth = _detuned_phrase(35.0, scatter_cents=15.0, seed=11)

    corrected = score(transcribe_audio(audio, pyin_config()).notes, truth)
    forced = score(
        transcribe_audio(audio, pyin_config(estimate_tuning=False)).notes, truth)

    assert corrected.note_f1 > forced.note_f1


def test_scattered_intonation_is_not_a_tuning_offset():
    """Singing that wanders either side of the grid must not be re-gridded.

    This is the failure that makes the correction dangerous rather than merely
    useless. Deviations spread symmetrically about zero sit, on a circle one
    semitone around, nearest the point half a semitone away -- so a scattered
    singer looks like a well-tuned one half a semitone off, and shifting them
    there moves every sharp note into the next semitone up.
    """
    audio, truth = ALL_CASES["mistuned"]()
    config = pyin_config()
    result = transcribe_audio(audio, config)

    assert result.tuning_offset == 0.0
    assert score(result.notes, truth).note_f1 == pytest.approx(
        score(transcribe_audio(audio, pyin_config(estimate_tuning=False)).notes,
              truth).note_f1)


def test_vibrato_does_not_masquerade_as_a_tuning_offset():
    """Measured per frame rather than per note, this case reports 49 cents."""
    audio, _ = ALL_CASES["vibrato"]()

    assert transcribe_audio(audio, pyin_config()).tuning_offset == 0.0


def test_tuning_estimate_can_be_switched_off():
    audio, _ = _detuned_phrase(-33.0)

    assert transcribe_audio(
        audio, pyin_config(estimate_tuning=False)).tuning_offset == 0.0


def test_offsets_past_a_quarter_tone_are_refused():
    from transcpybe.notes import estimate_tuning_offset

    config = pyin_config()
    notes = [
        Note(onset=i * 0.5, offset=i * 0.5 + 0.4, pitch=60, velocity=80,
             confidence=0.9, loudness_db=-12.0, pitch_range_semitones=0.1,
             pitch_hz=261.0, pitch_midi_float=60.0 + 0.45, hmm_pitch=60,
             start_frame=0, end_frame=40)
        for i in range(8)
    ]
    assert estimate_tuning_offset(notes, config) == 0.0

    within = [dataclasses.replace(n, pitch_midi_float=60.3) for n in notes]
    assert estimate_tuning_offset(within, config) == pytest.approx(0.3, abs=0.02)


def test_tuning_estimate_needs_enough_notes():
    from transcpybe.notes import estimate_tuning_offset

    config = pyin_config()
    notes = [
        Note(onset=i * 0.5, offset=i * 0.5 + 0.4, pitch=60, velocity=80,
             confidence=0.9, loudness_db=-12.0, pitch_range_semitones=0.1,
             pitch_hz=261.0, pitch_midi_float=60.3, hmm_pitch=60,
             start_frame=0, end_frame=40)
        for i in range(config.tuning_min_notes - 1)
    ]
    assert estimate_tuning_offset(notes, config) == 0.0


def test_self_test_uses_the_requested_backend():
    """--self-test built its own default config and ignored every flag.

    That made the one command offered for checking an install fail with an
    ImportError on the torch-free install the pyin flag exists to serve.
    """
    from transcpybe.cli import build_parser, config_from_args

    args = build_parser().parse_args(["--self-test", "--f0-backend", "pyin"])
    assert config_from_args(args).f0_backend == "pyin"
