"""Tests for the transcription benchmark suite.

The suite exists so a number can be defended, which means its own failure modes
have to be the loud kind. Two are covered here in particular: an annotation
format read the wrong way round, and a reference that is not aligned to its
audio. Both produce a plausible low score rather than an error, and both would
otherwise be blamed on the transcriber.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from transcpybe.benchmark.annotations import load_annotation
from transcpybe.benchmark.dataset import (
    dataset_digest,
    file_digest,
    pair_dataset,
)
from transcpybe.benchmark.metrics import (
    aggregate_scores,
    score_item,
)

REFERENCE = [(0.0, 0.5, 60), (1.0, 1.5, 62), (2.0, 2.5, 64),
             (3.0, 3.5, 65), (4.0, 4.5, 67)]


# --------------------------------------------------------------------------
# Annotation formats
# --------------------------------------------------------------------------

EXPECTED = [(0.5, 0.9, 60), (1.1, 1.45, 62), (1.6, 2.1, 64)]


def _approx(notes):
    """Times to the millisecond, pitch to the nearest note.

    Reference pitches keep their fractional part so that scoring happens in
    cents, so a Hz round trip leaves floating-point noise on an exact semitone.
    """
    return [(round(a, 3), round(b, 3), round(p)) for a, b, p in notes]


def test_csv_with_header_and_durations(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("onset_sec,duration_sec,midi_note\n0.5,0.4,60\n1.1,0.35,62\n1.6,0.5,64\n")
    annotation = load_annotation(path)
    assert _approx(annotation.notes) == EXPECTED
    assert annotation.inferences["second_column"].startswith("duration")


def test_tab_separated_with_hz_pitches(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("0.5\t0.9\t261.63\n1.1\t1.45\t293.66\n1.6\t2.1\t329.63\n")
    annotation = load_annotation(path)
    assert _approx(annotation.notes) == EXPECTED
    assert annotation.inferences["pitch_units"].startswith("hz")


def test_json_list_of_triples(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps([[0.5, 0.9, 60], [1.1, 1.45, 62], [1.6, 2.1, 64]]))
    assert _approx(load_annotation(path).notes) == EXPECTED


def test_json_objects_with_alternate_keys(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"notes": [
        {"start": 0.5, "end": 0.9, "midi_pitch": 60},
        {"start": 1.1, "end": 1.45, "midi_pitch": 62},
        {"start": 1.6, "end": 2.1, "midi_pitch": 64},
    ]}))
    assert _approx(load_annotation(path).notes) == EXPECTED


def test_midi_reference(tmp_path):
    pretty_midi = pytest.importorskip("pretty_midi")
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for onset, offset, pitch in EXPECTED:
        instrument.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch, start=onset, end=offset))
    midi.instruments.append(instrument)
    path = tmp_path / "f.mid"
    midi.write(str(path))
    assert _approx(load_annotation(path).notes) == EXPECTED


def test_jams_reference(tmp_path):
    path = tmp_path / "g.jams"
    path.write_text(json.dumps({"annotations": [{
        "namespace": "note_midi",
        "data": [{"time": a, "duration": b - a, "value": p} for a, b, p in EXPECTED],
    }]}))
    assert _approx(load_annotation(path).notes) == EXPECTED


def test_one_corrupt_row_does_not_reinterpret_the_whole_file(tmp_path):
    """A single out-of-range pitch must not turn a MIDI column into Hz.

    That mistake shifts every reference note down by octaves, and the resulting
    score looks like a broken transcriber rather than a misread file.
    """
    path = tmp_path / "e.csv"
    path.write_text(
        "onset,offset,pitch\n0.5,0.9,60\n\nbroken,row\n"
        "1.1,1.45,62\n2.0,2.0,64\n3.0,3.5,200\n"
    )
    annotation = load_annotation(path)
    assert annotation.inferences["pitch_units"].startswith("midi")
    assert annotation.inferences["second_column"] == "offset"
    assert _approx(annotation.notes) == [(0.5, 0.9, 60), (1.1, 1.45, 62)]
    assert len(annotation.warnings) >= 2


def test_a_genuine_hz_column_is_still_detected(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("\n".join(f"{i * 0.5},{i * 0.5 + 0.4},{200 + i * 10}" for i in range(20)))
    assert load_annotation(path).inferences["pitch_units"].startswith("hz")


def test_unsupported_format_is_rejected(tmp_path):
    path = tmp_path / "x.xml"
    path.write_text("<notes/>")
    with pytest.raises(ValueError, match="Unsupported annotation format"):
        load_annotation(path)


def test_missing_annotation_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_annotation(tmp_path / "nope.csv")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_perfect_transcription_scores_one():
    score = score_item("x", REFERENCE, REFERENCE)
    assert score.con.f == 1.0 and score.conp.f == 1.0 and score.conpoff.f == 1.0
    assert score.octave_errors == 0
    assert score.best_offset_s == pytest.approx(0.0, abs=1e-9)


def test_metrics_nest():
    """A note counted in COnPOff must be counted in COnP and COn."""
    estimated = [(0.02, 0.9, 60), (1.03, 1.4, 63), (2.0, 2.5, 64)]
    score = score_item("x", REFERENCE, estimated)
    assert score.con.f >= score.conp.f >= score.conpoff.f


def test_octave_errors_are_counted():
    estimated = [(0.0, 0.5, 72)] + REFERENCE[1:]
    score = score_item("x", REFERENCE, estimated)
    assert score.octave_errors == 1
    assert score.con.f == 1.0, "the onset was still found"
    assert score.conp.f < 1.0, "but the pitch was wrong"


@pytest.mark.parametrize("shift", [0.2, -0.15, 0.35])
def test_a_misaligned_reference_is_detected_and_measured(shift):
    """The failure this exists to catch: a constant offset between reference
    and audio, which scores near zero and looks like a broken transcriber."""
    estimated = [(a + shift, b + shift, p) for a, b, p in REFERENCE]
    score = score_item("x", REFERENCE, estimated)

    assert score.conp.f == 0.0, "as scored, the misalignment destroys the result"
    assert score.best_offset_s == pytest.approx(-shift, abs=0.011)
    assert score.best_offset_conp_f == 1.0
    assert score.misalignment_gain > 0.9


def test_empty_inputs_score_zero_without_raising():
    assert score_item("x", REFERENCE, []).conp.f == 0.0
    assert score_item("x", [], REFERENCE).conp.f == 0.0
    assert score_item("x", [], []).conp.f == 0.0


def test_negative_times_do_not_crash_the_scorer():
    """mir_eval rejects them outright; the scorer must not pass them through."""
    estimated = [(-0.2, 0.3, 60), (0.8, 1.3, 62)]
    score_item("x", REFERENCE, estimated, check_alignment=False)


# --------------------------------------------------------------------------
# Aggregation and confidence intervals
# --------------------------------------------------------------------------

def _scores(n, keep):
    return [
        score_item(f"i{i}", REFERENCE, REFERENCE[:keep], check_alignment=False)
        for i in range(n)
    ]


def test_confidence_interval_brackets_the_mean():
    aggregate = aggregate_scores(_scores(20, 4), seed=1)
    interval = aggregate.macro["conp_f"]
    assert interval.low <= interval.mean <= interval.high


def test_confidence_interval_narrows_with_more_recordings():
    """The whole point of the interval: a benchmark of ten says less than one
    of two hundred, and the number must show it."""
    rng = np.random.default_rng(0)

    def varied(n):
        out = []
        for i in range(n):
            keep = int(rng.integers(1, len(REFERENCE) + 1))
            out.append(score_item(f"i{i}", REFERENCE, REFERENCE[:keep],
                                  check_alignment=False))
        return out

    narrow = aggregate_scores(varied(200), seed=1).macro["conp_f"]
    wide = aggregate_scores(varied(10), seed=1).macro["conp_f"]
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


def test_aggregation_is_reproducible_for_a_fixed_seed():
    scores = _scores(20, 3)
    first = aggregate_scores(scores, seed=7).macro["conp_f"]
    second = aggregate_scores(scores, seed=7).macro["conp_f"]
    assert (first.low, first.high) == (second.low, second.high)


def test_misaligned_items_are_named_in_the_aggregate():
    good = score_item("good", REFERENCE, REFERENCE)
    shifted = [(a + 0.3, b + 0.3, p) for a, b, p in REFERENCE]
    bad = score_item("bad", REFERENCE, shifted)
    aggregate = aggregate_scores([good, bad], seed=0)
    assert aggregate.items_needing_alignment == ["bad"]


def test_empty_aggregate_does_not_raise():
    assert aggregate_scores([]).n_items == 0


def test_macro_and_micro_are_both_reported():
    aggregate = aggregate_scores(_scores(5, 3), seed=0)
    assert "conp_f" in aggregate.macro and "conp_f" in aggregate.micro


# --------------------------------------------------------------------------
# Dataset pairing
# --------------------------------------------------------------------------

def _make_audio(path: Path):
    soundfile = pytest.importorskip("soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, np.zeros(1600, dtype=np.float32), 16000)


def test_pairs_annotations_sharing_the_stem(tmp_path):
    for name in ("one", "two"):
        _make_audio(tmp_path / f"{name}.wav")
        (tmp_path / f"{name}.csv").write_text("0,1,60\n")
    pairing = pair_dataset(tmp_path)
    assert len(pairing.items) == 2
    assert all(item.match == "exact" for item in pairing.items)


def test_pairs_across_a_dataset_naming_tag(tmp_path):
    """vocadito ships vocadito_1.wav beside vocadito_1_notes.csv."""
    _make_audio(tmp_path / "vocadito_1.wav")
    (tmp_path / "vocadito_1_notes.csv").write_text("0,1,60\n")
    pairing = pair_dataset(tmp_path)
    assert len(pairing.items) == 1
    assert pairing.items[0].match == "tag-stripped"


def test_pairs_across_parallel_directories(tmp_path):
    _make_audio(tmp_path / "audio" / "song.wav")
    (tmp_path / "annot").mkdir()
    (tmp_path / "annot" / "song.txt").write_text("0\t1\t60\n")
    pairing = pair_dataset(tmp_path / "audio", tmp_path / "annot")
    assert len(pairing.items) == 1


def test_unpaired_audio_is_reported_not_dropped_silently(tmp_path):
    """A benchmark that scores half its files and says nothing is worse than
    one that fails."""
    _make_audio(tmp_path / "paired.wav")
    (tmp_path / "paired.csv").write_text("0,1,60\n")
    _make_audio(tmp_path / "lonely.wav")

    pairing = pair_dataset(tmp_path)
    assert len(pairing.items) == 1
    assert [p.name for p in pairing.unpaired_audio] == ["lonely.wav"]
    assert "1 audio files had no annotation" in pairing.summary()


def test_unused_annotations_are_reported(tmp_path):
    _make_audio(tmp_path / "a.wav")
    (tmp_path / "a.csv").write_text("0,1,60\n")
    (tmp_path / "orphan.csv").write_text("0,1,60\n")
    assert len(pair_dataset(tmp_path).unused_annotations) == 1


def test_dataset_digest_changes_when_a_file_changes(tmp_path):
    """A score is tied to exactly the files that produced it."""
    _make_audio(tmp_path / "a.wav")
    annotation = tmp_path / "a.csv"
    annotation.write_text("0,1,60\n")

    before = dataset_digest(pair_dataset(tmp_path).items)
    annotation.write_text("0,1,62\n")
    after = dataset_digest(pair_dataset(tmp_path).items)
    assert before != after


def test_dataset_digest_is_stable_across_calls(tmp_path):
    _make_audio(tmp_path / "a.wav")
    (tmp_path / "a.csv").write_text("0,1,60\n")
    items = pair_dataset(tmp_path).items
    assert dataset_digest(items) == dataset_digest(items)


def test_file_digest_is_deterministic(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc" * 1000)
    assert file_digest(path) == file_digest(path)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_benchmark_runs_end_to_end_and_records_provenance(tmp_path):
    from transcpybe.benchmark.runner import format_result, run_benchmark
    from transcpybe.config import TranscriptionConfig
    from transcpybe.synth import case_legato, case_scale

    soundfile = pytest.importorskip("soundfile")
    for name, make_case in (("scale", case_scale), ("legato", case_legato)):
        audio, ground_truth = make_case()
        soundfile.write(tmp_path / f"{name}.wav", audio, 16000)
        rows = ["onset,offset,pitch"] + [
            f"{a:.6f},{b:.6f},{p}" for a, b, p in ground_truth
        ]
        (tmp_path / f"{name}.csv").write_text("\n".join(rows) + "\n")

    pairing = pair_dataset(tmp_path)
    result = run_benchmark(
        pairing, TranscriptionConfig(f0_backend="pyin"),
        bootstrap_samples=200, check_alignment=False,
    )

    assert result.aggregate.n_items == 2
    assert result.aggregate.macro["conp_f"].mean > 0.9, "pyin handles these cases"
    assert result.provenance.dataset_digest
    assert result.provenance.config["f0_backend"] == "pyin"
    assert "torch" in result.provenance.packages
    assert "COnP" in format_result(result)
    json.dumps(result.to_dict())          # must be serialisable


def test_transcription_cache_is_keyed_on_the_configuration(tmp_path):
    """Changing a parameter must re-transcribe, not score the previous settings."""
    from transcpybe.benchmark.runner import run_benchmark
    from transcpybe.config import TranscriptionConfig

    soundfile = pytest.importorskip("soundfile")
    from transcpybe.synth import case_scale

    audio, ground_truth = case_scale()
    soundfile.write(tmp_path / "scale.wav", audio, 16000)
    rows = ["onset,offset,pitch"] + [f"{a},{b},{p}" for a, b, p in ground_truth]
    (tmp_path / "scale.csv").write_text("\n".join(rows) + "\n")

    pairing = pair_dataset(tmp_path)
    cache = tmp_path / "cache"

    run_benchmark(pairing, TranscriptionConfig(f0_backend="pyin"),
                  cache_dir=cache, bootstrap_samples=10, check_alignment=False)
    cached = list(cache.glob("*.json"))
    assert len(cached) == 1, "one recording, one cache entry"
    first_fingerprint = json.loads(cached[0].read_text())["fingerprint"]

    run_benchmark(pairing, TranscriptionConfig(f0_backend="pyin", min_note_duration_s=0.2),
                  cache_dir=cache, bootstrap_samples=10, check_alignment=False)
    second = json.loads(cached[0].read_text())
    assert second["fingerprint"] != first_fingerprint


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------

def test_dry_run_reports_pairing_and_interpretation_without_transcribing(tmp_path, capsys):
    """The check to run before committing to a long benchmark."""
    from transcpybe.benchmark.cli import main

    _make_audio(tmp_path / "song_01.wav")
    (tmp_path / "song_01_notes.csv").write_text("onset,offset,pitch\n0.5,0.9,60\n1.1,1.5,62\n")

    assert main([str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "1 items paired" in out
    assert "tag-stripped" in out
    assert "pitch_units=midi" in out
    assert "paired and read cleanly" in out


def test_dry_run_flags_unpaired_audio_rather_than_reassuring(tmp_path, capsys):
    from transcpybe.benchmark.cli import main

    _make_audio(tmp_path / "paired.wav")
    (tmp_path / "paired.csv").write_text("0,1,60\n")
    _make_audio(tmp_path / "lonely.wav")

    assert main([str(tmp_path), "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "no annotation" in out
    assert "worth checking" in out
    assert "read cleanly" not in out


def test_low_frequencies_are_resolved_not_merely_surfaced(tmp_path, capsys):
    """Frequencies too low to clear 127 are read as Hz, not shown as a problem.

    The dry-run used to report these as an implausible MIDI range and leave the
    reader to notice. Resolving them is better than surfacing them.
    """
    from transcpybe.benchmark.cli import main

    _make_audio(tmp_path / "s.wav")
    (tmp_path / "s.csv").write_text(
        "\n".join(f"{i * 0.5},{i * 0.5 + 0.3},{100 + i}" for i in range(10))
    )
    main([str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert "pitch_units=hz" in out
    assert "100-109" not in out


def test_dry_run_flags_a_pitch_column_it_cannot_rescue(tmp_path, capsys):
    """Values singable under neither reading are named, not quietly scored."""
    from transcpybe.benchmark.cli import main

    _make_audio(tmp_path / "s.wav")
    # 5-14: far below a voice as MIDI, and subsonic as Hz.
    (tmp_path / "s.csv").write_text(
        "\n".join(f"{i * 0.5},{i * 0.5 + 0.3},{5 + i}" for i in range(10))
    )
    exit_code = main([str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert "outside any sung range" in out
    assert exit_code == 1, "an unrescuable pitch column must not report a clean read"


def test_limit_restricts_how_many_items_are_scored(tmp_path, capsys):
    from transcpybe.benchmark.cli import main

    for i in range(4):
        _make_audio(tmp_path / f"s{i}.wav")
        (tmp_path / f"s{i}.csv").write_text("0,1,60\n")

    assert main([str(tmp_path), "--dry-run", "--limit", "2"]) == 0
    assert "Limited to the first 2 items" in capsys.readouterr().out


def test_helpful_error_when_nothing_pairs(tmp_path, capsys):
    from transcpybe.benchmark.cli import main

    _make_audio(tmp_path / "only_audio.wav")
    assert main([str(tmp_path), "--dry-run"]) == 1
    assert "no audio could be paired" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Pairing when a dataset ships several annotations per recording
# --------------------------------------------------------------------------

def _vocadito_like(root: Path, clips=(1, 2, 3)):
    """A dataset with f0, two note annotators and lyrics per clip."""
    (root / "Audio").mkdir(parents=True)
    (root / "Annotations" / "Notes").mkdir(parents=True)
    (root / "Annotations" / "F0").mkdir(parents=True)
    (root / "Annotations" / "Lyrics").mkdir(parents=True)
    for i in clips:
        _make_audio(root / "Audio" / f"vocadito_{i}.wav")
        for annotator in ("A", "B"):
            (root / "Annotations" / "Notes" / f"vocadito_{i}_notes{annotator}.csv").write_text(
                "0.5,0.9,60\n1.1,1.5,62\n")
        (root / "Annotations" / "F0" / f"vocadito_{i}_f0.csv").write_text("0.0,440.0\n")
        (root / "Annotations" / "Lyrics" / f"vocadito_{i}_lyrics.csv").write_text("0,1,la\n")
    return root


def test_pairs_when_the_tag_is_not_the_end_of_the_name(tmp_path):
    """vocadito_1_notesA does not end in _notes, so tag matching alone fails."""
    pairing = pair_dataset(_vocadito_like(tmp_path))
    assert len(pairing.items) == 3
    assert all("prefix" in item.match for item in pairing.items)


def test_note_annotation_is_preferred_over_f0_and_lyrics(tmp_path):
    """Picking the f0 contour would score the transcriber against the wrong
    kind of annotation entirely."""
    pairing = pair_dataset(_vocadito_like(tmp_path))
    for item in pairing.items:
        assert "notes" in item.annotation.name


def test_ambiguous_matches_are_reported_as_such(tmp_path):
    """Choosing one of several annotators must not look like a clean match."""
    pairing = pair_dataset(_vocadito_like(tmp_path))
    assert all("1 of" in item.match for item in pairing.items)


def test_annotation_contains_pins_a_specific_annotator(tmp_path):
    pairing = pair_dataset(_vocadito_like(tmp_path), annotation_contains="notesB")
    assert len(pairing.items) == 3
    assert all(item.annotation.name.endswith("_notesB.csv") for item in pairing.items)


def test_a_single_annotation_per_clip_is_not_reported_as_ambiguous(tmp_path):
    (tmp_path / "Audio").mkdir()
    (tmp_path / "Notes").mkdir()
    _make_audio(tmp_path / "Audio" / "clip_1.wav")
    (tmp_path / "Notes" / "clip_1_notes.csv").write_text("0.5,0.9,60\n")
    pairing = pair_dataset(tmp_path)
    assert len(pairing.items) == 1
    assert "1 of" not in pairing.items[0].match


def test_diagnose_shows_both_sides_when_nothing_pairs(tmp_path):
    """Filename conventions vary too much to guess at; print the names."""
    _make_audio(tmp_path / "track001.wav")
    (tmp_path / "ANNOT_song_alpha.csv").write_text("0,1,60\n")

    pairing = pair_dataset(tmp_path)
    assert not pairing.items
    diagnosis = pairing.diagnose()
    assert "track001" in diagnosis
    assert "ANNOT_song_alpha" in diagnosis


def test_failure_message_explains_the_matching_rules(tmp_path, capsys):
    from transcpybe.benchmark.cli import main

    _make_audio(tmp_path / "track001.wav")
    (tmp_path / "ANNOT_song_alpha.csv").write_text("0,1,60\n")

    assert main([str(tmp_path), "--dry-run"]) == 1
    captured = capsys.readouterr()
    assert "track001" in captured.out
    assert "--annotations" in captured.err


def test_dry_run_names_the_annotation_file_it_used(tmp_path, capsys):
    """With several candidates, the item id alone does not say what was scored."""
    from transcpybe.benchmark.cli import main

    _vocadito_like(tmp_path, clips=(1,))
    main([str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert "vocadito_1_notesA.csv" in out
    assert "several annotations" in out


# ---------------------------------------------------------------------------
# Column-order inference.
#
# vocadito writes onset/pitch/duration where most datasets write
# onset/offset/pitch. Reading the wrong column as pitch scored a working
# transcriber at exactly 0.000 COnP while COn stayed at 0.705 -- the onsets
# were still column 1 -- so the failure looked like a broken model. These
# pin the inference that tells the two layouts apart.
# ---------------------------------------------------------------------------


def _melody_rows(layout, n=60, hz=True):
    """Write the same melody out in a given column order."""
    midi = [62, 64, 65, 67, 65, 64, 62, 60]
    rows, t = [], 0.0
    for i in range(n):
        pitch = midi[i % len(midi)]
        value = 440.0 * 2 ** ((pitch - 69) / 12) if hz else float(pitch)
        duration = 0.35
        if layout == "onset_pitch_duration":
            rows.append(f"{t:.4f},{value:.4f},{duration:.4f}")
        else:
            rows.append(f"{t:.4f},{t + duration:.4f},{value:.4f}")
        t += duration + 0.25
    return "\n".join(rows), midi


def test_pitch_column_inferred_when_pitch_precedes_duration(tmp_path):
    """vocadito's onset/pitch/duration must not be read as onset/offset/pitch."""
    text, midi = _melody_rows("onset_pitch_duration")
    path = tmp_path / "vocadito_1_notesA.csv"
    path.write_text(text)

    annotation = load_annotation(path)

    assert annotation.n_notes == 60
    assert set(round(pitch) for _, _, pitch in annotation.notes) == set(midi)
    assert annotation.inferences["columns"].startswith("onset, pitch")
    # Durations must come from the third column, not from Hz read as a time.
    assert all(0.3 < end - start < 0.4 for start, end, _ in annotation.notes)


def test_conventional_column_order_still_wins(tmp_path):
    """The common layout keeps the benefit of the doubt, in Hz and in MIDI."""
    for hz in (True, False):
        text, midi = _melody_rows("onset_offset_pitch", hz=hz)
        path = tmp_path / f"song_notes_{hz}.csv"
        path.write_text(text)

        annotation = load_annotation(path)

        assert annotation.inferences["columns"] == "onset, offset-or-duration, pitch"
        assert set(round(pitch) for _, _, pitch in annotation.notes) == set(midi)


def test_long_recording_does_not_look_like_pitch(tmp_path):
    """Elapsed seconds pass through the MIDI range; that must not fool it."""
    rows, t = [], 0.0
    for i in range(300):        # ten minutes: offsets sweep 0-600 s
        rows.append(f"{t:.4f},{t + 1.0:.4f},{60 + (i % 12)}")
        t += 2.0
    path = tmp_path / "long_notes.csv"
    path.write_text("\n".join(rows))

    annotation = load_annotation(path)

    assert annotation.inferences["columns"] == "onset, offset-or-duration, pitch"
    assert all(58 <= round(pitch) <= 74 for _, _, pitch in annotation.notes)


def test_misread_reference_is_reported_not_scored(tmp_path):
    """Onsets matching while pitches never do must name the likely cause."""
    reference = [(i * 1.0, i * 1.0 + 0.5, 60 + (i % 5)) for i in range(40)]
    # Same timing, pitches from a misread column: nothing can match on pitch.
    estimated = [(i * 1.0 + 0.01, i * 1.0 + 0.5, 1) for i in range(40)]

    score = score_item("misread", reference, estimated)

    assert score.con.f > 0.9
    assert score.conp.f == 0.0
    assert score.ref_pitch_lo == 60 and score.ref_pitch_hi == 64
    assert score.est_pitch_lo == 1 and score.est_pitch_hi == 1


def test_low_male_voice_hz_is_not_read_as_midi(tmp_path):
    """A bass clears 127 Hz on a minority of notes; counting them is not enough.

    This scored six vocadito recordings at exactly 0.000 COnP: the reference
    landed at MIDI 92-127, two octaves above any singer, while the transcriber
    had it right at 42-56.
    """
    low = [40, 42, 43, 45, 47, 45, 43, 42]
    rows, t = [], 0.0
    for i in range(60):
        hz = 440.0 * 2 ** ((low[i % len(low)] - 69) / 12)
        rows.append(f"{t:.4f},{hz:.4f},0.3500")
        t += 0.65
    path = tmp_path / "vocadito_4_notesA.csv"
    path.write_text("\n".join(rows))

    annotation = load_annotation(path)

    assert annotation.inferences["pitch_units"].startswith("hz")
    assert set(round(pitch) for _, _, pitch in annotation.notes) == set(low)
    assert not annotation.warnings


def test_genuine_midi_columns_are_left_alone(tmp_path):
    """Including a bass part written in MIDI, which the Hz reading would ruin."""
    for name, pitches in (("mid", range(60, 73)), ("low", range(36, 49))):
        rows, t = [], 0.0
        for i, pitch in enumerate(list(pitches) * 5):
            rows.append(f"{t:.4f},{t + 0.4:.4f},{pitch}")
            t += 0.7
        path = tmp_path / f"{name}_notes.csv"
        path.write_text("\n".join(rows))

        annotation = load_annotation(path)

        assert annotation.inferences["pitch_units"] == "midi"
        assert set(round(p) for _, _, p in annotation.notes) == set(pitches)


def test_annotator_pairing_finds_both_annotations(tmp_path):
    """vocadito ships two annotators; the ceiling comes from scoring one on the other."""
    from transcpybe.benchmark.dataset import pair_annotators

    for song in range(3):
        for tag in ("notesA", "notesB"):
            (tmp_path / f"vocadito_{song}_{tag}.csv").write_text("0.0,220.0,0.4")
        (tmp_path / f"vocadito_{song}_f0.csv").write_text("0.0,220.0")

    pairs = pair_annotators(tmp_path, "notesA", "notesB")

    assert len(pairs) == 3
    for _, first, second in pairs:
        assert "notesA" in first.stem and "notesB" in second.stem


def test_miss_anatomy_separates_the_reasons_notes_are_lost():
    """Recall below precision needs a cause, not just a count."""
    from transcpybe.benchmark.metrics import analyse_misses

    reference = [
        (0.0, 0.5, 60), (0.5, 1.0, 60),   # a re-articulation the estimate merges
        (2.0, 2.5, 64),                    # transcribed a semitone flat
        (3.0, 3.08, 67),                   # below the resolution floor
        (4.0, 4.6, 69),                    # nothing overlaps it
        (5.0, 5.5, 71),                    # found
    ]
    estimated = [(0.0, 1.0, 60), (2.0, 2.5, 63), (5.0, 5.5, 71)]

    anatomy = analyse_misses(reference, estimated)

    assert anatomy.merged == 1
    assert anatomy.wrong_pitch == 1
    assert anatomy.too_short == 1
    assert anatomy.absent == 1
    assert anatomy.total == 4


def test_miss_anatomy_handles_an_empty_transcription():
    from transcpybe.benchmark.metrics import analyse_misses

    anatomy = analyse_misses([(0.0, 0.5, 60), (1.0, 1.5, 62)], [])

    assert anatomy.absent == 2 and anatomy.total == 2


def test_temperament_line_actually_reaches_the_report():
    """The measurement was once computed and never printed. Pin the rendering.

    A number that reaches no output is worse than no number: it reads as
    evidence the question was asked and answered.
    """
    from transcpybe.benchmark.runner import (
        BenchmarkResult, Provenance, format_result,
    )

    scores = [score_item(f"r{i}", REFERENCE, REFERENCE) for i in range(3)]
    provenance = Provenance(
        timestamp="t", git_commit="c", dataset_digest="d", n_items=3,
        config={}, packages={}, python="3", platform="p",
    )
    result = BenchmarkResult(
        aggregate=aggregate_scores(scores), items=scores, provenance=provenance,
        temperament_cents=18.5,
    )

    report = format_result(result)

    assert "18.5 cents from equal temperament" in report
    assert "31.5 cents" in report, "the remaining slack must be shown"


def test_paired_comparison_sees_what_separate_intervals_hide():
    """Two runs over the same recordings are paired data, not two samples.

    Per-recording difficulty dominates the variance and is common to both runs.
    Comparing separate intervals counts it twice and hides real improvements.
    """
    import numpy as np
    from transcpybe.benchmark.metrics import ItemScore, PRF, compare_runs

    rng = np.random.default_rng(0)
    difficulty = rng.uniform(0.30, 0.85, 40)

    def run(offset):
        return {f"r{i}": ItemScore(
            f"r{i}", 50, 50,
            PRF(d + offset, d + offset, d + offset),
            PRF(d + offset, d + offset, d + offset),
            PRF(d + offset, d + offset, d + offset),
        ) for i, d in enumerate(difficulty)}

    before, after = run(0.0), run(0.02)

    # The separate intervals overlap, so the report alone reads as noise.
    interval_a = aggregate_scores(list(before.values())).macro["conp_f"]
    interval_b = aggregate_scores(list(after.values())).macro["conp_f"]
    assert interval_b.low < interval_a.high

    comparison = compare_runs(before, after, "conp_f")

    assert comparison.significant
    assert comparison.n_better == 40 and comparison.n_worse == 0
    assert comparison.delta == pytest.approx(0.02, abs=1e-6)


def test_paired_comparison_calls_a_wash_a_wash():
    """A change that helps some recordings and hurts others is not an improvement."""
    import numpy as np
    from transcpybe.benchmark.metrics import ItemScore, PRF, compare_runs

    rng = np.random.default_rng(1)
    difficulty = rng.uniform(0.3, 0.8, 40)
    noise = rng.normal(0, 0.05, 40)

    before = {f"r{i}": ItemScore(f"r{i}", 50, 50, PRF(d, d, d), PRF(d, d, d), PRF(d, d, d))
              for i, d in enumerate(difficulty)}
    after = {f"r{i}": ItemScore(f"r{i}", 50, 50, PRF(d + n, d + n, d + n),
                                PRF(d + n, d + n, d + n), PRF(d + n, d + n, d + n))
             for i, (d, n) in enumerate(zip(difficulty, noise))}

    comparison = compare_runs(before, after, "conp_f")

    assert not comparison.significant


def test_pitch_misses_are_attributed_to_intonation_or_decoding():
    """Whether an integer output could ever have reached them."""
    from transcpybe.benchmark.metrics import analyse_misses

    reference, estimated = [], []
    for i in range(30):                       # sung close to a semitone: reachable
        reference.append((i * 1.0, i * 1.0 + 0.5, 60.0 + (i % 5) + 0.05))
        estimated.append((i * 1.0, i * 1.0 + 0.5, float(60 + (i % 5))))
    for i in range(30, 40):                   # sung between two: unreachable
        reference.append((i * 1.0, i * 1.0 + 0.5, 64.48))
        estimated.append((i * 1.0, i * 1.0 + 0.5, 65.0))

    anatomy = analyse_misses(reference, estimated)

    assert anatomy.wrong_pitch == 10
    assert anatomy.wrong_pitch_temperament_cents > 40
    assert anatomy.found_temperament_cents < 10


def test_external_transcriptions_are_scored_identically(tmp_path):
    """A competitor's MIDI goes through the same metric as our own output.

    Comparing published figures compares datasets, not systems. This is the
    path that produces a number meaning the same thing on both sides.
    """
    import pretty_midi
    import soundfile as sf
    from transcpybe.benchmark.cli import main
    from transcpybe.synth import case_scale

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    midi_dir = tmp_path / "external"
    midi_dir.mkdir()

    for i in range(3):
        audio, truth = case_scale()
        sf.write(audio_dir / f"song_{i}.wav", audio, 16000)
        (audio_dir / f"song_{i}_notes.csv").write_text(
            "\n".join(f"{a},{b},{p}" for a, b, p in truth))
        # Right on every onset, a semitone flat on every third note.
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=53)
        for j, (start, end, pitch) in enumerate(truth):
            instrument.notes.append(pretty_midi.Note(
                velocity=90, pitch=int(pitch - 1 if j % 3 == 0 else pitch),
                start=start, end=end))
        midi.instruments.append(instrument)
        midi.write(str(midi_dir / f"song_{i}.mid"))

    output = tmp_path / "external.json"
    exit_code = main([str(audio_dir), "--estimates", str(midi_dir),
                      "--bootstrap", "50", "-o", str(output), "-q"])

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["provenance"]["system"] == "external", (
        "a result must record which system produced it, so a later comparison "
        "cannot silently put two runs of the same transcriber side by side")
    macro = payload["aggregate"]["macro"]
    assert macro["con_f"]["mean"] == pytest.approx(1.0)
    assert macro["conp_f"]["mean"] == pytest.approx(2 / 3, abs=0.02)


def test_missing_external_transcriptions_are_reported_not_silently_dropped(tmp_path):
    import soundfile as sf
    from transcpybe.benchmark.cli import main
    from transcpybe.synth import case_scale

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (tmp_path / "external").mkdir()
    for i in range(2):
        audio, truth = case_scale()
        sf.write(audio_dir / f"song_{i}.wav", audio, 16000)
        (audio_dir / f"song_{i}_notes.csv").write_text(
            "\n".join(f"{a},{b},{p}" for a, b, p in truth))

    exit_code = main([str(audio_dir), "--estimates", str(tmp_path / "external"), "-q"])

    assert exit_code == 1, "scoring nothing must fail rather than report an empty pass"


def test_keyed_corpus_with_parallel_arrays(tmp_path):
    """M4Singer's shape: one JSON for the corpus, durations, no onset column."""
    from transcpybe.benchmark.annotations import load_keyed

    corpus = {f"song_{i}": {"note_midi": [62, 64, 65, 67],
                            "note_dur": [0.4, 0.3, 0.5, 0.35],
                            "is_slur": [0, 0, 1, 0]}
              for i in range(3)}
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(corpus))

    loaded = load_keyed(path)

    assert len(loaded) == 3
    notes = loaded["song_0"].notes
    assert [round(p) for _, _, p in notes] == [62, 64, 65, 67]
    # Durations are consecutive, so onsets are their running total.
    assert notes[0][0] == pytest.approx(0.0)
    assert notes[1][0] == pytest.approx(0.4)
    assert notes[2][0] == pytest.approx(0.7)
    assert notes[3][1] == pytest.approx(1.55)


def test_keyed_corpus_with_triples(tmp_path):
    """MIR-ST500's shape: one JSON, a list of triples per song id."""
    from transcpybe.benchmark.annotations import load_keyed

    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"1": [[0.0, 0.5, 60], [0.6, 1.0, 62]],
                                "2": [[0.0, 0.4, 67]]}))

    loaded = load_keyed(path)

    assert loaded["1"].n_notes == 2 and loaded["2"].n_notes == 1


def test_single_recording_is_not_mistaken_for_a_corpus(tmp_path):
    """Its wrapper keys are not recording ids."""
    from transcpybe.benchmark.annotations import load_keyed

    path = tmp_path / "one.json"
    path.write_text(json.dumps({"notes": [[0.0, 0.5, 60]]}))

    with pytest.raises(ValueError):
        load_keyed(path)


def test_items_file_restricts_to_a_published_split(tmp_path, capsys):
    """Scoring a model on recordings it trained on is not a result."""
    import soundfile as sf
    from transcpybe.benchmark.cli import main
    from transcpybe.synth import case_scale

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    corpus = {}
    for i in range(5):
        audio, truth = case_scale()
        sf.write(audio_dir / f"song_{i}.wav", audio, 16000)
        corpus[f"song_{i}"] = [[a, b, p] for a, b, p in truth]
    (tmp_path / "meta.json").write_text(json.dumps(corpus))
    (tmp_path / "split.txt").write_text("# test split\nsong_1\nsong_3\n")

    main([str(audio_dir), "--annotations-json", str(tmp_path / "meta.json"),
          "--items", str(tmp_path / "split.txt"), "--dry-run"])

    out = capsys.readouterr().out
    assert "Restricted to the 2 of 5" in out
    assert "song_1" in out and "song_3" in out
    assert "song_0" not in out and "song_2" not in out


def test_inspector_reports_layout_and_previews(tmp_path):
    """Reading the directory beats inferring the format from a paper."""
    import numpy as np
    import soundfile as sf
    from transcpybe.benchmark.inspect import inspect_directory

    for singer in ("Alto-1", "Tenor-2"):
        for segment in range(2):
            directory = tmp_path / singer / f"song#{segment}"
            directory.mkdir(parents=True)
            sf.write(directory / "0000.wav", np.zeros(1600), 16000)
            # Praat writes UTF-16 as often as UTF-8.
            (directory / "0000.TextGrid").write_text(
                'File type = "ooTextFile"\nsize = 2\n'
                '        name = "words"\n        name = "phones"\n',
                encoding="utf-16")

    report = inspect_directory(tmp_path)

    assert "4 audio" in report
    assert ".textgrid" in report
    assert "['words', 'phones']" in report, "tier names decide whether notes are present"
    assert "3 level(s): 8 files" in report


def test_inspector_says_what_the_loader_makes_of_a_file(tmp_path):
    """A census alone is not enough: it must say whether the file will load."""
    from transcpybe.benchmark.inspect import inspect_directory

    (tmp_path / "song_notes.csv").write_text(
        "\n".join(f"{i * 0.6},{440.0 * 2 ** ((62 - 69) / 12):.3f},0.4" for i in range(20)))

    report = inspect_directory(tmp_path)

    assert "loader:" in report
    assert "inferred:" in report
    assert "pitch_units=hz" in report


def test_inspector_handles_an_unreadable_file_without_dying(tmp_path):
    from transcpybe.benchmark.inspect import inspect_directory

    (tmp_path / "broken.json").write_bytes(b"\x00\x01 not json at all")

    report = inspect_directory(tmp_path)

    assert "broken.json" in report
    assert "unreadable" in report or "failed" in report


# ---------------------------------------------------------------------------
# Item identity.
#
# M4Singer numbers segments inside a directory per song: 20896 files across 700
# folders, every one of them 0000.wav, 0001.wav. The stem alone is both the
# cache key and the reported name, so identical ids would collapse the corpus
# onto a handful of cache entries and score the same few recordings repeatedly.
# ---------------------------------------------------------------------------


def _m4singer_tree(root, songs=(("Alto-3", "阿楚姑娘"), ("Alto-3", "另一首歌"),
                                ("Alto-1", "newboy")), segments=3):
    import numpy as np
    import pretty_midi
    import soundfile as sf

    for singer, song in songs:
        directory = root / f"{singer}#{song}"
        directory.mkdir(parents=True)
        for segment in range(segments):
            sf.write(directory / f"{segment:04d}.wav", np.zeros(1600), 16000)
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            instrument.notes.append(pretty_midi.Note(
                velocity=90, pitch=60 + segment, start=0.1, end=0.5))
            midi.instruments.append(instrument)
            midi.write(str(directory / f"{segment:04d}.mid"))


def test_segments_numbered_per_song_get_distinct_ids(tmp_path):
    _m4singer_tree(tmp_path)

    pairing = pair_dataset(tmp_path)
    ids = [item.item_id for item in pairing.items]

    assert len(ids) == 9
    assert len(set(ids)) == 9, "every recording must be distinguishable"
    assert "Alto-3#阿楚姑娘/0000" in ids


def test_cache_names_survive_non_ascii_song_titles():
    """Stripping Chinese titles would map every song by one singer onto one name."""
    from transcpybe.benchmark.runner import _cache_name

    names = [_cache_name(i) for i in ("Alto-3#阿楚姑娘/0004", "Alto-3#另一首歌/0004",
                                      "Alto-3#阿楚姑娘/0005", "a/" * 80 + "x")]

    assert len(set(names)) == 4
    assert all("/" not in name for name in names), "a cache name is a filename"
    assert all(len(name) < 150 for name in names)


def test_split_list_matches_whatever_separator_the_corpus_uses(tmp_path, capsys):
    """M4Singer writes Alto-1#newboy#0000; the pairing builds Alto-1#newboy/0000."""
    from transcpybe.benchmark.cli import main

    _m4singer_tree(tmp_path, songs=(("Alto-1", "newboy"), ("Alto-3", "另一首歌")))
    (tmp_path / "split.txt").write_text("Alto-1#newboy#0000\nAlto-1#newboy#0002\n")

    main([str(tmp_path), "--items", str(tmp_path / "split.txt"), "--dry-run"])

    out = capsys.readouterr().out
    assert "Restricted to the 2 of 6" in out


def test_each_recording_pairs_with_its_own_directory(tmp_path):
    """The bug: 20896 recordings paired against ~114 annotation files.

    Every song numbers its segments from 0000, so a single index keyed on the
    bare stem handed every song the first song's notes. Every item paired and
    every file read cleanly, and the score would have been computed against
    another recording's annotations.
    """
    import numpy as np
    import pretty_midi
    import soundfile as sf

    songs = [("Alto-1", "newboy"), ("Alto-1", "云烟成雨"), ("Alto-3", "阿楚姑娘")]
    for index, (singer, song) in enumerate(songs):
        directory = tmp_path / f"{singer}#{song}"
        directory.mkdir(parents=True)
        for segment in range(3):
            sf.write(directory / f"{segment:04d}.wav", np.zeros(1600), 16000)
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            for note in range(3 + index * 4):     # a different count per song
                instrument.notes.append(pretty_midi.Note(
                    velocity=90, pitch=60 + note,
                    start=0.1 * note, end=0.1 * note + 0.08))
            midi.instruments.append(instrument)
            midi.write(str(directory / f"{segment:04d}.mid"))

    pairing = pair_dataset(tmp_path)

    assert len(pairing.items) == 9
    assert len({item.annotation for item in pairing.items}) == 9
    for item in pairing.items:
        assert item.annotation.parent == item.audio.parent
    assert "WARNING" not in pairing.summary()


def test_ambiguous_stem_is_left_unpaired_rather_than_guessed(tmp_path):
    """Two annotations share a stem and neither sits beside the audio."""
    import numpy as np
    import soundfile as sf

    (tmp_path / "audio").mkdir()
    sf.write(tmp_path / "audio" / "0000.wav", np.zeros(1600), 16000)
    for song in ("song_a", "song_b"):
        directory = tmp_path / "refs" / song
        directory.mkdir(parents=True)
        (directory / "0000.csv").write_text("0.0,0.5,60\n")

    pairing = pair_dataset(tmp_path / "audio", tmp_path / "refs")

    assert not pairing.items, "picking one of two would score against the wrong notes"
    assert len(pairing.unpaired_audio) == 1


def test_summary_warns_when_one_annotation_serves_many_recordings(tmp_path):
    """The backstop, for pairing paths not enumerated above."""
    from transcpybe.benchmark.dataset import Item, Pairing

    shared = tmp_path / "shared.csv"
    pairing = Pairing(items=[
        Item(item_id=f"r{i}", audio=tmp_path / f"r{i}.wav", annotation=shared)
        for i in range(4)
    ])

    summary = pairing.summary()

    assert "WARNING" in summary
    assert "share 1 annotation" in summary
    assert "3 are reusing" in summary


# ---------------------------------------------------------------------------
# Segmented corpora.
#
# M4Singer cuts each performance into segments. Those rows are not independent
# observations, and directory order is not a sample.
# ---------------------------------------------------------------------------


def test_interval_resamples_performances_not_segments():
    """Segments of one song share its difficulty; resampling them overstates certainty."""
    import numpy as np
    from transcpybe.benchmark.metrics import PRF, ItemScore

    rng = np.random.default_rng(0)
    scores, flat = [], []
    for song in range(20):
        quality = rng.uniform(0.35, 0.85)          # difficulty belongs to the song
        for segment in range(30):
            f = float(np.clip(quality + rng.normal(0, 0.02), 0, 1))
            prf = PRF(f, f, f)
            scores.append(ItemScore(f"singer#song{song}/{segment:04d}", 12, 12,
                                    prf, prf, prf))
            flat.append(ItemScore(f"item{len(flat)}", 12, 12, prf, prf, prf))

    grouped = aggregate_scores(scores)
    ungrouped = aggregate_scores(flat)

    assert grouped.n_groups == 20
    assert ungrouped.n_groups == 0
    grouped_width = grouped.macro["conp_f"].high - grouped.macro["conp_f"].low
    flat_width = ungrouped.macro["conp_f"].high - ungrouped.macro["conp_f"].low
    assert grouped_width > 3 * flat_width, (
        "resampling segments reports an interval far narrower than the evidence")
    # The point estimate is unchanged; only the uncertainty around it moves.
    assert grouped.macro["conp_f"].mean == pytest.approx(
        ungrouped.macro["conp_f"].mean, abs=1e-9)


def test_sample_spreads_across_performances_and_is_reproducible(tmp_path):
    """Directory order would take every segment from the first song."""
    from transcpybe.benchmark.dataset import (
        Item, group_of, stratified_sample,
    )

    items = [Item(item_id=f"singer{s}#song{g}/{n:04d}",
                  audio=tmp_path / f"{s}{g}{n}.wav",
                  annotation=tmp_path / f"{s}{g}{n}.mid")
             for s in range(3) for g in range(2) for n in range(10)]

    chosen = stratified_sample(items, 6, seed=0)

    assert len(chosen) == 6
    assert len({group_of(i) for i in chosen}) == 6, "one per performance"
    assert [i.item_id for i in chosen] == [i.item_id for i in
                                           stratified_sample(items, 6, seed=0)]
    assert [i.item_id for i in chosen] != [i.item_id for i in items[:6]]


def test_sample_returns_everything_when_the_budget_exceeds_the_corpus(tmp_path):
    from transcpybe.benchmark.dataset import Item, stratified_sample

    items = [Item(item_id=f"a/{i}", audio=tmp_path / f"{i}.wav",
                  annotation=tmp_path / f"{i}.mid") for i in range(4)]

    assert len(stratified_sample(items, 99, seed=0)) == 4


def test_report_names_performances_when_items_are_segments():
    from transcpybe.benchmark.runner import (
        BenchmarkResult, Provenance, format_result,
    )
    from transcpybe.benchmark.metrics import PRF, ItemScore

    scores = [ItemScore(f"singer#song{g}/{n:04d}", 10, 10,
                        PRF(0.6, 0.6, 0.6), PRF(0.6, 0.6, 0.6), PRF(0.6, 0.6, 0.6))
              for g in range(4) for n in range(5)]
    result = BenchmarkResult(
        aggregate=aggregate_scores(scores), items=scores,
        provenance=Provenance(timestamp="t", git_commit="c", dataset_digest="d",
                              n_items=20, config={}, packages={}, python="3",
                              platform="p"))

    report = format_result(result)

    assert "20 recordings from 4 performance(s)" in report
    assert "resamples of the performances" in report


# ---------------------------------------------------------------------------
# Annotation conventions.
#
# M4Singer scored 0.214 COnP with 90% of recordings flagged misaligned, a median
# shift of -0.090s, and reference and estimated pitch ranges that matched almost
# exactly. Right notes, right pitches, consistently late: two definitions of
# where a note starts, not a transcription failure.
# ---------------------------------------------------------------------------


def _syllable_vs_voiced(offset=0.0, songs=30, seed=0):
    """A reference marking syllable onsets against a transcriber marking vowels."""
    import numpy as np

    rng = np.random.default_rng(seed)
    items = []
    for song in range(songs):
        reference, estimated = [], []
        for i in range(12):
            start = i * 0.55
            reference.append((start, start + 0.4, 60 + (i % 6)))
            voiced = start + rng.uniform(0.07, 0.16) + offset
            estimated.append((max(0.0, voiced), max(0.0, voiced) + 0.4, 60 + (i % 6)))
        items.append(score_item(f"singer#song{song}/0000", reference, estimated))
    return items


def test_a_clock_offset_is_named_as_one_and_gets_the_whole_note_correction():
    """This fixture shifts both boundaries, which is a clock, not a convention."""
    from transcpybe.benchmark.runner import (
        BenchmarkResult, Provenance, format_result,
    )

    items = _syllable_vs_voiced()
    result = BenchmarkResult(
        aggregate=aggregate_scores(items), items=items,
        provenance=Provenance(timestamp="t", git_commit="c", dataset_digest="d",
                              n_items=30, config={}, packages={}, python="3",
                              platform="p"))

    report = format_result(result, worst=0)

    assert "share a sign" in report
    assert "Starts and ends are shifted alike" in report
    assert "--time-offset" in report
    assert "--onset-shift" not in report, "the ends moved too, so starts-only is wrong"


def test_a_few_misaligned_items_do_not_trigger_the_corpus_wide_claim():
    """A handful of bad files is a data problem, not a convention difference."""
    from transcpybe.benchmark.runner import (
        BenchmarkResult, Provenance, format_result,
    )

    aligned = [score_item(f"song{i}", REFERENCE, REFERENCE) for i in range(20)]
    broken = _syllable_vs_voiced(songs=2, seed=1)
    items = aligned + broken

    result = BenchmarkResult(
        aggregate=aggregate_scores(items), items=items,
        provenance=Provenance(timestamp="t", git_commit="c", dataset_digest="d",
                              n_items=22, config={}, packages={}, python="3",
                              platform="p"))

    report = format_result(result, worst=0)

    assert "misaligned and the shifts agree in sign" not in report


def test_a_stated_correction_recovers_a_convention_difference():
    """And the residue is what a single constant cannot fix."""
    uncorrected = aggregate_scores(_syllable_vs_voiced(0.0)).macro["conp_f"].mean
    corrected = aggregate_scores(_syllable_vs_voiced(-0.115)).macro["conp_f"].mean

    assert uncorrected == pytest.approx(0.0, abs=0.01)
    assert corrected > 0.9


def test_a_corrected_run_says_so_in_the_report_and_the_provenance(tmp_path):
    """A conditional number must not be readable as a plain one."""
    import soundfile as sf
    from transcpybe.benchmark.cli import main
    from transcpybe.synth import case_scale

    audio, truth = case_scale()
    sf.write(tmp_path / "s.wav", audio, 16000)
    (tmp_path / "s.csv").write_text("\n".join(f"{a},{b},{p}" for a, b, p in truth))

    output = tmp_path / "corrected.json"
    main([str(tmp_path), "--time-offset", "-0.115", "--f0-backend", "pyin",
          "--bootstrap", "20", "-o", str(output), "-q"])

    payload = json.loads(output.read_text())
    assert payload["provenance"]["time_offset"] == pytest.approx(-0.115)


def _convention_items(mode="none", duration=0.25, songs=30, shift=0.09):
    """A reference marking syllables against a transcriber marking vowels.

    The note ends agree, because both mark the end of the vowel. Only the
    starts differ, by the length of the leading consonant.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    items = []
    for song in range(songs):
        reference, estimated = [], []
        for i in range(12):
            start = i * (duration + 0.15)
            reference.append((start, start + duration, 60 + (i % 5)))
            voiced, end = start + rng.uniform(shift - 0.02, shift + 0.02), start + duration
            if mode == "whole":
                voiced, end = voiced - shift, end - shift
            elif mode == "onset":
                voiced = min(voiced - shift, end - 0.02)
            estimated.append((max(0.0, voiced), max(0.02, end), 60 + (i % 5)))
        items.append(score_item(f"singer#song{song}/0000", reference, estimated))
    return items


def test_boundary_deltas_separate_a_clock_offset_from_an_onset_convention():
    reference = [(i * 0.6, i * 0.6 + 0.45, 60 + (i % 5)) for i in range(20)]
    convention = [(i * 0.6 + 0.09, i * 0.6 + 0.45, 60 + (i % 5)) for i in range(20)]
    clock = [(i * 0.6 + 0.09, i * 0.6 + 0.45 + 0.09, 60 + (i % 5)) for i in range(20)]

    a = score_item("convention", reference, convention)
    b = score_item("clock", reference, clock)

    assert a.onset_delta_median == pytest.approx(0.09, abs=0.01)
    assert a.end_delta_median == pytest.approx(0.0, abs=0.01)
    assert b.onset_delta_median == pytest.approx(0.09, abs=0.01)
    assert b.end_delta_median == pytest.approx(0.09, abs=0.01)


def test_shifting_whole_notes_breaks_ends_that_already_agreed():
    """Why --time-offset raised COnP on M4Singer and halved COnPOff.

    The offset tolerance is max(50 ms, 20% of the reference note). On short
    notes, moving the end by 90 ms puts it outside a boundary that was right.
    """
    whole = aggregate_scores(_convention_items("whole")).macro["conpoff_f"].mean
    onset = aggregate_scores(_convention_items("onset")).macro["conpoff_f"].mean

    assert onset > 0.9
    assert whole < 0.1, "moving the ends destroys alignments that were correct"


def test_no_convention_is_claimed_when_the_shifts_do_not_agree(tmp_path):
    """Symmetric scatter about zero is not a direction, and no constant fixes it."""
    import numpy as np
    from transcpybe.benchmark.metrics import PRF
    from transcpybe.benchmark.runner import (
        BenchmarkResult, Provenance, format_result,
    )

    rng = np.random.default_rng(1)
    items = []
    for song in range(30):
        reference, estimated = [], []
        for i in range(12):
            start = i * 0.6
            reference.append((start, start + 0.45, 60 + (i % 5)))
            jitter = rng.uniform(-0.12, 0.12)
            estimated.append((max(0.0, start + jitter), start + 0.45, 60 + (i % 5)))
        items.append(score_item(f"singer#song{song}/0000", reference, estimated))

    result = BenchmarkResult(
        aggregate=aggregate_scores(items), items=items,
        provenance=Provenance(timestamp="t", git_commit="c", dataset_digest="d",
                              n_items=30, config={}, packages={}, python="3",
                              platform="p"))
    report = format_result(result, worst=0)

    assert "do not agree in sign" in report
    assert "no constant corrects it" in report
    assert "--onset-shift" not in report, "must not suggest a correction that cannot work"


def test_onset_convention_recommends_shifting_starts_only():
    from transcpybe.benchmark.runner import (
        BenchmarkResult, Provenance, format_result,
    )

    items = _convention_items("none")
    result = BenchmarkResult(
        aggregate=aggregate_scores(items), items=items,
        provenance=Provenance(timestamp="t", git_commit="c", dataset_digest="d",
                              n_items=30, config={}, packages={}, python="3",
                              platform="p"))

    report = format_result(result, worst=0)

    assert "share a sign" in report
    assert "--onset-shift" in report
    assert "ends agree and the starts do not" in report


def _shifted_items(start_delta, end_delta, songs=30, seed=0):
    """Estimates displaced from the reference at each boundary independently."""
    import numpy as np

    rng = np.random.default_rng(seed)
    items = []
    for song in range(songs):
        reference, estimated = [], []
        for i in range(12):
            start = i * 0.6
            reference.append((start, start + 0.40, 60 + (i % 5)))
            jitter = rng.uniform(-0.008, 0.008)
            estimated.append((max(0.0, start + start_delta + jitter),
                              start + 0.40 + end_delta + jitter, 60 + (i % 5)))
        items.append(score_item(f"singer#song{song}/0000", reference, estimated))
    return items


def _report_for(items):
    from transcpybe.benchmark.runner import (
        BenchmarkResult, Provenance, format_result,
    )

    return format_result(BenchmarkResult(
        aggregate=aggregate_scores(items), items=items,
        provenance=Provenance(timestamp="t", git_commit="c", dataset_digest="d",
                              n_items=len(items), config={}, packages={},
                              python="3", platform="p")), worst=0)


def test_boundaries_moving_opposite_ways_are_a_length_difference():
    """ROSVOT's notes started 41 ms early and ended 25 ms late: wider, not shifted.

    The report called that "a clock that does not match" and recommended a
    translation, which cannot correct a difference in length.
    """
    report = _report_for(_shifted_items(-0.09, +0.07))

    assert "opposite directions" in report
    assert "longer than the reference" in report
    assert "--time-offset" not in report, "no translation corrects a length difference"
    assert "--onset-shift +0.089" in report or "--onset-shift +0.090" in report


def test_a_clock_offset_is_corrected_in_the_direction_that_helps():
    """The recommendation must undo the measured displacement, not double it."""
    late = _report_for(_shifted_items(+0.08, +0.08))
    early = _report_for(_shifted_items(-0.08, -0.08))

    assert "shifted alike" in late and "shifted alike" in early
    # An estimate that is late must be moved earlier, and vice versa.
    assert "--time-offset -0.0" in late
    assert "--time-offset +0.0" in early


def test_an_onset_convention_is_corrected_in_the_right_direction():
    report = _report_for(_shifted_items(+0.08, 0.0))

    assert "ends agree and the starts do not" in report
    assert "--onset-shift -0.0" in report


def test_metadata_join_finds_the_id_column_rather_than_assuming_it(tmp_path):
    """Datasets call it track_id, item_name, id, or the filename."""
    from transcpybe.benchmark.dataset import load_item_metadata

    path = tmp_path / "vocadito_metadata.csv"
    path.write_text("track_id,singer_id,language\n1,S1,English\n2,S2,Spanish\n"
                    "10,S3,Mandarin\n")

    mapping, id_column = load_item_metadata(
        f"{path}:language", ["vocadito_1", "vocadito_2", "vocadito_10"])

    assert id_column == "track_id"
    assert mapping["vocadito_10"] == "Mandarin"
    assert len(mapping) == 3


def test_metadata_join_refuses_rather_than_joining_nothing(tmp_path):
    from transcpybe.benchmark.dataset import load_item_metadata

    path = tmp_path / "meta.csv"
    path.write_text("unrelated,language\nzzz,English\n")

    with pytest.raises(ValueError, match="matched any item id"):
        load_item_metadata(f"{path}:language", ["vocadito_1"])

    with pytest.raises(ValueError, match="no column"):
        load_item_metadata(f"{path}:missing_column", ["vocadito_1"])


def test_breakdown_separates_a_material_effect_from_a_model_one(tmp_path, capsys):
    """The question the breakdown exists to answer.

    A system that trails everywhere is a worse model. One that trails only
    outside its training distribution is a domain effect, and the difference
    is visible only per group.
    """
    from transcpybe.benchmark.cli import main

    languages = {1: "Mandarin", 2: "Mandarin", 3: "English",
                 4: "English", 5: "Spanish", 6: "Spanish"}

    def write(name, score_for):
        payload = {"provenance": {"system": "x", "time_offset": 0.0},
                   "aggregate": {"macro": {}}, "items": []}
        for track, language in languages.items():
            f = score_for(language)
            prf = {"precision": f, "recall": f, "f": f}
            payload["items"].append({"item_id": f"vocadito_{track}", "n_reference": 20,
                                     "n_estimated": 20, "con": prf, "conp": prf,
                                     "conpoff": prf})
        (tmp_path / name).write_text(json.dumps(payload))

    write("a.json", lambda lang: 0.60 if lang == "Mandarin" else 0.20)
    write("b.json", lambda lang: 0.58)
    (tmp_path / "meta.csv").write_text(
        "track_id,language\n" + "\n".join(f"{t},{l}" for t, l in languages.items()))

    main(["--compare", str(tmp_path / "a.json"), str(tmp_path / "b.json"),
          "--by-metadata", f"{tmp_path / 'meta.csv'}:language"])
    out = capsys.readouterr().out

    assert "By language (joined on track_id, 6 of 6 items matched)" in out
    mandarin = next(l for l in out.splitlines() if "Mandarin" in l)
    english = next(l for l in out.splitlines() if "English" in l)
    assert "-0.020" in mandarin, "the trained system holds up in its own domain"
    assert "+0.380" in english, "and collapses outside it"


def test_compare_refuses_flags_it_cannot_apply(tmp_path, capsys):
    """A saved result holds scores, not notes.

    --onset-shift was accepted and ignored, so the comparison came back
    identical and read as evidence that the correction changed nothing.
    """
    from transcpybe.benchmark.cli import main

    prf = {"precision": 0.5, "recall": 0.5, "f": 0.5}
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps({
            "provenance": {"system": "x", "time_offset": 0.0},
            "aggregate": {"macro": {}},
            "items": [{"item_id": "r1", "n_reference": 5, "n_estimated": 5,
                       "con": prf, "conp": prf, "conpoff": prf}]}))

    for flag, value in (("--onset-shift", "0.041"), ("--time-offset", "-0.09"),
                        ("--estimates", str(tmp_path)), ("--limit", "5")):
        code = main(["--compare", str(tmp_path / "a.json"), str(tmp_path / "b.json"),
                     flag, value])
        assert code == 2, f"{flag} must be refused, not ignored"
        assert flag in capsys.readouterr().err

    assert main(["--compare", str(tmp_path / "a.json"),
                 str(tmp_path / "b.json")]) == 0


# ---------------------------------------------------------------------------
# Clock rate.
#
# A reference notated at a nominal tempo drifts against a recorded performance,
# by seconds over a song. The misalignment guard cannot see it: it looks for a
# constant shift that improves the score, and no constant shift does.
# ---------------------------------------------------------------------------


def _song(tempo_ratio=1.0, offset=0.0, notes=120):
    reference = [(i * 0.6, i * 0.6 + 0.4, 60 + (i % 7)) for i in range(notes)]
    estimated = [(i * 0.6 * tempo_ratio + offset,
                  i * 0.6 * tempo_ratio + offset + 0.4, 60 + (i % 7))
                 for i in range(notes)]
    return reference, estimated


def test_drift_separates_a_rate_mismatch_from_a_constant_offset():
    reference, drifted = _song(tempo_ratio=1.03)
    _, shifted = _song(offset=0.08)

    rate = score_item("rate", reference, drifted)
    constant = score_item("constant", reference, shifted)

    assert abs(rate.onset_drift) > 0.5, "a 3% tempo error drifts by over a second"
    assert abs(constant.onset_drift) < 0.05, "a shift has the same error throughout"
    assert constant.onset_delta_early == pytest.approx(0.08, abs=0.02)


def test_a_rate_mismatch_is_reported_though_the_old_guard_cannot_see_it():
    """The case that passed silently: no constant shift helps, so nothing flagged."""
    from transcpybe.benchmark.runner import (
        BenchmarkResult, Provenance, format_result,
    )

    items = [score_item(f"song{i}", *_song(tempo_ratio=1.03)) for i in range(12)]
    result = BenchmarkResult(
        aggregate=aggregate_scores(items), items=items,
        provenance=Provenance(timestamp="t", git_commit="c", dataset_digest="d",
                              n_items=12, config={}, packages={}, python="3",
                              platform="p"))

    assert not result.aggregate.items_needing_alignment, (
        "precondition: the shift-based guard is blind to a rate mismatch")

    report = format_result(result, worst=0)

    assert "DIFFERENT CLOCK RATE" in report
    assert "sequence of pitches" in report, "it must say what can still be measured"
    assert "neither" in report, "it must say the constant corrections will not help"


def test_a_constant_offset_does_not_raise_the_rate_warning():
    from transcpybe.benchmark.runner import (
        BenchmarkResult, Provenance, format_result,
    )

    items = [score_item(f"song{i}", *_song(offset=0.08)) for i in range(12)]
    result = BenchmarkResult(
        aggregate=aggregate_scores(items), items=items,
        provenance=Provenance(timestamp="t", git_commit="c", dataset_digest="d",
                              n_items=12, config={}, packages={}, python="3",
                              platform="p"))

    report = format_result(result, worst=0)

    assert "DIFFERENT CLOCK RATE" not in report
    assert result.aggregate.items_needing_alignment


def test_drift_is_not_measured_on_material_that_cannot_show_it():
    """Music repeats, so a wide search on a short clip aligns a phrase to the next.

    On a 6.6 s fixture the search reported 2.9 s of drift, which is nearly half
    the clip. Too short or too sparse is now not measured rather than measured
    wrongly.
    """
    short_reference = [(i * 0.6, i * 0.6 + 0.4, 60 + (i % 5)) for i in range(12)]
    short_estimate = [(i * 0.6 + 0.08, i * 0.6 + 0.48, 60 + (i % 5)) for i in range(12)]

    score = score_item("short", short_reference, short_estimate)

    assert score.onset_drift == 0.0
    assert score.onset_delta_early == 0.0

    # The same shift over material long enough to measure is still seen.
    long_reference, long_estimate = _song(offset=0.08)
    long_score = score_item("long", long_reference, long_estimate)
    assert long_score.onset_delta_early == pytest.approx(0.08, abs=0.02)


# ---------------------------------------------------------------------------
# Explicit pairing.
#
# Separated stems are every one of them lead-vocal.wav, identified only by their
# directory, and their references are named by year and rank. The two share no
# string, so no name matching can join them.
# ---------------------------------------------------------------------------


def _stem_tree(root, songs):
    import numpy as np
    import soundfile as sf

    rows = ["audio,annotation"]
    for song in songs:
        directory = root / "stems" / song
        directory.mkdir(parents=True)
        sf.write(directory / "lead-vocal.wav", np.zeros(1600), 16000)
        annotation = root / f"{song}.csv"
        annotation.write_text("\n".join(f"{i * 0.5},{i * 0.5 + 0.3},{60 + i}"
                                        for i in range(8)))
        rows.append(f"{directory / 'lead-vocal.wav'},{annotation}")
    (root / "pairs.csv").write_text("\n".join(rows))
    return root / "pairs.csv"


def test_explicit_pairing_identifies_stems_by_their_directory(tmp_path):
    from transcpybe.benchmark.dataset import pair_from_file

    path = _stem_tree(tmp_path, ["Wiz_Khalifa_-_See_You_Again", "Adele_-_Hello"])

    pairing = pair_from_file(path)

    assert len(pairing.items) == 2
    ids = {i.item_id for i in pairing.items}
    assert ids == {"Wiz_Khalifa_-_See_You_Again/lead-vocal",
                   "Adele_-_Hello/lead-vocal"}
    assert all(i.match == "explicit" for i in pairing.items)
    assert "WARNING" not in pairing.summary()


def test_explicit_pairing_rejects_a_repeated_item_id(tmp_path):
    """Ids are cache keys; duplicates would collapse recordings onto each other."""
    from transcpybe.benchmark.dataset import pair_from_file

    path = _stem_tree(tmp_path, ["A_-_One", "B_-_Two"])
    text = path.read_text().splitlines()
    path.write_text("\n".join(["item_id," + text[0]]
                              + [f"same,{line}" for line in text[1:]]))

    with pytest.raises(ValueError, match="repeats item_id"):
        pair_from_file(path)


def test_explicit_pairing_fails_loudly_on_a_missing_annotation(tmp_path):
    """A wrong path in the mapping must not silently shrink the dataset."""
    from transcpybe.benchmark.dataset import pair_from_file

    path = _stem_tree(tmp_path, ["A_-_One"])
    path.write_text(path.read_text().replace("A_-_One.csv", "does_not_exist.csv"))

    with pytest.raises(ValueError, match="no such annotation"):
        pair_from_file(path)


def test_explicit_pairing_needs_its_columns(tmp_path):
    from transcpybe.benchmark.dataset import pair_from_file

    path = tmp_path / "pairs.csv"
    path.write_text("wav,midi\na.wav,b.mid\n")

    with pytest.raises(ValueError, match="needs columns"):
        pair_from_file(path)
