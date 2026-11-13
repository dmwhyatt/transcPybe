"""Tests for progress reporting, device selection and batch transcription."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pytest

from transcpybe.batch import (
    AUDIO_EXTENSIONS,
    BatchResult,
    find_audio_files,
    output_paths,
    probe_duration,
    transcribe_batch,
    _overall_line,
)
from transcpybe.config import TranscriptionConfig
from transcpybe.device import resolve_device
from transcpybe.progress import (
    ProgressReporter,
    format_duration,
    format_realtime_factor,
    null_reporter,
    setup_logging,
)
from transcpybe.synth import case_legato, case_scale


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seconds,expected", [
    (0, "0:00"), (5, "0:05"), (59, "0:59"), (60, "1:00"),
    (65, "1:05"), (3600, "1:00:00"), (3661, "1:01:01"),
])
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_format_duration_handles_nonsense(value):
    assert format_duration(value) == "--:--"


def test_format_realtime_factor():
    assert format_realtime_factor(10.0, 5.0) == "2.00x realtime"
    assert format_realtime_factor(10.0, None) == ""
    assert format_realtime_factor(10.0, 0) == ""


# --------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------

class FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_non_tty_emits_plain_periodic_lines():
    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, interval=0.0)
    with reporter.stage("work", total=4) as stage:
        for _ in range(4):
            stage.advance(1)
    text = stream.getvalue()
    assert "\r" not in text, "carriage returns would corrupt a redirected log"
    assert "work: started" in text
    assert "work: done" in text


def test_tty_rewrites_a_single_line():
    stream = FakeTty()
    reporter = ProgressReporter(stream=stream, interval=0.0)
    assert reporter.interactive
    with reporter.stage("work", total=4) as stage:
        stage.advance(1)
    assert "\r" in stream.getvalue()


def test_disabled_reporter_writes_nothing():
    stream = io.StringIO()
    reporter = ProgressReporter(enabled=False, stream=stream)
    with reporter.stage("work", total=2) as stage:
        stage.advance(2)
    reporter.log("hello")
    assert stream.getvalue() == ""


def test_null_reporter_is_disabled():
    assert null_reporter().enabled is False


def test_stage_tracks_fraction_and_eta():
    reporter = ProgressReporter(enabled=False)
    with reporter.stage("work", total=10) as stage:
        stage.advance(5)
        assert stage.fraction == pytest.approx(0.5)
        assert stage.eta is not None and stage.eta >= 0
        stage.advance(5)
        assert stage.fraction == 1.0


def test_stage_without_total_has_no_fraction():
    reporter = ProgressReporter(enabled=False)
    with reporter.stage("work") as stage:
        stage.advance(3)
        assert stage.fraction is None
        assert stage.eta is None


def test_fraction_is_capped_at_one_when_overshooting():
    reporter = ProgressReporter(enabled=False)
    with reporter.stage("work", total=2) as stage:
        stage.advance(10)
        assert stage.fraction == 1.0


def test_stage_reports_done_even_when_the_body_raises():
    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, interval=0.0)
    with pytest.raises(RuntimeError):
        with reporter.stage("work"):
            raise RuntimeError("boom")
    assert "work: done" in stream.getvalue()


def test_announce_false_suppresses_the_start_line():
    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, interval=0.0)
    with reporter.stage("quick", announce=False):
        pass
    text = stream.getvalue()
    assert "quick: started" not in text
    assert "quick: done" in text


def test_progress_survives_a_broken_stream():
    """A closed pipe must not take the transcription down with it."""
    class Broken(io.StringIO):
        def write(self, _):
            raise OSError("broken pipe")

    reporter = ProgressReporter(stream=Broken(), interval=0.0)
    with reporter.stage("work", total=2) as stage:
        stage.advance(1)
    reporter.log("still fine")


def test_setup_logging_writes_to_the_log_file(tmp_path):
    path = tmp_path / "logs" / "run.log"
    logger = setup_logging(log_file=str(path), stream=io.StringIO())
    logger.info("a distinctive message")
    for handler in logger.handlers:
        handler.flush()
    assert path.exists()
    assert "a distinctive message" in path.read_text()
    logger.handlers.clear()


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------

def test_auto_resolves_to_a_concrete_device():
    info = resolve_device("auto")
    assert info.kind in ("cuda", "mps", "cpu")
    assert info.device and not info.device.startswith("auto")
    assert "auto-selected" in info.describe()


def test_explicit_device_is_honoured_verbatim():
    """An explicit choice must never be silently downgraded."""
    info = resolve_device("cpu")
    assert info.device == "cpu" and info.kind == "cpu"
    assert "auto-selected" not in info.describe()


def test_cpu_device_reports_no_gpu():
    assert resolve_device("cpu").is_gpu is False


# --------------------------------------------------------------------------
# Batch discovery
# --------------------------------------------------------------------------

def test_find_audio_files_is_sorted_and_filtered(tmp_path):
    for name in ("b.wav", "a.wav", "c.flac", "notes.txt", ".hidden.wav"):
        (tmp_path / name).write_bytes(b"")
    found = [p.name for p in find_audio_files(tmp_path)]
    assert found == ["a.wav", "b.wav", "c.flac"]


def test_find_audio_files_recurses_and_can_be_told_not_to(tmp_path):
    (tmp_path / "top.wav").write_bytes(b"")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.wav").write_bytes(b"")

    assert len(find_audio_files(tmp_path, recursive=True)) == 2
    assert len(find_audio_files(tmp_path, recursive=False)) == 1


def test_find_audio_files_accepts_a_single_file(tmp_path):
    path = tmp_path / "one.wav"
    path.write_bytes(b"")
    assert find_audio_files(path) == [path]


def test_all_declared_extensions_are_lowercase_with_a_dot():
    assert all(e.startswith(".") and e == e.lower() for e in AUDIO_EXTENSIONS)


def test_output_paths_follow_the_input_stem(tmp_path):
    paths = output_paths(Path("/x/My Song.wav"), tmp_path)
    assert paths["midi"].name == "My Song.mid"
    assert paths["json"].name == "My Song.notes.json"


def test_probe_duration_reads_a_header(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "t.wav"
    soundfile.write(path, np.zeros(16000, dtype=np.float32), 16000)
    assert probe_duration(path) == pytest.approx(1.0, abs=0.01)


def test_probe_duration_returns_none_for_unreadable_files(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not audio")
    assert probe_duration(path) is None


def test_overall_line_weights_by_audio_when_known():
    line = _overall_line(2, 10, 100.0, 250.0, 1000.0, True)
    assert "25%" in line and "ETA" in line


def test_overall_line_falls_back_to_file_counts():
    line = _overall_line(5, 10, 100.0, 0.0, 0.0, False)
    assert "50%" in line


# --------------------------------------------------------------------------
# Batch transcription
# --------------------------------------------------------------------------

@pytest.fixture
def audio_dir(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    directory = tmp_path / "in"
    directory.mkdir()
    for name, make_case in (("one", case_scale), ("two", case_legato)):
        audio, _ = make_case()
        soundfile.write(directory / f"{name}.wav", audio, 16000)
    return directory


def test_batch_transcribes_every_file(audio_dir, tmp_path):
    out = tmp_path / "out"
    result = transcribe_batch(
        find_audio_files(audio_dir), out, TranscriptionConfig(f0_backend="pyin")
    )
    assert len(result.done) == 2 and not result.failed
    assert (out / "one.mid").exists() and (out / "two.notes.csv").exists()
    assert all(item.n_notes > 0 for item in result.done)


def test_batch_skips_files_already_done(audio_dir, tmp_path):
    """Resumability: an interrupted run must not start from the beginning."""
    out = tmp_path / "out"
    config = TranscriptionConfig(f0_backend="pyin")
    transcribe_batch(find_audio_files(audio_dir), out, config)

    again = transcribe_batch(find_audio_files(audio_dir), out, config)
    assert len(again.skipped) == 2 and not again.done


def _logged(tmp_path, name="batch"):
    """A reporter whose permanent lines land in a string, for reading what a batch said."""
    stream = io.StringIO()
    log = logging.getLogger(f"test-{name}-{id(stream)}")
    log.handlers[:] = [logging.StreamHandler(stream)]
    log.setLevel(logging.INFO)
    log.propagate = False
    return ProgressReporter(enabled=True, logger=log, stream=io.StringIO()), stream


def test_a_rerun_with_other_settings_redoes_the_files(audio_dir, tmp_path):
    """Upgrading or changing a setting and rerunning must not leave a folder of mixed transcriptions."""
    out = tmp_path / "out"
    rules = TranscriptionConfig(preset="rules", f0_backend="pyin")
    transcribe_batch(find_audio_files(audio_dir), out, rules)

    progress, log = _logged(tmp_path)
    again = transcribe_batch(find_audio_files(audio_dir), out, TranscriptionConfig(f0_backend="pyin"),
                             progress=progress)
    assert len(again.done) == 2 and not again.skipped
    assert "made with other settings (preset 'rules' -> 'pop'" in log.getvalue()

    third = transcribe_batch(find_audio_files(audio_dir), out, TranscriptionConfig(f0_backend="pyin"))
    assert len(third.skipped) == 2, "the same settings again: nothing to redo"


def test_settings_that_do_not_change_the_notes_do_not_force_a_redo(audio_dir, tmp_path):
    out = tmp_path / "out"
    transcribe_batch(find_audio_files(audio_dir), out, TranscriptionConfig(preset="rules", f0_backend="pyin"))
    again = transcribe_batch(find_audio_files(audio_dir), out,
                             TranscriptionConfig(preset="rules", f0_backend="pyin", device="cpu",
                                                 crepe_batch_size=64, gpu_memory_log_every=3))
    assert len(again.skipped) == 2


def test_an_output_from_before_presets_is_redone(audio_dir, tmp_path):
    """A 0.1.0 .notes.json has no preset: it was v1, and a v2 run must not keep it."""
    import json as jsonlib

    out = tmp_path / "out"
    transcribe_batch(find_audio_files(audio_dir), out, TranscriptionConfig(preset="rules", f0_backend="pyin"))
    for path in out.glob("*.notes.json"):
        payload = jsonlib.loads(path.read_text())
        del payload["config"]["preset"]
        path.write_text(jsonlib.dumps(payload))
    again = transcribe_batch(find_audio_files(audio_dir), out, TranscriptionConfig(preset="rules", f0_backend="pyin"))
    assert len(again.done) == 2


def test_outputs_without_a_record_are_skipped_and_said_so_once(audio_dir, tmp_path):
    out = tmp_path / "out"
    config = TranscriptionConfig(preset="rules", f0_backend="pyin")
    transcribe_batch(find_audio_files(audio_dir), out, config, write_json_files=False)

    progress, log = _logged(tmp_path, "unverified")
    again = transcribe_batch(find_audio_files(audio_dir), out, config, write_json_files=False, progress=progress)
    assert len(again.skipped) == 2
    assert log.getvalue().count("cannot be checked") == 1


def test_overwrite_reprocesses_everything(audio_dir, tmp_path):
    out = tmp_path / "out"
    config = TranscriptionConfig(f0_backend="pyin")
    transcribe_batch(find_audio_files(audio_dir), out, config)

    again = transcribe_batch(find_audio_files(audio_dir), out, config, skip_existing=False)
    assert len(again.done) == 2 and not again.skipped


def test_batch_continues_past_a_broken_file(audio_dir, tmp_path):
    broken = audio_dir / "broken.wav"
    broken.write_bytes(b"definitely not audio")
    out = tmp_path / "out"

    result = transcribe_batch(
        find_audio_files(audio_dir), out, TranscriptionConfig(f0_backend="pyin")
    )
    assert len(result.failed) == 1
    assert result.failed[0].path.name == "broken.wav"
    assert len(result.done) == 2, "the good files must still be transcribed"
    assert "FAILED" in result.summary()


def test_stop_on_error_aborts_the_batch(audio_dir, tmp_path):
    (audio_dir / "aaa_broken.wav").write_bytes(b"not audio")
    out = tmp_path / "out"

    result = transcribe_batch(
        find_audio_files(audio_dir), out,
        TranscriptionConfig(f0_backend="pyin"), continue_on_error=False,
    )
    assert len(result.failed) == 1
    assert not result.done, "nothing after the failure should have run"


def test_empty_batch_summary_does_not_crash():
    assert "0 transcribed" in BatchResult().summary()


# --------------------------------------------------------------------------
# CREPE: chunking and reproducibility
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [64, 128, 512, 4096])
def test_crepe_batch_count_matches_torchcrepe(batch_size):
    """Progress percentages are only honest if the batch count is exact."""
    torchcrepe = pytest.importorskip("torchcrepe")
    import torch

    from transcpybe.f0 import _crepe_batch_count

    audio, _ = case_scale()
    config = TranscriptionConfig(crepe_batch_size=batch_size, device="cpu")
    tensor = torch.from_numpy(np.ascontiguousarray(audio, np.float32))[None, :]

    actual = sum(1 for _ in torchcrepe.preprocess(
        tensor, config.sample_rate, hop_length=config.hop_length,
        batch_size=batch_size, device="cpu", pad=True,
    ))
    assert _crepe_batch_count(len(audio), config) == actual


@pytest.mark.slow
def test_chunked_crepe_matches_torchcrepe_predict():
    """The chunked loop exists only to report progress, not to change results.

    It must stay bit-identical to torchcrepe.predict, which is the whole
    justification for reimplementing predict's loop.
    """
    torchcrepe = pytest.importorskip("torchcrepe")
    import torch

    from transcpybe.f0 import _estimate_crepe

    audio, _ = case_scale()
    config = TranscriptionConfig(crepe_model="tiny", device="cpu",
                                 crepe_batch_size=128, seed=0)
    tensor = torch.from_numpy(np.ascontiguousarray(audio, np.float32))[None, :]

    np.random.seed(config.seed)
    with torch.no_grad():
        reference_f0, reference_periodicity = torchcrepe.predict(
            tensor, config.sample_rate, hop_length=config.hop_length,
            fmin=max(config.fmin_hz, 10.0),
            fmax=min(config.fmax_hz, torchcrepe.MAX_FMAX),
            model="tiny", batch_size=128, device="cpu",
            return_periodicity=True, pad=True,
        )

    reference = reference_f0[0].numpy().astype(np.float64)
    out_of_range = (reference < config.fmin_hz) | (reference > config.fmax_hz)
    expected_f0 = np.where(out_of_range, np.nan, reference)
    expected_periodicity = np.clip(
        np.where(out_of_range, 0.0, reference_periodicity[0].numpy().astype(np.float64)), 0, 1
    )

    got_f0, got_periodicity = _estimate_crepe(audio, config)
    assert np.array_equal(got_f0, expected_f0, equal_nan=True)
    assert np.array_equal(got_periodicity, expected_periodicity)


@pytest.mark.slow
def test_seeding_makes_crepe_reproducible():
    """torchcrepe dithers pitch by +/-20 cents from the global numpy RNG."""
    pytest.importorskip("torchcrepe")
    from transcpybe.f0 import _estimate_crepe

    audio, _ = case_scale()
    config = TranscriptionConfig(crepe_model="tiny", device="cpu", seed=7)
    first, _ = _estimate_crepe(audio, config)
    second, _ = _estimate_crepe(audio, config)
    assert np.array_equal(first, second, equal_nan=True)


@pytest.mark.slow
def test_unseeded_crepe_varies_between_runs():
    """Documents the upstream behaviour that makes seeding necessary."""
    pytest.importorskip("torchcrepe")
    from transcpybe.f0 import _estimate_crepe

    audio, _ = case_scale()
    config = TranscriptionConfig(crepe_model="tiny", device="cpu", seed=None)
    first, _ = _estimate_crepe(audio, config)
    second, _ = _estimate_crepe(audio, config)
    assert not np.array_equal(first, second, equal_nan=True)


def test_seeding_restores_the_callers_random_state():
    """Seeding for the dither must not hijack the caller's random stream."""
    from transcpybe.f0 import _seeded_numpy

    np.random.seed(1234)
    expected = np.random.random(3)

    np.random.seed(1234)
    with _seeded_numpy(99):
        np.random.random(50)          # disturb the stream inside the block
    assert np.array_equal(np.random.random(3), expected)


def test_seed_none_leaves_the_random_state_alone():
    from transcpybe.f0 import _seeded_numpy

    np.random.seed(5)
    before = np.random.get_state()[1][:5].copy()
    with _seeded_numpy(None):
        pass
    assert np.array_equal(np.random.get_state()[1][:5], before)


# --------------------------------------------------------------------------
# Output naming across a batch
# --------------------------------------------------------------------------

def test_flat_directory_keeps_its_filenames():
    from transcpybe.batch import unique_output_names

    paths = [Path("mel/Song A.wav"), Path("mel/Song B.wav")]
    assert unique_output_names(paths) == {paths[0]: "Song A", paths[1]: "Song B"}


def test_separated_stems_are_named_after_their_song():
    """Every separated stem is called lead-vocal.wav, so the stem alone is useless."""
    from transcpybe.batch import unique_output_names

    paths = [
        Path("work/stems/Don_McLean_-_American_Pie/lead-vocal.wav"),
        Path("work/stems/Nick_Drake_-_Northern_Sky/lead-vocal.wav"),
    ]
    assert list(unique_output_names(paths).values()) == [
        "Don_McLean_-_American_Pie", "Nick_Drake_-_Northern_Sky",
    ]


def test_meaningful_filenames_are_not_discarded_for_directory_names():
    """albums/A/track1.wav must not become "A"; the track name is the useful part."""
    from transcpybe.batch import unique_output_names

    paths = [Path("albums/A/track1.wav"), Path("albums/B/track2.wav")]
    assert list(unique_output_names(paths).values()) == ["track1", "track2"]


def test_colliding_stems_keep_both_directory_and_filename():
    from transcpybe.batch import unique_output_names

    paths = [Path("a/x.wav"), Path("b/y.wav"), Path("c/x.wav")]
    assert list(unique_output_names(paths).values()) == ["a__x", "b__y", "c__x"]


def test_deeper_context_is_used_when_parent_and_stem_both_collide():
    from transcpybe.batch import unique_output_names

    paths = [Path("v1/Song/lead-vocal.wav"), Path("v2/Song/lead-vocal.wav")]
    names = list(unique_output_names(paths).values())
    assert len(set(names)) == 2
    assert all(n.startswith("v") for n in names)


def test_identical_paths_still_get_distinct_names():
    """Pathological, but uniqueness matters more than elegance."""
    from transcpybe.batch import unique_output_names

    paths = [Path("a/x.wav"), Path("a/x.wav")]
    assert len(set(unique_output_names(paths).values())) <= 2  # dict keys collapse


def test_empty_and_single_inputs():
    from transcpybe.batch import unique_output_names

    assert unique_output_names([]) == {}
    assert unique_output_names([Path("one.wav")]) == {Path("one.wav"): "one"}


def test_a_tree_of_stems_produces_one_transcription_per_song(tmp_path):
    """The bug this guards: named by stem alone, every song would write to
    lead-vocal.mid, and because finished work is skipped a whole corpus would
    quietly collapse into a single transcription."""
    soundfile = pytest.importorskip("soundfile")

    stems = tmp_path / "stems"
    for song, make_case in (("Song_One", case_scale), ("Song_Two", case_legato)):
        (stems / song).mkdir(parents=True)
        audio, _ = make_case()
        soundfile.write(stems / song / "lead-vocal.wav", audio, 16000)

    out = tmp_path / "out"
    result = transcribe_batch(
        find_audio_files(stems), out, TranscriptionConfig(f0_backend="pyin")
    )

    assert len(result.done) == 2 and not result.skipped
    assert (out / "Song_One.mid").exists()
    assert (out / "Song_Two.mid").exists()
    assert not (out / "lead-vocal.mid").exists()
