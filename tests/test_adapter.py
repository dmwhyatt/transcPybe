"""The adapter a benchmark uses to run this transcriber in process.

Thin, but not trivial: it is the one place where a file path meets a function
that wants samples, and where a device gets chosen. Both were wrong the first
time and neither showed up anywhere else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcpybe.adapter import TranscPybe
from transcpybe.config import TranscriptionConfig


@pytest.fixture
def recording(tmp_path):
    sf = pytest.importorskip("soundfile")
    from transcpybe.synth import case_scale

    audio, truth = case_scale()
    path = tmp_path / "scale.wav"
    sf.write(path, audio, 16000)
    return path, truth


def test_the_adapter_takes_a_path_and_returns_notes(recording):
    """A benchmark has a file. Everything inside this package has samples."""
    path, truth = recording

    notes = TranscPybe(TranscriptionConfig(f0_backend="pyin", device="cpu")).transcribe(path)

    assert notes, "a scale must transcribe to something"
    assert all(len(note) == 3 for note in notes)
    onset, offset, pitch = notes[0]
    assert offset > onset
    assert isinstance(pitch, float), (
        "float, so a benchmark comparing against an unquantised reference is "
        "not silently quantising on its behalf")
    assert len(notes) == pytest.approx(len(truth), abs=2)


def test_settings_report_the_whole_configuration(recording):
    """A parameter left out is one that can change between a cached run and a
    fresh one without the cache noticing."""
    import dataclasses

    config = TranscriptionConfig(f0_backend="pyin", device="cpu")
    settings = TranscPybe(config).settings()

    expected = {f.name for f in dataclasses.fields(TranscriptionConfig)}
    assert set(settings) == expected
    assert settings["f0_backend"] == "pyin"


def test_audio_is_loaded_at_the_rate_the_configuration_asks_for(recording, monkeypatch):
    """A sample rate mismatch here shifts every onset, which reads as a bad
    transcriber rather than a misread file."""
    from transcpybe import audio as audio_module

    path, _ = recording
    rates = []
    original = audio_module.load_audio

    def counted(p, sample_rate):
        rates.append(sample_rate)
        return original(p, sample_rate)

    monkeypatch.setattr(audio_module, "load_audio", counted)

    config = TranscriptionConfig(f0_backend="pyin", device="cpu")
    TranscPybe(config).transcribe(path)

    assert rates == [config.sample_rate]


def test_the_adapter_satisfies_what_the_benchmark_asks_for():
    """The contract is one method. Pinned here so it cannot drift silently."""
    benchmark = pytest.importorskip(
        "transcription_benchmarking.transcribers",
        reason="needs transcpybe[tune]")

    adapter = TranscPybe(TranscriptionConfig(f0_backend="pyin", device="cpu"))

    assert isinstance(adapter, benchmark.Transcriber)
