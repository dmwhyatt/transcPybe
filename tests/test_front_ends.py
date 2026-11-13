"""Swappable pitch front ends, and measured pitch in the MIDI file.

``config.f0_backend`` picks the tracker from a registry that callers can add
to; every track records which tracker made it. ``config.twelve_tet`` decides
whether the MIDI file holds whole semitones or carries each note's measured
pitch as a pitch bend, which ``read_midi_notes`` reads back.
"""
import importlib.util

import numpy as np
import pretty_midi
import pytest

from transcpybe import f0 as f0_module
from transcpybe.cli import build_parser, config_from_args
from transcpybe.config import TranscriptionConfig
from transcpybe.core import transcribe_audio
from transcpybe.evaluate import score
from transcpybe.f0 import backend_label, estimate_f0, f0_backends, register_f0_backend
from transcpybe.midi_io import read_midi_notes, write_midi
from transcpybe.notes import Note
from transcpybe.synth import case_mistuned

SR = 16000
HAS_PESTO = importlib.util.find_spec("pesto") is not None


def _sine(hz=220.0, seconds=2.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _note(onset, offset, measured, pitch=None):
    pitch = int(round(measured)) if pitch is None else pitch
    return Note(onset=onset, offset=offset, pitch=pitch, velocity=80, confidence=0.9, loudness_db=-20.0,
                pitch_range_semitones=0.1, pitch_hz=440.0 * 2 ** ((measured - 69) / 12),
                pitch_midi_float=measured, hmm_pitch=pitch, start_frame=int(onset * 100),
                end_frame=int(offset * 100))


@pytest.fixture
def fixed_backend():
    """A tracker that always hears 220 Hz, registered for one test only."""
    def estimate(audio, config, progress=None):
        n = 1 + audio.shape[0] // config.hop_length
        return np.full(n, 220.0), np.full(n, 0.9)

    register_f0_backend("fixed", estimate)
    yield "fixed"
    del f0_module._BACKENDS["fixed"]


def test_the_built_in_backends_are_listed_and_an_unknown_one_is_refused():
    assert f0_backends()[:3] == ("crepe", "pyin", "pesto")
    with pytest.raises(ValueError, match="f0_backend must be one of"):
        TranscriptionConfig(f0_backend="nope").validate()
    with pytest.raises(ValueError, match="available: crepe, pyin, pesto"):
        estimate_f0(_sine(), TranscriptionConfig(f0_backend="nope"))


def test_a_registered_backend_is_used_and_named_on_its_track(fixed_backend):
    config = TranscriptionConfig(f0_backend=fixed_backend)
    config.validate()
    track = estimate_f0(_sine(), config)
    assert track.backend == "fixed"
    assert track.n_frames == 1 + int(2.0 * SR) // config.hop_length
    assert np.allclose(track.f0_hz, 220.0)


def test_each_track_records_the_tracker_that_made_it():
    assert backend_label(TranscriptionConfig()) == "pyin"
    assert backend_label(TranscriptionConfig(f0_backend="crepe")) == "crepe-full"
    assert backend_label(TranscriptionConfig(f0_backend="crepe", crepe_model="tiny")) == "crepe-tiny"
    assert backend_label(TranscriptionConfig(f0_backend="pesto")) == "pesto"
    track = estimate_f0(_sine(), TranscriptionConfig(f0_backend="pyin"))
    assert track.backend == "pyin"


@pytest.mark.skipif(not HAS_PESTO, reason="pesto-pitch is not installed")
@pytest.mark.parametrize("sr", [16000, 22050])
def test_pesto_lands_on_this_package_s_frame_grid_and_hears_the_pitch(sr):
    # At 22050 Hz a 10 ms step is 220 samples here and 221 in PESTO, so its
    # frames drift and have to be put back on ours.
    config = TranscriptionConfig(f0_backend="pesto", device="cpu", sample_rate=sr)
    audio = _sine(220.0, 3.0, sr)
    track = estimate_f0(audio, config)
    assert track.backend == "pesto"
    assert track.n_frames == 1 + audio.shape[0] // config.hop_length
    voiced = track.f0_hz[np.isfinite(track.f0_hz)]
    assert abs(np.median(1200 * np.log2(voiced / 220.0))) < 25
    assert np.median(track.periodicity) > 0.5


@pytest.mark.skipif(not HAS_PESTO, reason="pesto-pitch is not installed")
@pytest.mark.parametrize("sr", [16000, 22050])
def test_pesto_in_chunks_gives_the_frames_of_one_pass(monkeypatch, sr):
    # Long files go to PESTO in chunks so its memory stays bounded; with a
    # margin of half its widest kernel either side, each chunk's frames are
    # the ones one pass over the file gives. Chunks of 0.37 s make many seams.
    import transcpybe.f0 as f0_module

    config = TranscriptionConfig(f0_backend="pesto", device="cpu", sample_rate=sr)
    t = np.arange(int(4.0 * sr)) / sr
    glide = 2 * np.pi * np.cumsum(150.0 * 2 ** (t / 2.0)) / sr        # 150 Hz up two octaves
    audio = (0.3 * np.sin(glide) * (np.sin(2 * np.pi * 1.3 * t) > -0.5)
             + 0.01 * np.random.default_rng(0).standard_normal(t.size)).astype(np.float32)
    whole_f0, whole_conf = f0_module._estimate_pesto(audio, config)   # 4 s: one chunk
    monkeypatch.setattr(f0_module, "PESTO_CHUNK_S", 0.37)
    f0_hz, conf = f0_module._estimate_pesto(audio, config)
    assert f0_hz.shape == whole_f0.shape
    np.testing.assert_allclose(f0_hz, whole_f0, rtol=1e-4, equal_nan=True)
    np.testing.assert_allclose(conf, whole_conf, atol=1e-4)


def test_twelve_tet_is_off_by_default_and_the_cli_can_turn_it_on():
    assert TranscriptionConfig().twelve_tet is False
    assert config_from_args(build_parser().parse_args(["x.wav"])).twelve_tet is False
    assert config_from_args(build_parser().parse_args(["x.wav", "--12tet"])).twelve_tet is True


def test_measured_pitch_survives_the_midi_file_and_twelve_tet_drops_it(tmp_path):
    notes = [_note(0.5, 1.0, 60.3), _note(1.0, 1.4, 61.8), _note(1.6, 2.0, 64.0)]
    path = write_midi(notes, tmp_path / "bent.mid", TranscriptionConfig())
    back = read_midi_notes(path)
    assert [round(p, 3) for _, _, p in back] == [60.3, 61.8, 64.0]
    assert [n.pitch for n in pretty_midi.PrettyMIDI(str(path)).instruments[0].notes] == [60, 62, 64]

    path = write_midi(notes, tmp_path / "whole.mid", TranscriptionConfig(twelve_tet=True))
    assert [p for _, _, p in read_midi_notes(path)] == [60.0, 62.0, 64.0]
    assert pretty_midi.PrettyMIDI(str(path)).instruments[0].pitch_bends == []


def test_a_note_off_the_tuning_grid_bends_from_its_tuned_integer(tmp_path):
    # pitch_reference='tuned' can give an integer up to 0.9 semitone from what
    # was sung; the bend still lands on the measured pitch.
    notes = [_note(0.5, 1.0, 60.85, pitch=60)]
    back = read_midi_notes(write_midi(notes, tmp_path / "tuned.mid", TranscriptionConfig()))
    assert round(back[0][2], 3) == 60.85


def test_another_writer_s_bend_range_is_read_from_its_rpn(tmp_path):
    midi = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=0)
    for time, number, value in [(0.0, 101, 0), (0.01, 100, 0), (0.02, 6, 12), (0.03, 38, 0)]:
        track.control_changes.append(pretty_midi.ControlChange(number, value, time))
    track.pitch_bends.append(pretty_midi.PitchBend(4096, 0.5))       # half of 12 semitones
    track.notes.append(pretty_midi.Note(80, 60, 0.5, 1.0))
    midi.instruments.append(track)
    midi.write(str(tmp_path / "wide.mid"))
    assert round(read_midi_notes(tmp_path / "wide.mid")[0][2], 3) == 66.0


def test_scoring_measured_pitch_credits_a_note_its_integer_would_miss():
    # Sung 60.6 against a reference of 60.45: the integer, 61, is 55 cents off.
    notes = [_note(0.5, 1.0, 60.6)]
    reference = [(0.5, 1.0, 60.45)]
    assert score(notes, reference).note_f1 == 0.0
    assert score(notes, reference, measured_pitch=True).note_f1 == 1.0


def test_a_transcription_reads_back_at_its_measured_pitches(tmp_path):
    audio, _ = case_mistuned(SR)
    result = transcribe_audio(audio, TranscriptionConfig(f0_backend="pyin"))
    assert result.f0_track.backend == "pyin"
    back = read_midi_notes(write_midi(result.notes, tmp_path / "sung.mid", result.config))
    assert len(back) == len(result.notes)
    for (_, _, pitch), note in zip(back, result.notes):
        assert abs(pitch - note.pitch_midi_float) < 0.001


def test_pesto_runs_without_tf32_on_a_gpu_and_restores_the_setting():
    # TF32 moved PESTO's pitch by octaves on an RTX 3080; full float32 gives the CPU's frames.
    torch = pytest.importorskip("torch")   # the base install, as CI runs it, has no torch

    from transcpybe.f0 import _full_float32

    before = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    try:
        with _full_float32("cuda:0"):
            assert torch.backends.cudnn.allow_tf32 is False
            assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is True
        with _full_float32("cpu"):
            assert torch.backends.cudnn.allow_tf32 is True        # nothing to change off a GPU
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = before
