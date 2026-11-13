"""transcPybe v2: the shipped networks and voicing maps, and the presets that use them.

The networks need torch, so those tests skip without it; none needs torchcrepe, because the end-to-end test tracks
pitch with pYIN. That the package gives the notes the presets were scored with is checked on the scored recordings
themselves, which cannot ship with the tests; here the files are checked to be those networks, by hash.
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from importlib import resources

import numpy as np
import pytest

from transcpybe import TranscriptionConfig, transcribe_audio
from transcpybe.landmarks import DECODER_FIELDS, LANDMARK_DECODER, PRESETS, preset_config, voicing_map

SHA256 = {
    "seed0.pt": "b374fc0e7f0086a55a6a6f90fdec65e7c2d5d42df41f405d5152a1dd3d02852a",
    "seed1.pt": "47f2c5d14fb014a92608a8abac14a712e0370343098ece3000703f8f63068b92",
    "seed2.pt": "29f4237d7d66b2dcb1e675b9f52008536cd8c41adf0f2a97c1238ab4e5faf9a7",
    "voicing-pesto.json": "cda7f2bd195f3e076f1a83987c7e001b9bf0b989faf582c5d9f11f6b41ebe7ba",
    "voicing-pyin.json": "55023bb82ca23fdc36c03a24b66c3e2874d850881efd90cefd353d3ed08eaaa1",
}


def weights():
    return resources.files("transcpybe.landmarks") / "weights"


def test_weights_are_the_scored_ones():
    for name, want in SHA256.items():
        assert hashlib.sha256((weights() / name).read_bytes()).hexdigest() == want, name


def test_weights_carry_their_own_licence():
    assert (weights() / "LICENSE").read_text(encoding="utf-8").startswith(
        "Attribution-NonCommercial-ShareAlike 4.0 International")
    notice = (weights() / "README.md").read_text(encoding="utf-8")
    assert "CC BY-NC-SA 4.0" in notice and "M4Singer" in notice and "GTSinger" in notice


def test_voicing_maps():
    for tracker in ("pesto", "pyin"):
        m = voicing_map(tracker)
        assert np.all(np.diff(m["src"]) > 0) and np.all(np.diff(m["dst"]) >= 0)
        assert 0.0 <= m["dst"][0] and m["dst"][-1] <= 1.0
    assert voicing_map("crepe-full") is None and voicing_map("crepe-tiny") is None


def test_preset_config_sets_only_the_decoder_fields():
    base = TranscriptionConfig(preset="rules", f0_backend="pyin", twelve_tet=True, voicing_threshold=0.25)
    assert preset_config(base) is base
    for name, p in PRESETS.items():
        given = dataclasses.replace(base, preset=name)
        cfg = preset_config(given)
        assert {k: getattr(cfg, k) for k in LANDMARK_DECODER} == LANDMARK_DECODER
        assert cfg.release_drop_db == p.release_drop_db
        changed = {f for f in given.to_dict() if getattr(cfg, f) != getattr(given, f)}
        assert changed <= set(DECODER_FIELDS)
        assert cfg.twelve_tet and cfg.f0_backend == "pyin" and cfg.voicing_threshold == 0.25
        assert preset_config(cfg) == cfg


def test_preset_validation():
    TranscriptionConfig(preset="acappella").validate()
    with pytest.raises(ValueError, match="preset must be"):
        TranscriptionConfig(preset="v2").validate()
    with pytest.raises(ValueError, match="use_onsets"):
        TranscriptionConfig(preset="pop", use_onsets=False).validate()
    with pytest.raises(ValueError, match="hop_ms"):
        TranscriptionConfig(preset="pop", hop_ms=5.0).validate()


def test_cli_preset_says_which_flags_it_overrides(capsys):
    from transcpybe.cli import build_parser, config_from_args

    cfg = config_from_args(build_parser().parse_args(["x.wav", "--preset", "pop", "--f0-backend", "pyin"]))
    assert cfg.preset == "pop" and capsys.readouterr().err == ""
    config_from_args(build_parser().parse_args(["x.wav", "--preset", "pop", "--min-note", "0.2"]))
    assert "--min-note" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        config_from_args(build_parser().parse_args(["x.wav", "--preset", "pop", "--no-onsets"]))


def test_the_default_is_v2_pop_with_pyin():
    from transcpybe.cli import build_parser, config_from_args

    for cfg in (TranscriptionConfig(), config_from_args(build_parser().parse_args(["x.wav"]))):
        assert cfg.preset == "pop" and cfg.f0_backend == "pyin"


def test_the_cli_says_when_a_flag_does_nothing_with_the_tracker(capsys):
    from transcpybe.cli import build_parser, config_from_args

    config_from_args(build_parser().parse_args(["x.wav", "--crepe-model", "tiny", "--device", "cuda"]))
    err = capsys.readouterr().err
    assert "--crepe-model, --device do nothing with --f0-backend pyin" in err
    config_from_args(build_parser().parse_args(["x.wav", "--f0-backend", "crepe", "--crepe-model", "tiny",
                                                "--device", "cpu"]))
    assert capsys.readouterr().err == ""
    config_from_args(build_parser().parse_args(["x.wav", "--f0-backend", "pesto", "--device", "cpu"]))
    assert capsys.readouterr().err == ""


def test_validation_names_the_fix_for_the_default_preset():
    with pytest.raises(ValueError, match=r"needs? preset='rules'.*the default is 'pop'"):
        TranscriptionConfig(use_onsets=False).validate()
    with pytest.raises(ValueError, match=r"needs? preset='rules'.*the default is 'pop'"):
        TranscriptionConfig(sample_rate=22050).validate()
    TranscriptionConfig(preset="rules", use_onsets=False, sample_rate=22050).validate()


def test_the_version_is_set_in_one_place():
    import re

    import transcpybe

    pyproject = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    project = pyproject.split("[project]", 1)[1].split("\n[", 1)[0]          # the [project] table alone
    assert 'dynamic = ["version"]' in project and not re.search(r"^version\s*=", project, re.M)
    assert re.fullmatch(r"\d+\.\d+\.\d+", transcpybe.__version__)


def test_onset_position_is_validated():
    TranscriptionConfig(onset_position=0.0).validate()
    TranscriptionConfig(preset="acappella", onset_position=1.0).validate()
    with pytest.raises(ValueError, match="onset_position must be in"):
        TranscriptionConfig(onset_position=1.5).validate()
    with pytest.raises(ValueError, match="needs a version 2 preset"):
        TranscriptionConfig(preset="rules", onset_position=0.5).validate()


def test_cli_onset_position():
    from transcpybe.cli import build_parser, config_from_args

    assert config_from_args(build_parser().parse_args(["x.wav"])).onset_position is None
    assert config_from_args(build_parser().parse_args(["x.wav", "--onset-position", "0.25"])).onset_position == 0.25
    with pytest.raises(SystemExit):
        config_from_args(build_parser().parse_args(["x.wav", "--preset", "rules", "--onset-position", "0.25"]))


@pytest.mark.parametrize("preset,position,frame", [("pop", None, 110), ("acappella", None, 118), ("pop", 0.0, 100),
                                                   ("pop", 1.0, 120), ("acappella", 0.25, 105)])
def test_onset_position_places_the_start_between_syllable_and_vowel(monkeypatch, preset, position, frame):
    """A syllable start at frame 100 and its vowel at 120: the note starts at 100 + F x 20, F the preset's unless set."""
    pytest.importorskip("torch")
    import transcpybe.landmarks as L
    import transcpybe.landmarks.infer as infer
    from transcpybe.f0 import estimate_f0
    from transcpybe.synth import case_scale

    def posteriors(models, mel, aux, device="cpu"):
        post = np.zeros((len(mel), 5))
        post[100, 0] = post[120, 4] = 0.9
        return post

    monkeypatch.setattr(infer, "averaged_posteriors", posteriors)
    monkeypatch.setattr(L, "networks", lambda: ())
    audio, _ = case_scale()
    config = L.preset_config(TranscriptionConfig(preset=preset, f0_backend="pyin", onset_position=position))
    _, onsets, _, _ = L.landmark_notes(audio, estimate_f0(audio, config), config)
    assert list(np.round(onsets.times / 0.01).astype(int)) == [frame]


def test_networks_load():
    pytest.importorskip("torch")
    from transcpybe.landmarks import networks

    nets = networks()
    assert len(nets) == 3 and networks() is nets
    for net in nets:
        assert net.cfg.n_out == 5 and not net.training
        assert sum(p.numel() for p in net.parameters()) == 823_717


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_presets_transcribe_a_scale(preset):
    pytest.importorskip("torch")
    from transcpybe.evaluate import score
    from transcpybe.synth import case_scale

    audio, truth = case_scale()
    result = transcribe_audio(audio, TranscriptionConfig(preset=preset, f0_backend="pyin"))
    assert result.config.preset == preset
    assert result.config.release_drop_db == PRESETS[preset].release_drop_db
    assert result.onsets is not None and len(result.onsets.times) > 0
    assert score(result.notes, truth).note_f1 >= 0.9
