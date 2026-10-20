"""The boundary pipeline's pieces, and one run of the whole thing on three synthetic recordings."""
from pathlib import Path

import numpy as np
import pytest

from boundary import HOP_S, items
from boundary.features import targets
from boundary.infer import onset_track, pick_peaks
from boundary.sources import _vowel_onsets, textgrid_tiers
from boundary.train import resung


def test_targets_peak_on_the_labelled_frames_and_taper():
    y = targets([[1.0, 1.5, 60]], 300)
    assert y[100, 0] == 1 and y[150, 1] == 1
    assert y[99, 0] == pytest.approx(2 / 3) and y[98, 0] == pytest.approx(1 / 3) and y[97, 0] == 0
    assert y[100:150, 2].all() and not y[99, 2] and not y[150, 2]


def test_targets_keep_the_stronger_of_two_overlapping_ramps():
    y = targets([[1.0, 1.02, 60], [1.02, 1.5, 60]], 300)
    assert y[100, 0] == 1 and y[102, 0] == 1 and y[101, 0] == pytest.approx(2 / 3)


def test_pick_peaks_drops_the_weaker_of_two_close_peaks():
    p = np.zeros(100)
    p[[10, 13, 40]] = [0.9, 0.95, 0.4]
    assert pick_peaks(p, 0.5).tolist() == [13]
    assert pick_peaks(p, 0.3).tolist() == [13, 40]


def test_onset_track_sits_on_the_f0_grid():
    p = np.zeros(200)
    p[57] = 0.8
    tr = onset_track(p)
    assert tr.times.tolist() == [pytest.approx(57 * HOP_S)] and tr.strengths.tolist() == [0.8]
    assert len(tr.envelope) == len(tr.envelope_times) == 200


def test_resung_finds_only_same_pitch_notes_with_no_gap():
    notes = [[0, 1, 60], [1, 2, 60], [2.2, 3, 60], [3, 4, 62]]
    assert resung(notes) == [1]


def test_vowel_onsets_moves_a_consonant_start_to_its_vowel(tmp_path):
    tg = tmp_path / "x.TextGrid"
    tg.write_text('\n'.join([
        'File type = "ooTextFile"', 'Object class = "TextGrid"', 'xmin = 0', 'xmax = 1', 'tiers? <exists>',
        'size = 2', 'item []:', 'item [1]:', 'class = "IntervalTier"', 'name = "None"', 'xmin = 0', 'xmax = 1',
        'intervals: size = 1', 'intervals [1]:', 'xmin = 0.0', 'xmax = 1.0', 'text = "shi"',
        'item [2]:', 'class = "IntervalTier"', 'name = "None"', 'xmin = 0', 'xmax = 1', 'intervals: size = 2',
        'intervals [1]:', 'xmin = 0.0', 'xmax = 0.08', 'text = "sh"',
        'intervals [2]:', 'xmin = 0.08', 'xmax = 1.0', 'text = "i"']))
    tiers = list(textgrid_tiers(tg).values())
    assert len(tiers) == 2 and tiers[1][0] == (0.0, 0.08, "sh")
    assert _vowel_onsets([[0.0, 1.0, 60]], tiers[1]) == [[0.08, 1.0, 60]]
    assert _vowel_onsets([[0.0, 0.1, 60]], tiers[1]) == [[0.0, 0.1, 60]]     # would leave no note


def test_model_keeps_one_output_per_frame():
    torch = pytest.importorskip("torch")
    from boundary.model import BoundaryNet

    out = BoundaryNet()(torch.zeros(2, 123, 80), torch.zeros(2, 123, 5))
    assert out.shape == (2, 123, 3)


def _synthetic_vocals(tmp_path: Path) -> list:
    """Three short phrases from transcPybe's own synthesiser, one of them held out, one with a re-sung note."""
    import soundfile as sf
    from transcpybe.synth import synth_phrase

    rows = []
    for i, pitches in enumerate(([60, 62, 64, 64, 65], [67, 65, 64, 62, 60], [57, 57, 59, 60, 62])):
        audio, truth = synth_phrase([(0.3 + 0.5 * k, 0.8 + 0.5 * k, p) for k, p in enumerate(pitches)],
                                    16000, consonants=True, seed=i)
        path = tmp_path / "audio" / f"v{i}.wav"
        path.parent.mkdir(exist_ok=True)
        sf.write(path, audio, 16000)
        rows.append(dict(id=f"synth/{i}", source="synth", audio=str(path), notes=[list(n) for n in truth],
                         split="valid" if i == 0 else "train"))
    return rows


def test_whole_pipeline_on_three_synthetic_recordings(tmp_path: Path):
    pytest.importorskip("torch")
    from boundary import features, infer, mix, separate, train

    items.write(tmp_path / "vocals.jsonl", _synthetic_vocals(tmp_path))
    mix.main([str(tmp_path / "vocals.jsonl"), str(tmp_path / "mix"), "--accomp", "synthetic", "--keep-dry"])
    rows = items.read(tmp_path / "mix" / "mixes.jsonl")
    vocal = items.read(tmp_path / "vocals.jsonl")
    assert rows[0]["notes"][0][0] == pytest.approx(vocal[0]["notes"][0][0] + rows[0]["pad_before"], abs=1e-3)

    again = mix.render(vocal[0], 0, 0, 0, "synthetic", tmp_path / "mix", True)     # rebuilt from its draws
    assert again["notes"] == rows[0]["notes"] and again["gain"] == rows[0]["gain"]

    separate.main([str(tmp_path / "mix" / "mixes.jsonl"), str(tmp_path / "sep"), "--backend", "dry"])
    features.main([str(tmp_path / "sep" / "stems.jsonl"), str(tmp_path / "feat"), "--f0-backend", "pyin", "--device", "cpu"])
    z = np.load(tmp_path / "feat" / f"{rows[0]['mix_id']}.npz")
    assert z["mel"].shape[0] == z["aux"].shape[0] == z["y"].shape[0] and z["y"][:, 0].max() == 1

    train.main([str(tmp_path / "feat" / "features.jsonl"), "--out", str(tmp_path / "run"), "--steps", "20",
                "--eval-every", "10", "--batch", "4", "--device", "cpu"])
    assert (tmp_path / "run" / "best.pt").exists() and len((tmp_path / "run" / "log.jsonl").read_text().splitlines()) == 2
    stem = items.read(tmp_path / "sep" / "stems.jsonl")[0]["stem"]
    infer.main([str(tmp_path / "run" / "best.pt"), stem, "--out-dir", str(tmp_path / "inf"), "--device", "cpu",
                "--f0-backend", "pyin"])
    assert (tmp_path / "inf" / f"{Path(stem).stem}.json").exists()
