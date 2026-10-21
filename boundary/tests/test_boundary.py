"""The boundary pipeline's pieces, and one run of the whole thing on three synthetic recordings."""
import json
from pathlib import Path

import numpy as np
import pytest

from boundary import HOP_S, items
from boundary.features import LEVEL_RANGE, targets
from boundary.infer import onset_track, pick_peaks
from boundary.sources import _vowel_onsets, textgrid_tiers
from boundary.train import Crops, onset_timing, resung, timing_summary


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


def test_progress_prints_rate_and_eta_at_the_end(capsys):
    from boundary.progress import Progress

    p = Progress(4, "demo", every_s=3600)
    p.tick(3)
    assert capsys.readouterr().out == ""                       # nothing until a minute passes or it finishes
    p.tick()
    line = capsys.readouterr().out
    assert "[demo] 4/4 (100%)" in line and "eta" in line


def test_resung_finds_only_same_pitch_notes_with_no_gap():
    notes = [[0, 1, 60], [1, 2, 60], [2.2, 3, 60], [3, 4, 62]]
    assert resung(notes) == [1]


def test_onset_timing_reports_a_consistent_lead_as_early():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    off = onset_timing(ref, ref - 0.06)                        # every estimate 60 ms early
    assert off == pytest.approx([-0.06] * 4)
    t = timing_summary(off)
    assert t["timing_median_ms"] == pytest.approx(-60.0) and t["early_share"] == 1.0 and t["late_share"] == 0.0
    assert onset_timing(ref, np.array([1.3])) == []           # nothing within 150 ms of anything
    assert timing_summary([])["n_timed"] == 0


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
    assert _vowel_onsets([[0.0, 1.0, 60]], tiers[1], 0.5) == [[0.04, 1.0, 60]]   # half way to the vowel


def test_relabel_moves_new_labels_by_the_pad_and_keeps_the_audio():
    from boundary.relabel import relabel

    rows = [dict(id="a", mix_id="000000-0-x", stem="/s.wav", pad_before=1.5, notes=[[1.5, 2.5, 60]], onsets="consonant")]
    out = relabel(rows, [dict(id="a", notes=[[0.04, 1.0, 60]], onsets="between-0.5")])
    assert out[0]["notes"] == [[1.54, 2.5, 60]] and out[0]["stem"] == "/s.wav" and out[0]["onsets"] == "between-0.5"
    with pytest.raises(SystemExit):
        relabel(rows, [])


def test_same_pitch_policy_keeps_splits_where_pitch_is_level():
    from types import SimpleNamespace

    n = 200
    f0 = np.where(np.arange(n) < 100, 220.0, 247.0)                    # a two-semitone step at frame 100
    track = SimpleNamespace(f0_hz=f0, voicing=np.ones(n))
    p = np.zeros(n)
    p[[50, 100]] = [0.9, 0.9]
    level = onset_track(p, 0.5, track=track, split_policy="same-pitch")
    assert np.isfinite(level.strengths[0]) and np.isnan(level.strengths[1])   # inside the note may split; at the step, snap only
    assert np.isfinite(onset_track(p, 0.5).strengths).all()                    # the default: every peak may split
    assert np.isnan(onset_track(p, 0.5, track=track, split_policy="same-pitch", split_threshold=0.95).strengths).all()


def test_across_gap_policy_splits_a_note_sung_again_after_a_consonant():
    from types import SimpleNamespace

    n = 200
    f0, v = np.full(n, 220.0), np.ones(n)
    f0[95:105], v[95:105] = np.nan, 0.0                                  # a 100 ms unvoiced consonant, same pitch either side
    f0[150:], v[150:] = np.nan, 0.0                                      # then a rest
    track = SimpleNamespace(f0_hz=f0, voicing=v)
    p = np.zeros(n)
    p[[96, 175]] = 0.9
    assert np.isnan(onset_track(p, 0.5, track=track, split_policy="same-pitch").strengths[0])      # blocked by the gap
    gap = onset_track(p, 0.5, track=track, split_policy="across-gap").strengths
    assert np.isfinite(gap[0]) and np.isnan(gap[1])                      # re-sung note splits; nothing voiced near the rest


def test_m4singer_holds_out_a_validation_singers_songs_from_every_singer(tmp_path):
    pretty_midi = pytest.importorskip("pretty_midi")
    import soundfile as sf
    from boundary.sources import m4singer

    for d in ("Alto-1#song", "Bass-1#song", "Bass-1#other"):
        (tmp_path / d).mkdir()
        sf.write(tmp_path / d / "0000.wav", np.zeros(4800), 48000)
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0)
        inst.notes.append(pretty_midi.Note(80, 60, 0.0, 0.05))
        pm.instruments.append(inst)
        pm.write(str(tmp_path / d / "0000.mid"))
    split = {r["id"].split("/")[1]: r["split"] for r in m4singer(tmp_path, valid_singers=("Alto-1",))}
    assert split == {"Alto-1#song": "valid", "Bass-1#song": "excluded", "Bass-1#other": "train"}


def test_model_keeps_one_output_per_frame():
    torch = pytest.importorskip("torch")
    from boundary.model import BoundaryNet

    out = BoundaryNet()(torch.zeros(2, 123, 80), torch.zeros(2, 123, 5))
    assert out.shape == (2, 123, 3)


def test_a_padded_batch_gives_each_recording_what_it_gives_alone():
    torch = pytest.importorskip("torch")
    from boundary.model import BoundaryNet

    torch.manual_seed(0)
    net = BoundaryNet().eval()
    a, b = torch.randn(1, 90, 80), torch.randn(1, 140, 80)
    xa, xb = torch.randn(1, 90, 5), torch.randn(1, 140, 5)
    mel = torch.zeros(2, 140, 80)
    aux = torch.zeros(2, 140, 5)
    mel[0, :90], mel[1], aux[0, :90], aux[1] = a[0], b[0], xa[0], xb[0]
    with torch.no_grad():
        batched = net(mel, aux, torch.tensor([90, 140]))
        assert torch.allclose(batched[0, :90], net(a, xa)[0], atol=1e-5)
        assert torch.allclose(batched[1], net(b, xb)[0], atol=1e-5)


def test_short_recordings_are_joined_to_others_not_padded():
    data = [dict(mel=np.full((30, 2), k, np.float32), aux=np.zeros((30, 1), np.float32),
                 y=np.zeros((30, 3), np.float32)) for k in (1, 2, 3)]
    mel, aux, y = Crops(data, 100).batch(np.random.default_rng(0), 8)
    assert mel.shape == (8, 100, 2) and (mel != 0).all()


def test_augment_varies_every_draw_and_leaves_the_targets_alone():
    from boundary.train import augment

    rng = np.random.default_rng(0)
    mel = rng.normal(size=(400, 80)).astype(np.float32)
    aux = np.full((400, 5), LEVEL_RANGE[0] + 0.01, np.float32)
    kept = mel.copy()
    (m1, a1), (m2, _) = augment(mel, aux, rng), augment(mel, aux, rng)
    assert m1.shape == mel.shape and not np.allclose(m1, m2) and np.array_equal(mel, kept)
    assert a1[:, 4].min() >= LEVEL_RANGE[0]
    data = [dict(mel=mel, aux=aux, y=rng.random((400, 3)).astype(np.float32))]
    _, _, y = Crops(data, 300).batch(np.random.default_rng(2), 3)
    _, _, y_aug = Crops(data, 300, augment=True).batch(np.random.default_rng(2), 3)
    assert np.array_equal(y, y_aug)


def test_silence_features_stay_in_range():
    from boundary.features import frame_features
    from transcpybe.f0 import F0Track

    audio = np.zeros(16000, np.float32)
    audio[4000:12000] = 0.1 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000)
    n = 101
    track = F0Track(times=np.arange(n) * HOP_S, f0_hz=np.full(n, np.nan), periodicity=np.zeros(n),
                    loudness_db=np.r_[np.full(30, -200.0), np.full(41, -20.0), np.full(30, -200.0)],
                    loudness_gate=np.zeros(n), voicing=np.zeros(n))
    mel, aux = frame_features(audio, track)
    assert aux[:, 4].min() >= LEVEL_RANGE[0] and np.isfinite(mel).all()
    assert mel.min() > -10                       # the -80 dB floor, not log(1e-6)


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


def test_whole_pipeline_on_three_synthetic_recordings(tmp_path: Path, capsys):
    pytest.importorskip("torch")
    import soundfile as sf
    from boundary import features, infer, mix, separate, train

    items.write(tmp_path / "vocals.jsonl", _synthetic_vocals(tmp_path))
    mix.main([str(tmp_path / "vocals.jsonl"), str(tmp_path / "mix"), "--accomp", "synthetic", "--keep-dry"])
    rows = items.read(tmp_path / "mix" / "mixes.jsonl")
    vocal = items.read(tmp_path / "vocals.jsonl")
    assert rows[0]["notes"][0][0] == pytest.approx(vocal[0]["notes"][0][0] + rows[0]["pad_before"], abs=1e-3)
    assert all(Path(r["mix"]).is_absolute() for r in rows)

    # rebuilt from its draws alone, elsewhere: the same id, the same labels, the same audio
    again = mix.render(vocal[0], 0, 0, 0, "synthetic", tmp_path / "mix2", True, mix.setup_key("synthetic"))
    assert (again["mix_id"], again["notes"], again["gain"]) == (rows[0]["mix_id"], rows[0]["notes"], rows[0]["gain"])
    assert np.array_equal(sf.read(again["mix"])[0], sf.read(rows[0]["mix"])[0])
    # a rerun renders nothing that is already there; a changed label is a different mix
    stamp = Path(rows[0]["mix"]).stat().st_mtime_ns
    mix.main([str(tmp_path / "vocals.jsonl"), str(tmp_path / "mix"), "--accomp", "synthetic", "--keep-dry"])
    assert Path(rows[0]["mix"]).stat().st_mtime_ns == stamp
    moved = dict(vocal[0], notes=[[a + 0.01, b, p] for a, b, p in vocal[0]["notes"]])
    assert mix.render(moved, 0, 0, 0, "synthetic", tmp_path / "mix2", True, mix.setup_key("synthetic"))["mix_id"] != rows[0]["mix_id"]

    separate.main([str(tmp_path / "mix" / "mixes.jsonl"), str(tmp_path / "sep"), "--backend", "dry"])
    feat = [str(tmp_path / "sep" / "stems.jsonl"), str(tmp_path / "feat"), "--f0-backend", "pyin", "--device", "cpu"]
    features.main(feat)
    z = np.load(tmp_path / "feat" / f"{rows[0]['mix_id']}.npz")
    assert z["mel"].shape[0] == z["aux"].shape[0] == z["y"].shape[0] and z["y"][:, 0].max() == 1
    features.main(feat)
    assert "(0 computed, 3 reused)" in capsys.readouterr().out

    run = [str(tmp_path / "feat" / "features.jsonl"), "--out", str(tmp_path / "run"), "--steps", "20",
           "--eval-every", "10", "--batch", "4", "--device", "cpu",
           "--extra-valid", f"again={tmp_path / 'feat' / 'features.jsonl'}"]
    train.main(run)
    assert (tmp_path / "run" / "best.pt").exists() and len((tmp_path / "run" / "log.jsonl").read_text().splitlines()) == 2
    rec = json.loads((tmp_path / "run" / "log.jsonl").read_text().splitlines()[-1])
    assert {"onset_p", "onset_r", "onsets_per_note", "timing_median_ms", "early_share"} <= set(rec)
    assert rec["extra"]["again"]["onset_f"] == rec["onset_f"]  # the same set evaluated twice agrees
    best = (tmp_path / "run" / "best.pt").stat().st_mtime_ns
    train.main(run)                                   # finished: resumes past the end, overwrites nothing
    assert "resuming" in capsys.readouterr().out and (tmp_path / "run" / "best.pt").stat().st_mtime_ns == best
    assert len((tmp_path / "run" / "log.jsonl").read_text().splitlines()) == 2
    with pytest.raises(SystemExit):
        train.main(run[:-4] + ["--seed", "1", "--device", "cpu"])
    capsys.readouterr()
    train.main([str(tmp_path / "feat" / "features.jsonl"), "--eval-only", str(tmp_path / "run" / "best.pt"), "--device", "cpu"])
    assert "timing_median_ms" in json.loads(capsys.readouterr().out)   # a different run is refused, not mixed in

    # a smaller, augmented network stopped early; its checkpoint carries its size
    small = [str(tmp_path / "feat" / "features.jsonl"), "--out", str(tmp_path / "small"), "--steps", "40",
             "--stop-after", "10", "--eval-every", "5", "--batch", "4", "--device", "cpu",
             "--channels", "8,16", "--hidden", "16", "--augment"]
    train.main(small)
    assert [json.loads(line)["step"] for line in (tmp_path / "small" / "log.jsonl").read_text().splitlines()] == [5, 10]
    from boundary.model import load
    assert load(tmp_path / "small" / "best.pt").cfg.channels == (8, 16)

    stem = items.read(tmp_path / "sep" / "stems.jsonl")[0]["stem"]
    infer.main([str(tmp_path / "run" / "best.pt"), stem, "--out-dir", str(tmp_path / "inf"), "--device", "cpu",
                "--f0-backend", "pyin"])
    assert (tmp_path / "inf" / f"{Path(stem).stem}.json").exists()


def test_var_db_sets_the_accompaniment_level_and_the_mix_backend_uses_the_mix(tmp_path):
    from boundary import mix, separate

    items.write(tmp_path / "vocals.jsonl", _synthetic_vocals(tmp_path)[:1])
    mix.main([str(tmp_path / "vocals.jsonl"), str(tmp_path / "m"), "--accomp", "synthetic", "--var-db", "30", "30"])
    rows = items.read(tmp_path / "m" / "mixes.jsonl")
    assert rows[0]["var_db"] in (30.0, None)                    # None: the 10% drawn with no accompaniment
    separate.main([str(tmp_path / "m" / "mixes.jsonl"), str(tmp_path / "s"), "--backend", "mix"])
    assert items.read(tmp_path / "s" / "stems.jsonl")[0]["stem"] == rows[0]["mix"]


def test_mp3_round_trip_is_realigned_even_where_the_encoder_delays_it():
    from boundary.mix import MIX_SR, _mp3

    rng = np.random.default_rng(0)
    x = np.zeros((MIX_SR * 3, 2), np.float32)
    for k in range(6):
        a = int((0.3 + 0.4 * k) * MIX_SR)
        x[a:a + 400] = rng.standard_normal((400, 2)) * 0.5
    for kbps in (320, 128, 48):                                # 48 kbps is where libsndfile's MP3 comes back late
        y, lag = _mp3(x, kbps)
        n = len(x)
        cc = np.fft.irfft(np.fft.rfft(x.sum(1), 2 * n).conj() * np.fft.rfft(y.sum(1), 2 * n))
        assert y.shape == x.shape and int(np.argmax(np.r_[cc[-500:], cc[:501]]) - 500) == 0


def test_separate_fails_loudly_when_a_stem_is_missing(tmp_path):
    from boundary import separate

    items.write(tmp_path / "mixes.jsonl", [dict(mix_id="a", mix=str(tmp_path / "a.flac"), dry=str(tmp_path / "gone.flac"),
                                                notes=[])])
    with pytest.raises(SystemExit):
        separate.main([str(tmp_path / "mixes.jsonl"), str(tmp_path / "sep"), "--backend", "dry"])
    assert items.read(tmp_path / "sep" / "failed.jsonl")[0]["mix_id"] == "a"
