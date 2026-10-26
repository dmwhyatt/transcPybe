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


def _gtsinger_clip(root: Path, rel: str, words: list) -> None:
    import soundfile as sf

    path = root / f"{rel}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(words), encoding="utf-8")
    sf.write(path.with_suffix(".wav"), np.zeros(1600, np.float32), 16000)


def _word(text, phones, notes):
    """A GTSinger JSON word: phones as (phone, start, end), notes as (midi, start, end)."""
    return dict(word=text, start_time=phones[0][1], end_time=phones[-1][2], ph=[p for p, _, _ in phones],
                ph_start=[s for _, s, _ in phones], ph_end=[e for _, _, e in phones], note=[n for n, _, _ in notes],
                note_start=[s for _, s, _ in notes], note_end=[e for _, _, e in notes])


def test_gtsinger_vowels_in_each_transcription():
    from boundary.sources import gtsinger_is_vowel

    assert gtsinger_is_vowel("AH0", "English") and not gtsinger_is_vowel("DH", "English")
    assert gtsinger_is_vowel("iang", "Chinese") and not gtsinger_is_vowel("sh", "Chinese")
    assert gtsinger_is_vowel("i̥", "Japanese") and not gtsinger_is_vowel("ɰ̃", "Japanese")   # devoiced vowel; glide
    assert gtsinger_is_vowel("ɛː", "Korean") and not gtsinger_is_vowel("<SP>", "English")


def test_gtsinger_reads_notes_holds_out_singers_and_moves_onsets_with_care(tmp_path):
    from boundary.sources import gtsinger

    song = [
        _word("<SP>", [("<SP>", 0.0, 1.0)], [(0, 0.0, 1.0)]),
        # a cluster before the vowel: half way is from the note start to the first vowel
        _word("street", [("S", 1.0, 1.05), ("T", 1.05, 1.08), ("R", 1.08, 1.12), ("IY1", 1.12, 1.6)], [(60, 1.0, 1.6)]),
        # a held, pitched consonant longer than 0.25 s: the note starts there
        _word("word", [("W", 2.0, 2.7), ("ER1", 2.7, 3.0)], [(62, 2.0, 3.0)]),
    ]
    _gtsinger_clip(tmp_path, "English/EN-Alto-1/Breathy/Song A/Breathy_Group/0000", song)
    _gtsinger_clip(tmp_path, "English/EN-Alto-1/Breathy/Song A/Paired_Speech_Group/0000", song)   # spoken: skipped
    _gtsinger_clip(tmp_path, "English/EN-Alto-2/Vibrato/song a/Control_Group/0000", song)        # held-out singer
    _gtsinger_clip(tmp_path, "English/EN-Tenor-1/Vibrato/Other Song/Vibrato_Group/0000", song)
    _gtsinger_clip(tmp_path, "Japanese/JA-Soprano-1/Breathy/Uta/Control_Group/0000", [
        _word("き", [("k", 0.0, 0.06), ("i̥", 0.06, 0.1)], [(64, 0.0, 0.5)]),
        # evenly divided phones (b and ɾ both 0.1 s): timing not trusted, so the note keeps its start
        _word("ぶら", [("b", 1.0, 1.1), ("ɾ", 1.1, 1.2), ("a", 1.2, 1.5)], [(65, 1.0, 1.5)]),
    ])
    rows = {r["id"]: r for r in gtsinger(tmp_path, "between")}
    assert len(rows) == 4 and not any("Paired_Speech" in i for i in rows)
    splits = {r["singer"]: r["split"] for r in rows.values()}
    assert splits == {"EN-Alto-1": "excluded", "EN-Alto-2": "valid", "EN-Tenor-1": "train", "JA-Soprano-1": "train"}
    english = rows["gtsinger/English/EN-Tenor-1/Vibrato/Other Song/Vibrato_Group/0000"]
    assert english["notes"] == [[1.06, 1.6, 60], [2.0, 3.0, 62]] and english["technique"] == "Vibrato"
    japanese = next(r for r in rows.values() if r["language"] == "Japanese")
    assert japanese["notes"] == [[0.03, 0.5, 64], [1.0, 1.5, 65]]
    assert [r["notes"][0][0] for r in gtsinger(tmp_path, "consonant", languages=["English"])] == [1.0] * 3
    # annotations without their audio are a broken download: refuse, rather than write a short or empty list
    (tmp_path / "Japanese/JA-Soprano-1/Breathy/Uta/Control_Group/0000.wav").unlink()
    with pytest.raises(SystemExit, match="1 of 4 GTSinger clips"):
        list(gtsinger(tmp_path, "between"))
    with pytest.raises(SystemExit, match="no GTSinger annotations"):
        list(gtsinger(tmp_path / "nowhere", "between"))


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
    assert out.shape == (2, 123, 4)


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
    features.main([feat[0], str(tmp_path / "feat2")] + feat[2:] + ["--workers", "2"])   # two processes, same files
    z2 = np.load(tmp_path / "feat2" / f"{rows[0]['mix_id']}.npz")
    assert all(np.array_equal(z[k], z2[k]) for k in ("mel", "aux", "y"))

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


def test_a_shard_gives_back_what_the_feature_files_hold(tmp_path):
    from boundary import pack
    from boundary.train import load_split

    rng = np.random.default_rng(0)
    rows = []
    for i, n in enumerate((40, 7, 25)):
        path = tmp_path / f"m{i}.npz"
        np.savez(path, mel=rng.normal(size=(n, 80)).astype(np.float16), aux=rng.normal(size=(n, 5)).astype(np.float16),
                 y=targets([[0.1, 0.2, 60]], n).astype(np.float16), meta=np.array("x"))
        rows.append(dict(mix_id=f"m{i}", notes=[[0.1, 0.2, 60]], split="train" if i else "valid", features=str(path)))
    packed = pack.pack(rows, tmp_path / "s.npz")
    assert [r["index"] for r in packed] == [0, 1, 2] and all("features" not in r for r in packed)
    for split in ("train", "valid"):
        for a, b in zip(load_split(rows, split), load_split(packed, split)):
            assert a["id"] == b["id"] and all(np.array_equal(a[k], b[k]) for k in ("mel", "aux", "y"))


def test_prepare_in_parts_makes_the_whole_runs_mixes_into_shards_that_train(tmp_path, capsys):
    pytest.importorskip("torch")
    from boundary import mix, prepare, train

    items.write(tmp_path / "vocals.jsonl", _synthetic_vocals(tmp_path))
    common = ["--accomp", "synthetic", "--var-db", "20", "35", "--scratch", str(tmp_path / "scratch"),
              "--device", "cpu", "--f0-backend", "pyin", "--mix-workers", "1"]
    prepare.main([str(tmp_path / "vocals.jsonl"), str(tmp_path / "out"), "--part", "0/2", "--workers", "2"] + common)
    prepare.main([str(tmp_path / "vocals.jsonl"), str(tmp_path / "out"), "--part", "1/2", "--workers", "1"] + common)
    shards = sorted((tmp_path / "out" / "shards").glob("*.jsonl"))
    rows = [r for s in shards for r in items.read(s)]
    assert len(shards) == 2 and not any((tmp_path / "scratch").iterdir())   # the node-local work is cleared away

    # the parts' mixes are the ones a whole run makes: same ids, same labels
    mix.main([str(tmp_path / "vocals.jsonl"), str(tmp_path / "whole"), "--accomp", "synthetic", "--var-db", "20", "35"])
    whole = {r["mix_id"]: r["notes"] for r in items.read(tmp_path / "whole" / "mixes.jsonl")}
    assert {r["mix_id"]: r["notes"] for r in rows} == whole

    capsys.readouterr()
    prepare.main([str(tmp_path / "vocals.jsonl"), str(tmp_path / "out"), "--part", "0/2", "--workers", "2"] + common)
    assert "already done" in capsys.readouterr().out
    train.main([str(s) for s in shards] + ["--out", str(tmp_path / "run"), "--steps", "10", "--eval-every", "5",
                                           "--batch", "4", "--device", "cpu"])
    assert len((tmp_path / "run" / "log.jsonl").read_text().splitlines()) == 2


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


class _FakeMlflow:
    """Answers the few MLflow REST calls the tracker makes, and records them."""

    def __init__(self):
        import http.server
        import threading

        calls, logins, experiments = [], [], {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                calls.append(("experiments/get-by-name", None))
                logins.append(self.headers.get("Authorization"))
                name = self.path.split("experiment_name=")[1]
                if name in experiments:
                    self._reply(200, {"experiment": {"experiment_id": experiments[name]}})
                else:
                    self._reply(404, {"error_code": "RESOURCE_DOES_NOT_EXIST"})

            def do_POST(self):
                endpoint = self.path.split("/api/2.0/mlflow/")[1]
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((endpoint, body))
                logins.append(self.headers.get("Authorization"))
                if endpoint == "experiments/create":
                    experiments[body["name"]] = "7"
                    self._reply(200, {"experiment_id": "7"})
                elif endpoint == "runs/create":
                    self._reply(200, {"run": {"info": {"run_id": "run1"}}})
                else:
                    self._reply(200, {})

        self.calls, self.logins = calls, logins
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.uri = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def sent(self, endpoint):
        return [body for e, body in self.calls if e == endpoint]


def test_tracker_reports_a_run_and_resumes_it_in_the_same_mlflow_run(tmp_path, monkeypatch, capsys):
    from boundary.track import Tracker

    server = _FakeMlflow()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", server.uri)
    with Tracker.start(tmp_path, "r1/seed0", {"seed": 0, "features": "a.jsonl"}, {"node": "gpu-q-8"}) as t:
        t.log({"score": 0.8, "timing_median_ms": float("nan"), "augment": True, "extra": {"vowel": {"onset_f": 0.6}}}, 1000)
    assert server.sent("experiments/create") == [{"name": "boundary"}]
    assert server.sent("runs/create")[0]["run_name"] == "r1/seed0"
    params, metrics = server.sent("runs/log-batch")
    assert {p["key"]: p["value"] for p in params["params"]} == {"seed": "0", "features": "a.jsonl"}
    assert {m["key"]: (m["value"], m["step"]) for m in metrics["metrics"]} == {   # no NaN, no flags, nested sets named
        "score": (0.8, 1000), "extra.vowel.onset_f": (0.6, 1000)}
    assert server.sent("runs/update")[-1]["status"] == "FINISHED"
    assert (tmp_path / "mlflow_run_id").read_text().strip() == "run1"

    with pytest.raises(SystemExit), Tracker.start(tmp_path, "r1/seed0", {"seed": 0}, {}) as t:   # resumed, then stopped
        raise SystemExit(143)
    assert len(server.sent("runs/create")) == 1 and len(server.sent("runs/log-batch")) == 2   # no new run, no params
    assert [u["status"] for u in server.sent("runs/update")[-2:]] == ["RUNNING", "KILLED"]
    server.server.shutdown()


def test_tracker_gives_up_quietly_when_the_server_is_unreachable(tmp_path, monkeypatch, capsys):
    import socket

    from boundary.track import MAX_FAILURES, Tracker

    with socket.socket() as s:                       # a port that nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{port}")
    with Tracker.start(tmp_path, "r1/seed0", {}, {}) as t:
        for step in range(5):
            t.log({"score": 0.5}, step)
    out = capsys.readouterr().out
    assert out.count("[track]") == MAX_FAILURES + 1 and "stops reporting" in out
    assert t.uri is None and not (tmp_path / "mlflow_run_id").exists()

    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    with Tracker.start(tmp_path, "r1/seed0", {}, {}) as t:   # not configured: silent
        t.log({"score": 0.5}, 1)
    assert capsys.readouterr().out == ""


def test_tracker_signs_in_with_the_account_in_the_mlflow_credentials_file(tmp_path, monkeypatch):
    import base64

    from boundary.track import Tracker

    server = _FakeMlflow()
    (tmp_path / ".mlflow").mkdir()
    (tmp_path / ".mlflow" / "credentials").write_text(
        "[mlflow]\nmlflow_tracking_username = boundary\nmlflow_tracking_password = not-a-real-one\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", server.uri)
    with Tracker.start(tmp_path, "r1/seed0", {}, {}) as t:
        t.log({"score": 0.5}, 1)
    login = "Basic " + base64.b64encode(b"boundary:not-a-real-one").decode()
    assert server.logins and set(server.logins) == {login}
    server.server.shutdown()


def test_the_fourth_target_marks_only_re_sung_note_starts():
    from boundary.train import resung_target

    notes = [[1.0, 1.5, 60], [1.52, 2.0, 60], [2.0, 2.5, 62], [2.8, 3.0, 62]]   # re-sung; pitch change; after a rest
    t = resung_target(notes, 350)
    assert t[152] == 1 and t[151] == pytest.approx(2 / 3)
    assert t[100] == 0 and t[200] == 0 and t[280] == 0


def test_loading_adds_the_fourth_target_and_refuses_notes_that_disagree_with_the_stored_targets(tmp_path):
    from boundary.train import load_split

    notes = [[0.5, 1.0, 60], [1.02, 1.5, 60]]
    rows = []
    for name, stored in (("good", notes), ("bad", [[0.7, 1.0, 60], [1.02, 1.5, 60]])):
        path = tmp_path / f"{name}.npz"
        np.savez(path, mel=np.zeros((200, 80)), aux=np.zeros((200, 5)), y=targets(stored, 200).astype(np.float16))
        rows.append(dict(mix_id=name, notes=notes, features=str(path), split=name))
    good = load_split(rows, "good")[0]["y"]
    assert good.shape == (200, 4) and good[102, 3] == 1 and good[50, 3] == 0 and good[50, 0] == 1
    with pytest.raises(SystemExit, match="don't match its notes"):
        load_split(rows, "bad")


def test_a_three_output_checkpoint_from_before_still_loads_and_evaluates(tmp_path):
    pytest.importorskip("torch")
    from boundary.model import BoundaryNet, NetConfig, load, save
    from boundary.train import evaluate

    save(tmp_path / "old.pt", BoundaryNet(NetConfig(n_out=3)), step=1)
    model = load(tmp_path / "old.pt")
    notes = [[0.5, 1.0, 60], [1.02, 1.5, 60]]
    data = [dict(id="a", notes=notes, mel=np.zeros((200, 80), np.float32), aux=np.zeros((200, 5), np.float32),
                 y=np.zeros((200, 4), np.float32))]
    res = evaluate(model, data, "cpu")
    assert model.cfg.n_out == 3 and res["n_resung"] == 1 and "resung_head_recall" not in res
    res4 = evaluate(BoundaryNet().eval(), data, "cpu")
    assert {"resung_head_recall", "resung_head_precision", "resung_head_per_note"} <= set(res4)


def test_onset_track_can_leave_cutting_to_split_only_times():
    p = np.zeros(200)
    p[[30, 90]] = [0.9, 0.8]
    tr = onset_track(p, 0.5, split_policy="none", split_only_times=[1.2, 0.6])
    assert np.isnan(tr.strengths).all() and tr.times.tolist() == [pytest.approx(0.3), pytest.approx(0.9)]
    assert tr.split_only_times.tolist() == [0.6, 1.2]
    with pytest.raises(ValueError):
        onset_track(p, 0.5, split_policy="sometimes")


def test_the_decoder_cuts_one_held_note_only_at_split_only_times():
    import dataclasses

    from transcpybe.config import TranscriptionConfig
    from transcpybe.core import decode_notes
    from transcpybe.f0 import F0Track

    n = 150                                      # 1.5 s on one pitch, voiced from 0.1 to 1.4 s
    voiced = np.r_[np.zeros(10), np.ones(130), np.zeros(10)]
    track = F0Track(times=np.arange(n) * HOP_S, f0_hz=np.where(voiced > 0, 220.0, np.nan), periodicity=0.95 * voiced,
                    loudness_db=np.where(voiced > 0, -20.0, -60.0), loudness_gate=voiced, voicing=0.95 * voiced)
    cfg = dataclasses.replace(TranscriptionConfig(device="cpu"), timbre_split_ratio=0.0, split_on_voicing_breaks=False,
                              onset_split_min_dip_db=0.0)
    p = np.zeros(n)
    p[[40, 75]] = 0.9                           # onset-output peaks: they may move starts, never cut
    one = decode_notes(track, cfg, onsets=onset_track(p, 0.5, split_policy="none"))[0]
    two = decode_notes(track, cfg, onsets=onset_track(p, 0.5, split_policy="none", split_only_times=[0.75]))[0]
    assert len(one) == 1
    assert len(two) == 2 and abs(two[1].onset - 0.75) <= 0.05 and two[0].pitch == two[1].pitch == 57


def test_rwc_dev_reads_config_overrides_as_json_where_they_parse():
    from boundary.rwc_dev import parse_overrides

    assert parse_overrides(["measure_after_snap=true", "cut_settle_s=0.05", "onset_snap_direction=both"]) == dict(
        measure_after_snap=True, cut_settle_s=0.05, onset_snap_direction="both")
    with pytest.raises(SystemExit):
        parse_overrides(["cut_settle_s"])


def test_onset_track_carries_the_sounding_output_for_the_sustain_merge():
    from transcpybe.config import TranscriptionConfig
    from transcpybe.onsets import merge_sustained_boundaries
    from transcpybe.segment import Segment

    p, sounding = np.zeros(80), np.full(80, 0.99)
    p[15] = 0.9
    tr = onset_track(p, 0.5, sounding=sounding)
    assert tr.sounding is not None and len(tr.sounding) == len(tr.envelope_times) == 80
    parts = [Segment(0, 15, 62), Segment(15, 60, 64)]
    cfg = TranscriptionConfig(device="cpu", sustain_merge_above=0.98)
    assert len(merge_sustained_boundaries(parts, tr, np.arange(80) * HOP_S, cfg)) == 1
    assert onset_track(p, 0.5).sounding is None
