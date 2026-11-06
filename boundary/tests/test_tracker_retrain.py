"""Retraining on another pitch tracker's features (TRACKER-RETRAIN-PLAN.md): --label-voicing, swap_features and
voicing_map."""
import json

import numpy as np
import pytest

from boundary.features import targets

NOTES = [[0.5, 1.0, 60], [1.2, 1.8, 62]]           # the second opens on a vowel: its vowel target is a voicing onset
LANDMARKS = [[0.42, 0.58, False], [1.2, 1.2, True]]


def _row(tmp_path, name, voiced_from, mel=None, y=None, **extra):
    aux = np.zeros((250, 5))
    aux[voiced_from:, 3] = 0.9
    path = tmp_path / f"{name}.npz"
    np.savez(path, mel=np.zeros((250, 80)) if mel is None else mel, aux=aux,
             y=targets(NOTES, 250).astype(np.float16) if y is None else y)
    return dict(mix_id="a", notes=NOTES, landmarks=LANDMARKS, features=str(path), split="train", **extra)


def test_label_voicing_places_vowel_targets_with_the_twins_voicing(tmp_path):
    from boundary.train import load_split

    crepe = _row(tmp_path, "crepe", 127)            # CREPE hears the vowel at 1.27 s
    other = _row(tmp_path, "other", 133)            # the other tracker at 1.33 s
    own = load_split([other], "train", "landmarks")[0]
    twin = load_split([other], "train", "landmarks", twins={"a": crepe})[0]
    assert own["vowel_onsets"] == [0.58, 1.33] and twin["vowel_onsets"] == [0.58, 1.27]
    assert twin["y"][127, 4] == 1 and own["y"][133, 4] == 1
    assert np.array_equal(twin["aux"], own["aux"])  # the network still gets the other tracker's features


def test_label_voicing_refuses_a_twin_that_is_not_the_same_mix(tmp_path):
    from boundary.train import load_split

    other = _row(tmp_path, "other", 133)
    with pytest.raises(SystemExit, match="not in the --label-voicing manifest"):
        load_split([other], "train", "landmarks", twins={})
    y = targets(NOTES, 250).astype(np.float16)
    y[10, 2] = 1.0
    with pytest.raises(SystemExit, match="stored targets differ"):
        load_split([other], "train", "landmarks", twins={"a": _row(tmp_path, "t1", 127, y=y)})
    mel = np.zeros((250, 80))
    mel[5, 5] = 0.5
    with pytest.raises(SystemExit, match="spectrogram differs"):
        load_split([other], "train", "landmarks", twins={"a": _row(tmp_path, "t2", 127, mel=mel)})
    moved = _row(tmp_path, "t3", 127)
    moved["landmarks"] = [[0.40, 0.58, False], [1.2, 1.2, True]]
    with pytest.raises(SystemExit, match="notes or landmarks differ"):
        load_split([other], "train", "landmarks", twins={"a": moved})


def _shards(folder, backend, rows, vocals="/v.jsonl"):
    folder.mkdir(parents=True)
    (folder / "part-0000-of-0001.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "part-0000-of-0001.done").write_text(json.dumps(dict(
        vocals=vocals, accomp="/musdb", accomp_depth=2, var_db=[20.0, 35.0], variants=1, seed=0, crepe_model="full",
        f0_backend=backend), sort_keys=True))


def test_swap_features_points_the_labelled_rows_at_the_new_shards(tmp_path):
    from boundary import items as manifest
    from boundary.swap_features import main

    old, new = tmp_path / "old" / "shards", tmp_path / "new" / "shards"
    base = [dict(mix_id=f"m{i}", pad_before=0.1 * i, setup="s", variant=0, notes=NOTES) for i in range(3)]
    _shards(old, None, [dict(b, shard=str(old / "part-0000-of-0001.npz"), index=i) for i, b in enumerate(base)])
    _shards(new, "pesto", [dict(b, shard=str(new / "part-0000-of-0001.npz"), index=2 - i)
                           for i, b in enumerate(reversed(base))])
    labelled = [dict(b, shard=str(old / "part-0000-of-0001.npz"), index=i, landmarks=LANDMARKS, split="train")
                for i, b in enumerate(base)]
    manifest.write(tmp_path / "all-lm.jsonl", labelled)
    main([str(tmp_path / "all-lm.jsonl"), str(tmp_path / "out.jsonl"), "--shards", str(new),
          "--expect-f0-backend", "pesto"])
    out = manifest.read(tmp_path / "out.jsonl")
    assert [r["shard"] for r in out] == [str(new / "part-0000-of-0001.npz")] * 3
    assert [r["index"] for r in out] == [0, 1, 2] and all(r["landmarks"] == LANDMARKS for r in out)
    with pytest.raises(SystemExit, match="were not made with pyin"):
        main([str(tmp_path / "all-lm.jsonl"), str(tmp_path / "x.jsonl"), "--shards", str(new),
              "--expect-f0-backend", "pyin"])


def test_swap_features_stops_when_the_mixes_would_differ(tmp_path):
    from boundary import items as manifest
    from boundary.swap_features import main

    old, new = tmp_path / "old" / "shards", tmp_path / "new" / "shards"
    row = dict(mix_id="m0", pad_before=0.1, setup="s", variant=0, notes=NOTES)
    _shards(old, None, [dict(row, shard=str(old / "p.npz"), index=0)])
    _shards(new, "pesto", [dict(row, shard=str(new / "p.npz"), index=0)], vocals="/other.jsonl")
    manifest.write(tmp_path / "all-lm.jsonl", [dict(row, shard=str(old / "p.npz"), index=0)])
    with pytest.raises(SystemExit, match="vocals differ"):
        main([str(tmp_path / "all-lm.jsonl"), str(tmp_path / "out.jsonl"), "--shards", str(new),
              "--expect-f0-backend", "pesto"])
    (new / "part-0000-of-0001.done").unlink()
    _shards(tmp_path / "new2" / "shards", "pesto", [dict(row, pad_before=0.2, shard="q.npz", index=0)])
    with pytest.raises(SystemExit, match="pad_before differ"):
        main([str(tmp_path / "all-lm.jsonl"), str(tmp_path / "out.jsonl"), "--shards",
              str(tmp_path / "new2" / "shards"), "--expect-f0-backend", "pesto"])


def test_the_voicing_map_puts_a_lower_confidence_on_crepes_scale(tmp_path):
    from boundary import items as manifest
    from boundary.voicing_map import apply, fit, main, map_periodicity
    from transcpybe.f0 import F0Track

    rng = np.random.default_rng(0)
    crepe = rng.beta(2, 1, 20000)
    other = crepe ** 3                                # same ranking, lower scale: fewer frames pass 0.3
    m = fit(other, crepe)
    assert np.mean(other >= 0.3) < 0.75 < np.mean(crepe >= 0.3)
    assert abs(np.mean(map_periodicity(other, m) >= 0.3) - np.mean(crepe >= 0.3)) < 0.01

    n = 50
    named = {"backend": "pesto"} if "backend" in F0Track.__dataclass_fields__ else {}   # transcPybe c13e68b on
    track = F0Track(times=np.arange(n) * 0.01, f0_hz=np.where(np.arange(n) < 40, 220.0, np.nan),
                    periodicity=np.full(n, 0.2), loudness_db=np.zeros(n), loudness_gate=np.full(n, 0.5),
                    voicing=np.full(n, 0.1), **named)
    out = apply(track, m)
    p = float(map_periodicity(np.array([0.2]), m)[0])
    assert out.periodicity[0] == pytest.approx(p) and out.voicing[0] == pytest.approx(0.5 * p)
    assert out.voicing[45] == 0.0 and getattr(out, "backend", "pesto") == "pesto"

    rows = {}
    for name, p in (("crepe", crepe[:250]), ("other", other[:250])):
        aux = np.zeros((250, 5))
        aux[:, 2], aux[:, 3] = p, p
        np.savez(tmp_path / f"{name}.npz", mel=np.zeros((250, 80)), aux=aux, y=np.zeros((250, 3)))
        rows[name] = [dict(mix_id="a", split="valid", features=str(tmp_path / f"{name}.npz"))]
        manifest.write(tmp_path / f"{name}.jsonl", rows[name])
    main([str(tmp_path / "crepe.jsonl"), str(tmp_path / "other.jsonl"), str(tmp_path / "map.json")])
    report = json.loads((tmp_path / "map.json").read_text())
    assert report["frames"] == 250 and report["voiced_share"]["other"] < report["voiced_share"]["crepe"]
    assert report["voiced_share"]["other_mapped"] == pytest.approx(report["voiced_share"]["crepe"], abs=0.02)
