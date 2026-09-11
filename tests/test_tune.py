"""Fitting the transcriber's parameters, and not fooling ourselves about it.

A search over enough configurations finds one that flatters whatever it was
scored on. These pin the parts that exist to prevent that: a split that stays
put, a score that cannot be won by transcribing less, and a replay that is the
same code the transcriber runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcpybe.benchmark.dataset import Item
from transcpybe.benchmark.tune import (
    DEFAULT_SPACE, Split, StageCache, fbeta, make_split, random_search,
    sample_config, score_config, upstream_fingerprint,
)
from transcpybe.config import TranscriptionConfig


def test_fbeta_weights_recall_above_one():
    """Against a reference listing only part of what was sung, plain F rewards
    transcribing less: precision is capped by material the reference omits."""
    precise = fbeta(0.9, 0.5, beta=1.5)
    recalling = fbeta(0.5, 0.9, beta=1.5)

    assert recalling > precise
    assert fbeta(0.9, 0.5, beta=1.0) == pytest.approx(0.643, abs=0.001)
    assert fbeta(0.0, 0.0) == 0.0
    assert fbeta(1.0, 1.0) == pytest.approx(1.0)


def test_split_is_disjoint_and_covers_everything():
    ids = [f"song_{i:03d}" for i in range(100)]

    split = make_split(ids, seed=0)

    assert set(split.fit) | set(split.select) | set(split.report) == set(ids)
    assert not set(split.fit) & set(split.select)
    assert not set(split.fit) & set(split.report)
    assert not set(split.select) & set(split.report)


def test_split_is_stratified_so_one_era_cannot_land_on_one_side():
    """1950s and 2020s recordings differ in everything that affects a pitch
    tracker; an unstratified draw measures era rather than parameters."""
    ids = [f"song_{i:03d}" for i in range(120)]
    decades = {i: ("1950s" if n < 40 else "1980s" if n < 80 else "2010s")
               for n, i in enumerate(ids)}

    split = make_split(ids, decades, seed=0)

    for part in (split.fit, split.select, split.report):
        present = {decades[i] for i in part}
        assert present == {"1950s", "1980s", "2010s"}, (
            f"every part must see every decade, saw {present}")


def test_small_groups_still_leave_something_to_select_and_report_with():
    split = make_split([f"s{i}" for i in range(3)], seed=0)

    assert split.fit and split.select and split.report


def test_split_round_trips_so_the_report_set_stays_the_report_set(tmp_path):
    split = make_split([f"song_{i}" for i in range(30)], seed=7)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split.to_dict()))

    restored = Split.from_dict(json.loads(path.read_text()))

    assert restored.report == split.report
    assert restored.seed == 7


def test_sampled_configs_stay_inside_the_space_and_keep_types():
    base = TranscriptionConfig()
    import random

    rng = random.Random(0)
    for _ in range(30):
        candidate = sample_config(base, DEFAULT_SPACE, rng)
        for name, (low, high) in DEFAULT_SPACE.items():
            value = getattr(candidate, name)
            assert low <= value <= high
            assert isinstance(value, type(getattr(base, name)))


def test_cache_fingerprint_ignores_the_parameters_being_swept():
    """Otherwise every trial would recompute f0 and the sweep would be useless."""
    import dataclasses

    base = TranscriptionConfig()
    swept = dataclasses.replace(base, **{k: (v[0] + v[1]) / 2
                                         for k, v in DEFAULT_SPACE.items()})
    upstream = dataclasses.replace(base, hop_ms=20)

    assert upstream_fingerprint(swept) == upstream_fingerprint(base)
    assert upstream_fingerprint(upstream) != upstream_fingerprint(base)


def test_cache_discards_tracks_built_with_different_upstream_settings(tmp_path):
    """Reusing them would sweep parameters against the wrong f0."""
    import dataclasses

    import numpy as np

    base = TranscriptionConfig()
    cache = StageCache(tmp_path / "c", base)
    stale = Path(cache.directory) / "something.npz"
    np.savez(stale, x=np.zeros(3))
    assert stale.exists()

    StageCache(tmp_path / "c", dataclasses.replace(base, hop_ms=20))

    assert not stale.exists()


@pytest.mark.slow
def test_replay_from_cached_tracks_matches_the_full_transcription():
    """The equivalence the whole approach rests on."""
    from transcpybe.core import decode_notes, transcribe_audio
    from transcpybe.synth import case_rearticulated, case_scale

    config = TranscriptionConfig(f0_backend="pyin", device="cpu")
    for make_case in (case_scale, case_rearticulated):
        audio, _ = make_case()
        full = transcribe_audio(audio, config)
        replayed, _, _ = decode_notes(full.f0_track, config, onsets=full.onsets)

        assert [(n.onset, n.offset, n.pitch) for n in replayed] == \
               [(n.onset, n.offset, n.pitch) for n in full.notes]


@pytest.mark.slow
def test_search_recovers_a_detuned_parameter_on_held_out_recordings(tmp_path):
    """From a split gate so strict that repeated notes all merge."""
    import soundfile as sf

    from transcpybe.synth import (case_rearticulated,
                                           case_rearticulated_fast, case_scale)

    items, references = [], {}
    for i in range(12):
        make_case = (case_rearticulated, case_rearticulated_fast, case_scale)[i % 3]
        audio, truth = make_case()
        path = tmp_path / f"song_{i:02d}.wav"
        sf.write(path, audio, 16000)
        item_id = f"song_{i:02d}"
        items.append(Item(item_id=item_id, audio=path, annotation=path))
        references[item_id] = [(a, b, float(p)) for a, b, p in truth]

    detuned = TranscriptionConfig(f0_backend="pyin", device="cpu",
                                  onset_split_strength_ratio=1.6)
    split = make_split([i.item_id for i in items], fractions=(0.5, 0.25, 0.25), seed=0)
    by_id = {i.item_id: i for i in items}
    fit = [by_id[k] for k in split.fit]
    select = [by_id[k] for k in split.select]
    report = [by_id[k] for k in split.report]

    cache = StageCache(tmp_path / "cache", detuned)
    result = random_search(detuned, fit, select, references, cache, trials=25, seed=1)

    before = score_config(detuned, report, references, cache)
    after = score_config(result["best"]["config"], report, references, cache)

    assert after["fbeta"] > before["fbeta"] + 0.1, "held-out score must improve"
    assert after["recall"] > before["recall"]
    assert result["best"]["changes"]["onset_split_strength_ratio"] < 1.0


def _melody(pitches, drift=0.0, start=0.0):
    return [(start + i * 0.5 * (1 + drift), start + i * 0.5 * (1 + drift) + 0.3, float(p))
            for i, p in enumerate(pitches)]


MELODY = [60, 62, 64, 64, 64, 65, 67, 65, 64, 62, 60, 59, 60, 62, 64]


def test_sequence_score_survives_a_clock_that_does_not_match():
    """The property the whole approach rests on for a notated reference."""
    from transcpybe.benchmark.metrics import score_sequence

    exact = score_sequence(_melody(MELODY), _melody(MELODY))
    drifted = score_sequence(_melody(MELODY), _melody(MELODY, drift=0.03, start=2.0))

    assert exact.f == pytest.approx(1.0)
    assert drifted.f == pytest.approx(1.0), "a sequence has no clock to disagree with"


def test_sequence_score_still_sees_merged_repeated_notes():
    """The defect that matters most here shortens the sequence."""
    from transcpybe.benchmark.metrics import score_sequence

    merged = [p for i, p in enumerate(MELODY) if i not in (3, 4)]

    score = score_sequence(_melody(MELODY), _melody(merged))

    assert score.recall < 0.9
    assert score.matched == len(merged)


def test_interval_score_is_unchanged_by_transposition():
    from transcpybe.benchmark.metrics import score_sequence

    score = score_sequence(_melody(MELODY), _melody([p + 7 for p in MELODY]))

    assert score.f < 0.2, "the pitches are all wrong"
    assert score.interval_f == pytest.approx(1.0), "the shape is identical"


def test_the_chance_floor_is_measured_because_it_is_not_zero():
    """Two pop melodies in a key share most of their notes."""
    from transcpybe.benchmark.metrics import sequence_chance_floor

    melodies = [_melody([60, 62, 64, 65, 67, 69, 71, 72] * 3),
                _melody([72, 71, 69, 67, 65, 64, 62, 60] * 3),
                _melody([64, 65, 67, 69, 71, 72, 74, 76] * 3)]

    floor = sequence_chance_floor(melodies, melodies)

    assert floor["n"] == 3
    assert floor["f"] > 0.15, "unrelated melodies in one key are not dissimilar"
    assert floor["f"] < 0.9, "nor are they the same melody"


@pytest.mark.slow
def test_tuning_can_optimise_the_sequence_objective(tmp_path):
    """Against a reference with no usable clock, this is the only honest target."""
    import soundfile as sf

    from transcpybe.synth import case_rearticulated, case_rearticulated_fast

    items, references = [], {}
    for i in range(10):
        make_case = (case_rearticulated, case_rearticulated_fast)[i % 2]
        audio, truth = make_case()
        path = tmp_path / f"song_{i:02d}.wav"
        sf.write(path, audio, 16000)
        item_id = f"song_{i:02d}"
        items.append(Item(item_id=item_id, audio=path, annotation=path))
        # The reference is on a different clock, as a notated one would be.
        references[item_id] = [(a * 1.04 + 1.5, b * 1.04 + 1.5, float(p))
                               for a, b, p in truth]

    detuned = TranscriptionConfig(f0_backend="pyin", device="cpu",
                                  onset_split_strength_ratio=1.6)
    split = make_split([i.item_id for i in items], fractions=(0.5, 0.25, 0.25), seed=0)
    by_id = {i.item_id: i for i in items}
    cache = StageCache(tmp_path / "cache", detuned)

    result = random_search(detuned, [by_id[k] for k in split.fit],
                           [by_id[k] for k in split.select], references, cache,
                           trials=20, seed=1, objective="sequence")

    report = [by_id[k] for k in split.report]
    before = score_config(detuned, report, references, cache, objective="sequence")
    after = score_config(result["best"]["config"], report, references, cache,
                         objective="sequence")

    assert after["fbeta"] > before["fbeta"], "held-out sequence score must improve"
    assert after["recall"] > before["recall"]


@pytest.mark.slow
def test_tuning_with_alignment_warps_the_references_first(tmp_path, capsys):
    """Tuning against unaligned notated references optimises the mismatch.

    On BiMMuDa that scores 0.06 however good the transcription is, so a search
    against it would be chasing noise while reporting numbers.
    """
    import soundfile as sf

    from transcpybe.benchmark.cli import main
    from transcpybe.synth import synth_phrase

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    melody = [60, 62, 64, 65, 67, 65, 64, 62, 60, 59, 60, 62]
    for song in range(12):
        notes, t = [], 1.0
        for i in range(50):
            notes.append((t, t + 0.40, melody[i % len(melody)]))
            t += 0.55
        audio, truth = synth_phrase(notes, 16000, consonants=True, vibrato_cents=12.0,
                                    attack_s=0.012, release_s=0.004, seed=song)
        sf.write(audio_dir / f"song_{song:02d}.wav", audio, 16000)
        # Notated: the right notes at one nominal tempo, with its own start.
        (audio_dir / f"song_{song:02d}_notes.csv").write_text(
            "\n".join(f"{a * 0.98 + 0.35:.4f},{b * 0.98 + 0.35:.4f},{p}"
                      for a, b, p in truth))

    common = [str(audio_dir), "--tune", "--tune-trials", "4", "--f0-backend", "pyin",
              "--device", "cpu", "--split-file", str(tmp_path / "split.json"),
              "--tuned-config", str(tmp_path / "tuned.json"), "-q"]

    main(common + ["--align", "--f0-cache", str(tmp_path / "aligned")])
    aligned_out = capsys.readouterr().out
    main(common + ["--f0-cache", str(tmp_path / "raw")])
    raw_out = capsys.readouterr().out

    assert "Aligned 12 references" in aligned_out

    def baseline_fbeta(text):
        line = next(l for l in text.splitlines() if l.startswith("baseline"))
        return float(line.split()[1])

    assert baseline_fbeta(aligned_out) > 0.8
    assert baseline_fbeta(raw_out) < 0.4, (
        "precondition: without warping there is nothing worth tuning against")


def test_an_onset_objective_without_alignment_says_so(tmp_path, capsys):
    """Silence here is how a search against a mismatch looks like a result."""
    import soundfile as sf

    from transcpybe.benchmark.cli import main
    from transcpybe.synth import case_scale

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for song in range(8):
        audio, truth = case_scale()
        sf.write(audio_dir / f"song_{song}.wav", audio, 16000)
        (audio_dir / f"song_{song}_notes.csv").write_text(
            "\n".join(f"{a},{b},{p}" for a, b, p in truth))

    main([str(audio_dir), "--tune", "--tune-trials", "1", "--f0-backend", "pyin",
          "--device", "cpu", "--f0-cache", str(tmp_path / "c"),
          "--split-file", str(tmp_path / "s.json"),
          "--tuned-config", str(tmp_path / "t.json"), "-q"])

    assert "pass --align" in capsys.readouterr().out
