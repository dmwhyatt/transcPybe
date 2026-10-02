"""The realistic corpus: reproducible, and annotated the way it says it is."""

from __future__ import annotations

import numpy as np
import pytest

from transcpybe.corpus import (DEVELOPMENT_SEEDS, HELD_OUT_SEEDS, SR,
                               make_recording)


@pytest.fixture(scope="module")
def recording():
    return make_recording(DEVELOPMENT_SEEDS[0])


def test_a_seed_always_renders_the_same_recording(recording):
    again = make_recording(DEVELOPMENT_SEEDS[0])
    assert np.array_equal(recording.audio, again.audio)
    assert recording.reference() == again.reference()


def test_development_and_held_out_recordings_do_not_overlap():
    assert not set(DEVELOPMENT_SEEDS) & set(HELD_OUT_SEEDS)


@pytest.mark.parametrize("convention", [
    dict(release_point=0.0), dict(release_point=0.5), dict(release_point=1.0),
    dict(close_gaps=True), dict(intended_pitch=True),
])
def test_every_convention_gives_ordered_well_formed_notes(recording, convention):
    reference = recording.reference(**convention)
    assert len(reference) == len(recording.notes) > 0
    duration = recording.audio.shape[0] / SR
    for onset, offset, _ in reference:
        assert 0.0 <= onset < offset <= duration
    onsets = [onset for onset, _, _ in reference]
    assert onsets == sorted(onsets)


def test_release_point_moves_only_release_ends(recording):
    early, late = recording.reference(0.0), recording.reference(1.0)
    for note, (_, a, _), (_, b, _) in zip(recording.notes, early, late):
        assert (b > a) == (note.release > 0)


def test_closing_gaps_moves_only_consonant_ends(recording):
    open_, closed = recording.reference(), recording.reference(close_gaps=True)
    for i, (note, (_, a, _), (_, b, _)) in enumerate(zip(recording.notes, open_, closed)):
        if note.end_kind != "unvoiced" or i + 1 == len(recording.notes):
            assert a == b
        elif b != a:
            assert b == recording.notes[i + 1].onset


def test_sung_pitch_is_fractional_and_near_the_intended_note(recording):
    sung = [p for *_, p in recording.reference()]
    intended = [p for *_, p in recording.reference(intended_pitch=True)]
    assert all(float(p).is_integer() for p in intended)
    assert not all(float(p).is_integer() for p in sung)
    # half a semitone of tuning, 20 cents of drift, 30 of scatter at most-ish
    assert np.max(np.abs(np.subtract(sung, intended))) < 1.5
