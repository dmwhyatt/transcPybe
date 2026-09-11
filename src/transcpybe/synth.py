"""Synthetic singing-voice test signals with exact ground truth.

Used to validate and tune the transcriber without needing annotated audio. The
synthesiser is deliberately not a clean sine generator: it models the things
that actually break vocal transcribers -- vibrato, portamento, jitter, soft
attacks, consonant transients and the low-level residue that source separation
leaves behind -- so that scores here mean something on real stems.

Each generator returns ``(audio, ground_truth)`` where ``ground_truth`` is a
list of ``(onset_seconds, offset_seconds, midi_pitch)`` tuples.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

GroundTruth = List[Tuple[float, float, int]]

# Neutral-ish vowel formants (Hz) and their relative gains.
_FORMANTS = ((730.0, 1.0), (1090.0, 0.5), (2440.0, 0.22), (3400.0, 0.08))


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _resonator(signal: np.ndarray, freq: float, sample_rate: int, q: float = 12.0) -> np.ndarray:
    """Two-pole resonator, used to impose a formant on a harmonic source."""
    from scipy.signal import lfilter

    r = np.exp(-np.pi * freq / (q * sample_rate))
    theta = 2.0 * np.pi * freq / sample_rate
    a = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
    b = np.array([1.0 - r])
    return lfilter(b, a, signal)


def _adsr(n: int, sample_rate: int, attack: float, release: float) -> np.ndarray:
    """Amplitude envelope with raised-cosine attack and release."""
    envelope = np.ones(n)
    a = min(int(attack * sample_rate), n // 2)
    r = min(int(release * sample_rate), n // 2)
    if a > 0:
        envelope[:a] = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, a)))
    if r > 0:
        envelope[n - r:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, r)))
    return envelope


def synth_voice(
    f0_curve: np.ndarray,
    sample_rate: int,
    *,
    n_harmonics: int = 24,
    rolloff_db_per_octave: float = 12.0,
    jitter: float = 0.004,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesise a voiced sound following ``f0_curve`` (one value per sample).

    A harmonic stack with a realistic spectral rolloff is shaped by formant
    resonators. Small random jitter in the fundamental keeps CREPE from seeing
    an unnaturally perfect periodicity.
    """
    rng = rng or np.random.default_rng(0)
    n = f0_curve.shape[0]

    if jitter > 0:
        # Slow random walk, smoothed, so the perturbation is correlated in time
        # the way real vocal-fold jitter is.
        walk = np.cumsum(rng.normal(0.0, 1.0, n))
        walk = walk / (np.abs(walk).max() + 1e-9)
        f0_curve = f0_curve * (1.0 + jitter * walk)

    phase = 2.0 * np.pi * np.cumsum(f0_curve) / sample_rate
    nyquist = sample_rate / 2.0

    source = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        amplitude = 10.0 ** (-rolloff_db_per_octave * np.log2(k) / 20.0)
        # Silence harmonics that would alias.
        alive = (f0_curve * k) < nyquist * 0.95
        source += amplitude * np.sin(k * phase + rng.uniform(0, 2 * np.pi)) * alive

    shaped = np.zeros(n)
    for freq, gain in _FORMANTS:
        shaped += gain * _resonator(source, freq, sample_rate)

    peak = np.abs(shaped).max()
    return shaped / peak if peak > 0 else shaped


def _note_f0_curve(
    pitch: float,
    n: int,
    sample_rate: int,
    *,
    vibrato_hz: float,
    vibrato_cents: float,
    scoop_cents: float,
    scoop_s: float,
    glide_from: float | None,
    glide_s: float,
) -> np.ndarray:
    """Build the per-sample f0 contour for one note, in Hz."""
    t = np.arange(n) / sample_rate
    pitch_curve = np.full(n, float(pitch))

    if glide_from is not None and glide_s > 0:
        g = min(int(glide_s * sample_rate), n)
        if g > 0:
            ramp = np.linspace(0.0, 1.0, g)
            # Smoothstep rather than linear: real portamento eases in and out.
            ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            pitch_curve[:g] = glide_from + (pitch - glide_from) * ramp

    if scoop_cents and scoop_s > 0:
        s = min(int(scoop_s * sample_rate), n)
        if s > 0:
            decay = np.exp(-np.linspace(0.0, 4.0, s))
            pitch_curve[:s] -= (scoop_cents / 100.0) * decay

    if vibrato_cents > 0:
        # Vibrato fades in; singers rarely start a note with full vibrato.
        fade = np.clip(t / 0.25, 0.0, 1.0)
        pitch_curve += (vibrato_cents / 100.0) * fade * np.sin(2 * np.pi * vibrato_hz * t)

    return np.array([midi_to_hz(p) for p in pitch_curve])


def synth_phrase(
    notes: Sequence[Tuple[float, float, int]],
    sample_rate: int = 16000,
    *,
    vibrato_hz: float = 5.5,
    vibrato_cents: float = 0.0,
    scoop_cents: float = 0.0,
    scoop_s: float = 0.08,
    portamento_s: float = 0.0,
    consonants: bool = False,
    gains_db: Sequence[float] | None = None,
    detune_cents: Sequence[float] | None = None,
    attack_s: float = 0.025,
    release_s: float = 0.05,
    noise_db: float = -60.0,
    tail_s: float = 0.3,
    seed: int = 0,
) -> Tuple[np.ndarray, GroundTruth]:
    """Render a list of ``(onset, offset, pitch)`` notes as a vocal phrase.

    ``consonants`` prepends a short noise burst to each note, which is what
    gives repeated same-pitch notes a detectable attack. ``gains_db`` scales
    individual notes, and ``detune_cents`` shifts their pitch away from equal
    temperament without moving the ground truth, so that both dynamic range and
    intonation can be stress-tested.
    """
    rng = np.random.default_rng(seed)
    total = max(offset for _, offset, _ in notes) + tail_s
    n_total = int(total * sample_rate) + 1
    audio = np.zeros(n_total)

    previous_pitch: float | None = None
    for index, (onset, offset, pitch) in enumerate(notes):
        start = int(onset * sample_rate)
        n = max(int((offset - onset) * sample_rate), 1)

        glide_from = previous_pitch if (portamento_s > 0 and index > 0) else None
        offset_cents = detune_cents[index] if detune_cents is not None else 0.0
        sounding = float(pitch) + offset_cents / 100.0
        f0_curve = _note_f0_curve(
            sounding, n, sample_rate,
            vibrato_hz=vibrato_hz, vibrato_cents=vibrato_cents,
            scoop_cents=scoop_cents, scoop_s=scoop_s,
            glide_from=glide_from, glide_s=portamento_s,
        )

        voiced = synth_voice(f0_curve, sample_rate, rng=rng)
        voiced *= _adsr(n, sample_rate, attack_s, release_s)
        if gains_db is not None:
            voiced *= 10.0 ** (gains_db[index] / 20.0)

        if consonants:
            # Short filtered noise burst at the note start, like a plosive or
            # fricative. Without this, repeated notes at one pitch have no
            # attack for an onset detector to find.
            burst_n = min(int(0.030 * sample_rate), n)
            burst = rng.normal(0.0, 1.0, burst_n)
            burst *= np.exp(-np.linspace(0.0, 6.0, burst_n))
            burst = _resonator(burst, 2600.0, sample_rate, q=3.0)
            peak = np.abs(burst).max()
            if peak > 0:
                voiced[:burst_n] += 0.5 * burst / peak

        end = min(start + n, n_total)
        audio[start:end] += voiced[:end - start]
        previous_pitch = sounding

    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak

    if noise_db > -np.inf:
        audio += rng.normal(0.0, 10.0 ** (noise_db / 20.0), n_total)

    return audio.astype(np.float32), [(float(a), float(b), int(p)) for a, b, p in notes]


def add_separation_artifacts(
    audio: np.ndarray,
    ground_truth: GroundTruth,
    sample_rate: int = 16000,
    *,
    level_db: float = -34.0,
    seed: int = 1,
) -> np.ndarray:
    """Add the burbling residue that source separation leaves in silent gaps.

    This is the failure mode that matters most in practice: a pitch tracker will
    confidently report notes in passages where the singer is not singing at all,
    unless the transcriber gates on level. The residue here is pitched and
    slowly wandering, so it cannot be rejected by periodicity alone.
    """
    rng = np.random.default_rng(seed)
    out = audio.copy().astype(np.float64)
    n_total = out.shape[0]
    t = np.arange(n_total) / sample_rate

    wander = 200.0 * 2.0 ** (0.8 * np.sin(2 * np.pi * 0.13 * t) / 12.0)
    residue = np.zeros(n_total)
    for k in (1, 2, 3):
        residue += (1.0 / k) * np.sin(2 * np.pi * k * np.cumsum(wander) / sample_rate)
    residue += 0.3 * rng.normal(0.0, 1.0, n_total)

    # Amplitude-modulate so it burbles in and out rather than droning.
    residue *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28))
    residue *= 10.0 ** (level_db / 20.0) / (np.abs(residue).max() + 1e-9)

    # Only in the gaps between real notes.
    gap_mask = np.ones(n_total)
    for onset, offset, _ in ground_truth:
        lo = max(int((onset - 0.05) * sample_rate), 0)
        hi = min(int((offset + 0.05) * sample_rate), n_total)
        gap_mask[lo:hi] = 0.0

    out += residue * gap_mask
    peak = np.abs(out).max()
    return (out / peak if peak > 0 else out).astype(np.float32)


# --------------------------------------------------------------------------
# Named test cases
# --------------------------------------------------------------------------

def _sequence(pitches: Sequence[int], dur: float, gap: float, start: float = 0.4):
    notes, t = [], start
    for pitch in pitches:
        notes.append((t, t + dur, int(pitch)))
        t += dur + gap
    return notes


def case_scale(sample_rate: int = 16000):
    """Ascending then descending major scale, clearly separated notes."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]
    return synth_phrase(_sequence(pitches, 0.45, 0.15), sample_rate,
                        vibrato_cents=15.0, seed=11)


def case_vibrato(sample_rate: int = 16000):
    """Sustained notes with heavy vibrato -- must not fragment into many notes."""
    pitches = [67, 69, 67, 64, 65, 67]
    return synth_phrase(_sequence(pitches, 1.1, 0.2), sample_rate,
                        vibrato_cents=85.0, vibrato_hz=5.8, seed=12)


def case_legato(sample_rate: int = 16000):
    """Connected notes with no silence between them: pitch change is the only cue."""
    pitches = [60, 62, 64, 62, 60, 59, 60]
    notes, t = [], 0.4
    for pitch in pitches:
        notes.append((t, t + 0.5, int(pitch)))
        t += 0.5
    return synth_phrase(notes, sample_rate, vibrato_cents=20.0,
                        attack_s=0.04, release_s=0.02, seed=13)


def case_repeated(sample_rate: int = 16000):
    """The same pitch re-articulated. Only onsets can separate these."""
    notes, t = [], 0.4
    for _ in range(8):
        notes.append((t, t + 0.32, 62))
        t += 0.38
    return synth_phrase(notes, sample_rate, consonants=True,
                        vibrato_cents=10.0, attack_s=0.012, seed=14)


def case_rearticulated(sample_rate: int = 16000):
    """The same pitch re-articulated with no silence between the notes.

    A singer moving to a new syllable without stopping phonation. The pitch
    never changes and the sound never stops, so the pitch decoder sees one
    continuous note and only the onset detector can find the boundaries. This
    is the common case in real singing; ``case_repeated`` leaves a gap, which
    is the easy one.
    """
    notes, t = [], 0.4
    for _ in range(8):
        notes.append((t, t + 0.32, 62))
        t += 0.32
    return synth_phrase(notes, sample_rate, consonants=True,
                        vibrato_cents=10.0, attack_s=0.012,
                        release_s=0.004, seed=24)


def case_rearticulated_fast(sample_rate: int = 16000):
    """Re-articulation at speed, where each note is barely longer than a syllable."""
    notes, t = [], 0.4
    for _ in range(10):
        notes.append((t, t + 0.17, 67))
        t += 0.17
    return synth_phrase(notes, sample_rate, consonants=True,
                        vibrato_cents=8.0, attack_s=0.010,
                        release_s=0.004, seed=25)


def case_portamento(sample_rate: int = 16000):
    """Notes joined by audible glides, which tempt a decoder into extra notes."""
    pitches = [60, 67, 64, 69, 65, 60]
    notes, t = [], 0.4
    for pitch in pitches:
        notes.append((t, t + 0.6, int(pitch)))
        t += 0.6
    return synth_phrase(notes, sample_rate, portamento_s=0.14,
                        vibrato_cents=25.0, seed=15)


def case_scoops(sample_rate: int = 16000):
    """Notes approached from below, as in most pop singing."""
    pitches = [64, 67, 69, 67, 64, 62, 60]
    return synth_phrase(_sequence(pitches, 0.5, 0.12), sample_rate,
                        scoop_cents=180.0, scoop_s=0.10,
                        vibrato_cents=20.0, seed=16)


def case_wide_leaps(sample_rate: int = 16000):
    """Octave-plus leaps, which provoke octave errors in pitch trackers."""
    pitches = [48, 60, 50, 62, 55, 67, 52, 64, 48]
    return synth_phrase(_sequence(pitches, 0.45, 0.12), sample_rate,
                        vibrato_cents=20.0, seed=17)


def case_short_notes(sample_rate: int = 16000):
    """Rapid melisma near the duration floor."""
    pitches = [67, 69, 67, 65, 64, 65, 67, 65, 64, 62]
    return synth_phrase(_sequence(pitches, 0.14, 0.04), sample_rate,
                        consonants=False, attack_s=0.010, release_s=0.015,
                        vibrato_cents=0.0, seed=18)


def case_noisy_separation(sample_rate: int = 16000):
    """A scale buried in separation residue: the phantom-note stress test."""
    audio, truth = synth_phrase(
        _sequence([60, 64, 67, 72, 67, 64, 60], 0.5, 0.55), sample_rate,
        vibrato_cents=25.0, noise_db=-45.0, seed=19,
    )
    return add_separation_artifacts(audio, truth, sample_rate, level_db=-30.0), truth


def case_heavy_vibrato(sample_rate: int = 16000):
    """Operatic vibrato of +/-120 cents: wider than a semitone either way, so
    the contour spends much of its time nearer a neighbouring semitone than the
    note actually being sung."""
    pitches = [69, 72, 71, 69, 67, 69]
    return synth_phrase(_sequence(pitches, 1.3, 0.25), sample_rate,
                        vibrato_cents=120.0, vibrato_hz=6.2, seed=21)


def case_mistuned(sample_rate: int = 16000):
    """Singing consistently off equal temperament by up to 45 cents. The
    reported integer pitch should still be the note intended."""
    pitches = [62, 64, 66, 67, 66, 64, 62]
    detune = [-45.0, 30.0, -38.0, 42.0, -25.0, 35.0, -40.0]
    return synth_phrase(_sequence(pitches, 0.55, 0.15), sample_rate,
                        detune_cents=detune, vibrato_cents=20.0, seed=22)


def case_dynamics(sample_rate: int = 16000):
    """A phrase mixing full voice with genuinely quiet singing 25 dB down.

    Guards the quiet-note rejector: those soft notes are real and must survive,
    even though they sit far below the loudest notes in the phrase.
    """
    pitches = [60, 62, 64, 65, 67, 65, 64, 62]
    gains = [0.0, -25.0, -25.0, 0.0, 0.0, -22.0, -25.0, 0.0]
    return synth_phrase(_sequence(pitches, 0.5, 0.15), sample_rate,
                        gains_db=gains, vibrato_cents=20.0, noise_db=-70.0, seed=23)


def case_melisma_glides(sample_rate: int = 16000):
    """A fast run where every note is joined by a glide: the hardest
    combination, since short notes leave little stable pitch to measure."""
    pitches = [67, 69, 71, 72, 71, 69, 67, 65, 67]
    notes, t = [], 0.4
    for pitch in pitches:
        notes.append((t, t + 0.22, int(pitch)))
        t += 0.22
    return synth_phrase(notes, sample_rate, portamento_s=0.07,
                        vibrato_cents=15.0, attack_s=0.015, release_s=0.02, seed=24)


def case_breathy(sample_rate: int = 16000):
    """Breathy singing: broadband noise only 12 dB below the voiced signal."""
    pitches = [64, 65, 67, 69, 67, 65, 64]
    return synth_phrase(_sequence(pitches, 0.5, 0.15), sample_rate,
                        vibrato_cents=25.0, noise_db=-12.0, seed=25)


def case_low_register(sample_rate: int = 16000):
    """Low male register near the bottom of the tracked range, where harmonic
    spacing is tight and octave errors are most likely."""
    pitches = [41, 43, 45, 46, 45, 43, 41, 40]
    return synth_phrase(_sequence(pitches, 0.55, 0.15), sample_rate,
                        vibrato_cents=15.0, seed=26)


def case_loud_separation_residue(sample_rate: int = 16000):
    """Separation residue only 18 dB below the vocal: a badly separated stem."""
    audio, truth = synth_phrase(
        _sequence([62, 65, 69, 74, 69, 65, 62], 0.5, 0.6), sample_rate,
        vibrato_cents=25.0, noise_db=-40.0, seed=27,
    )
    return add_separation_artifacts(audio, truth, sample_rate, level_db=-18.0), truth


HARD_CASES = {
    "heavy_vibrato": case_heavy_vibrato,
    "mistuned": case_mistuned,
    "dynamics": case_dynamics,
    "melisma_glides": case_melisma_glides,
    "breathy": case_breathy,
    "low_register": case_low_register,
    "loud_residue": case_loud_separation_residue,
}


ALL_CASES = {
    "scale": case_scale,
    "vibrato": case_vibrato,
    "legato": case_legato,
    "repeated": case_repeated,
    "rearticulated": case_rearticulated,
    "rearticulated_fast": case_rearticulated_fast,
    "portamento": case_portamento,
    "scoops": case_scoops,
    "wide_leaps": case_wide_leaps,
    "short_notes": case_short_notes,
    "noisy_separation": case_noisy_separation,
}

ALL_CASES.update(HARD_CASES)
