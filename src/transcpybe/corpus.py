"""A randomised corpus of realistic synthetic singing, with stated annotation rules.

The built-in suite (``synth.py``) is a set of cases, each written to break one
stage, and it scores 1.000 on most of them. That is its job, and it is also why
it cannot rank two pipelines that both pass it: there is nothing left for the
better one to win. Nor does it contain the conditions that decide where real
note ends fall. Every case stops a note's audio at the instant the reference
ends it, in an anechoic room.

This corpus is the other half. Each recording is an unaccompanied, amateur-ish
performance in a room, recorded on a modest microphone -- the vocadito
situation -- with a random singer, melody, tempo, phrasing and room:

* room reverberation (RT60 0.1-0.8 s), which keeps a note periodic after the
  singer stops, and which pitch trackers report as voiced
* phrase-final releases that decay over 60-350 ms, sometimes falling in pitch
* consonants: unvoiced closures and fricatives that open a voicing gap before
  a syllable, and voiced consonants that dip the level and move the formants
  while staying pitched
* re-articulated vowels, diphthongs (a vowel change inside a held note: flux
  with no new note), inhales between phrases, breathiness
* level ripple, phrase-level crescendo and diminuendo, accents
* tuning: half the singers sit anywhere within 50 cents of A440, with drift
  across the recording and per-note intonation scatter of 5-30 cents
* scoops, and glides between legato notes

Every number in this package that is attributed to "the realistic corpus" was
measured on ``DEVELOPMENT_SEEDS`` while choosing, and confirmed once on
``HELD_OUT_SEEDS``.

Where a note starts and ends is a convention, and the corpus states its own so
that it can be varied rather than silently assumed:

* **Onset** -- where the note's pitched sound starts: the vowel, or the middle
  of a voiced consonant (whose start and end are both defensible, and both
  within 45 ms of its middle); for a glided legato transition, where the glide
  leaves the previous note.
* **Offset** -- where phonation for the note ends: the next note's onset in
  legato, the start of an unvoiced consonant, or a stated point along a
  release decay. Reverberation is never part of a note. ``close_gaps`` moves
  ends at consonants up to the next onset instead, which is how some
  annotators work -- and, by the evidence of the release-trim regression, how
  vocadito's did.
* **Pitch** -- the pitch sung, as fractional MIDI (the centre of the vibrato),
  as vocadito annotates; or the intended integer note.

Synthetic, and so subject to every caveat in the README: this is evidence that
a change helps on the phenomena it models, not a measurement of real singing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

SR = 16000

DEVELOPMENT_SEEDS = tuple(range(0, 40))
"""The recordings changes are designed and tuned against."""

HELD_OUT_SEEDS = tuple(range(1000, 1040))
"""The recordings a change is confirmed on, once, after it has been chosen."""

# (F1, F2, F3, F4) for an adult male voice; scaled up for higher voices.
VOWELS = {
    "a": (730, 1090, 2440, 3400),
    "e": (530, 1840, 2480, 3500),
    "i": (300, 2200, 2950, 3700),
    "o": (500, 850, 2400, 3300),
    "u": (330, 900, 2300, 3300),
}
NASAL = (280, 1100, 2400, 3300)
BANDWIDTHS = np.array([80.0, 100.0, 140.0, 200.0])
FORMANT_GAINS_DB = np.array([0.0, -4.0, -12.0, -18.0])

Reference = List[Tuple[float, float, float]]


@dataclass
class CorpusNote:
    """One sung note and everything needed to annotate it under any convention."""

    onset: float
    end: float              # phonation end, or where a release decay starts
    release: float          # length of the release decay; 0 for other endings
    sung_pitch: float       # fractional MIDI: the centre of what was sung
    intended_pitch: int     # the score's note
    end_kind: str           # 'legato', 'unvoiced' or 'release'
    onset_kind: str = ""    # 'vowel', 'voiced', 'plosive', 'fricative' or 'melisma'
    glide: bool = False
    # What the renderer put inside the note, recorded so a transcription's errors
    # can be traced to the phenomenon that caused them. None where absent. All
    # are absolute times in seconds; recording them draws nothing from the RNG.
    glide_start: float | None = None   # where the pitch leaves the previous note
    glide_end: float | None = None     # where it arrives at this note's pitch
    scoop_end: float | None = None     # where an onset scoop has (mostly) recovered
    diphthong_at: float | None = None  # a vowel change inside the held note

    def offset(self, release_point: float) -> float:
        """The end, with a release annotated ``release_point`` of the way down."""
        return self.end + release_point * self.release


@dataclass
class CorpusRecording:
    name: str
    audio: np.ndarray       # mono, at SR
    notes: List[CorpusNote]
    meta: Dict = field(default_factory=dict)

    def reference(self, release_point: float = 0.5, *, close_gaps: bool = False,
                  intended_pitch: bool = False, glide_onset: str = "departure") -> Reference:
        """``(onset, offset, pitch)`` triples under the chosen convention.

        ``release_point`` places a release's annotated end along its decay: 0 at
        the top, where it starts to fall; 0.5 half way, 20 dB down; 1 at the
        bottom, 40 dB down. ``close_gaps`` ends a note that stops on an
        unvoiced consonant at the next note's onset instead, when that is less
        than 200 ms later. ``intended_pitch`` gives the score's integer note
        rather than the sung pitch.

        ``glide_onset`` says where a glided legato note starts: 'departure',
        where the pitch leaves the previous note (the default, and the only
        reading before this option existed); 'midpoint', half way through the
        glide; or 'arrival', where the pitch reaches the new note. Annotators
        differ on this, so a transcriber should be measured under all three
        rather than fitted to one. The note before ends where this one starts,
        as legato notes share their boundary. An onset is never moved earlier
        than the departure reading.
        """
        if glide_onset not in ("departure", "midpoint", "arrival"):
            raise ValueError(f"glide_onset must be departure, midpoint or arrival, not {glide_onset!r}")
        onsets = []
        for note in self.notes:
            onset = note.onset
            if glide_onset != "departure" and note.glide_start is not None:
                frac = 0.5 if glide_onset == "midpoint" else 1.0
                onset = max(onset, note.glide_start + frac * (note.glide_end - note.glide_start))
            onsets.append(onset)
        out = []
        for i, note in enumerate(self.notes):
            end = note.offset(release_point)
            if note.end_kind == "legato" and i + 1 < len(self.notes) and end == self.notes[i + 1].onset:
                end = onsets[i + 1]
            if close_gaps and note.end_kind == "unvoiced" and i + 1 < len(self.notes):
                following = self.notes[i + 1]
                if following.onset - end < 0.2:
                    end = following.onset
            pitch = float(note.intended_pitch) if intended_pitch else note.sung_pitch
            out.append((onsets[i], end, pitch))
        return out


def _pink(n, rng):
    white = rng.normal(0, 1, n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, white)


def _rir(rt60, drr_db, rng):
    """A room: a direct path and an exponentially decaying, low-passed tail."""
    n = int(max(rt60, 0.05) * 1.2 * SR)
    t = np.arange(n) / SR
    tail = rng.normal(0, 1, n) * np.exp(-6.9 * t / rt60)
    tail[: int(0.004 * SR)] = 0.0      # pre-delay before the diffuse tail
    tail = sosfilt(butter(2, 5000, "low", fs=SR, output="sos"), tail)
    tail *= np.sqrt(1.0 / np.sum(tail ** 2) * 10 ** (-drr_db / 10))
    tail[0] += 1.0
    return tail


def make_recording(seed: int, *, with_reverb: bool = True) -> CorpusRecording:
    """Render the corpus recording for ``seed``. The same seed always gives the
    same samples and the same reference."""
    rng = np.random.default_rng(seed)

    # ---------------- singer ----------------
    voice = rng.choice(["bass", "tenor", "alto", "soprano"])
    centre = {"bass": 50, "tenor": 57, "alto": 64, "soprano": 70}[voice] + rng.integers(-3, 4)
    fscale = {"bass": 1.0, "tenor": 1.05, "alto": 1.15, "soprano": 1.2}[voice]
    vib_rate = rng.uniform(4.5, 6.5)
    vib_depth = rng.choice([rng.uniform(0, 15), rng.uniform(15, 50), rng.uniform(50, 90)], p=[0.35, 0.45, 0.2])
    vib_delay = rng.uniform(0.1, 0.4)
    scatter = rng.uniform(5, 30)          # cents, per note
    if rng.random() < 0.5:
        tuning = rng.normal(0, 10)
    else:
        tuning = rng.uniform(-50, 50)
    drift = rng.normal(0, 20)             # cents across the recording
    p_glide = rng.uniform(0.0, 0.6)
    p_scoop = rng.uniform(0.0, 0.5)
    p_fall = rng.uniform(0.0, 0.4)
    breath_db = rng.uniform(-35, -15)
    tilt = rng.uniform(9, 15)             # dB/octave
    shimmer_db = rng.uniform(0.5, 3.0)
    timing_jitter = rng.uniform(0.0, 0.03)

    # ---------------- score ----------------
    bpm = rng.uniform(70, 150)
    beat = 60.0 / bpm
    major = [0, 2, 4, 5, 7, 9, 11]
    minor = [0, 2, 3, 5, 7, 8, 10]
    scale = major if rng.random() < 0.6 else minor
    tonic = int(centre - 5 + rng.integers(0, 6))
    def degree_to_midi(d):
        octave, step = divmod(d, 7)
        return tonic + 12 * octave + scale[step]

    n_phrases = int(rng.integers(2, 4))
    t = rng.uniform(0.3, 0.8)
    notes = []   # dicts
    degree = int(rng.integers(0, 5))
    for ph in range(n_phrases):
        n_notes = int(rng.integers(4, 11))
        for k in range(n_notes):
            dur_beats = rng.choice([0.5, 1, 1.5, 2, 3], p=[0.35, 0.35, 0.1, 0.13, 0.07])
            if k == n_notes - 1:
                dur_beats = max(dur_beats, rng.choice([1, 2, 3]))
            dur = dur_beats * beat
            step = rng.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4], p=[0.03, 0.07, 0.17, 0.25, 0.12, 0.2, 0.1, 0.04, 0.02])
            if k > 0 or ph > 0:
                degree += int(step)
            midi = degree_to_midi(degree)
            if abs(midi - centre) > 8:
                degree -= int(np.sign(midi - centre)) * 2
                midi = degree_to_midi(degree)
            notes.append(dict(t=t + rng.normal(0, timing_jitter) if k else t, dur=dur, midi=midi,
                              phrase=ph))
            t += dur
        t += rng.uniform(0.3, 1.2)
    total = t + 0.8

    # ---------------- syllables and boundaries ----------------
    # Decide, per note, the onset consonant and per-transition details.
    n = int(total * SR)
    tt = np.arange(n) / SR
    pitch = np.full(n, np.nan)          # sung pitch, MIDI
    amp_db = np.full(n, -120.0)         # voiced amplitude envelope
    vowel_id = np.zeros(n, dtype=int)   # index into vowel list (-1 nasal)
    noise_sig = np.zeros(n)             # unvoiced consonant/breath noise (pre-room)
    vowel_names = list(VOWELS)

    refs: List[CorpusNote] = []
    for i, nt in enumerate(notes):
        prev = notes[i - 1] if i > 0 else None
        same_phrase_prev = prev is not None and prev["phrase"] == nt["phrase"]
        # syllable onset type
        if same_phrase_prev and prev["midi"] != nt["midi"] and rng.random() < 0.18:
            kind = "melisma"
        else:
            kind = rng.choice(["vowel", "voiced", "plosive", "fricative"], p=[0.3, 0.3, 0.25, 0.15])
        nt["kind"] = kind
        nt["glide"] = (same_phrase_prev and kind in ("melisma", "vowel", "voiced")
                       and prev["midi"] != nt["midi"] and rng.random() < p_glide)
        nt["cons_len"] = {"vowel": 0.0, "melisma": 0.0, "voiced": rng.uniform(0.04, 0.09),
                          "plosive": rng.uniform(0.04, 0.10), "fricative": rng.uniform(0.07, 0.16)}[kind]

    # Target pitches with tuning, drift, scatter
    for nt in notes:
        frac_pos = nt["t"] / total
        nt["target"] = nt["midi"] + (tuning + drift * frac_pos + rng.normal(0, scatter)) / 100.0

    vowel_choice = [int(rng.integers(0, 5)) for _ in notes]
    for nt in notes:
        # Where pitched sound of this note starts. Unvoiced consonants occupy
        # time *before* the beat; voiced ones are pitched and start the note.
        if nt["kind"] in ("plosive", "fricative"):
            onset = nt["t"]                  # vowel on the beat
        elif nt["kind"] == "voiced":
            onset = nt["t"] - nt["cons_len"]  # pitched consonant leads into the beat
        else:
            onset = nt["t"]
        nt["onset"] = onset
        # A voiced consonant is pitched, so where its note "starts" is a matter
        # of convention: its beginning or the vowel. Annotate its midpoint, so
        # both readings are within 45 ms of the reference.
        nt["ann_onset"] = onset + 0.5 * nt["cons_len"] if nt["kind"] == "voiced" else onset

    # An unvoiced consonant takes its time from the note before it; never let
    # it take more than 40% of that note, or a short note at a fast tempo ends
    # before it starts.
    for i, nt in enumerate(notes[:-1]):
        nxt = notes[i + 1]
        if nxt["phrase"] == nt["phrase"] and nxt["kind"] in ("plosive", "fricative"):
            nxt["cons_len"] = min(nxt["cons_len"], 0.4 * (nxt["onset"] - nt["onset"]))

    for i, nt in enumerate(notes):
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        connected = nxt is not None and nxt["phrase"] == nt["phrase"]
        if connected:
            if nxt["kind"] in ("plosive", "fricative"):
                end = nxt["onset"] - nxt["cons_len"]
                end_kind = "unvoiced"
            else:
                end = nxt["ann_onset"]
                end_kind = "legato"
            release = 0.0
        else:
            # phrase end: release decay; sometimes an unvoiced coda
            nominal_end = nt["t"] + nt["dur"] * rng.uniform(0.75, 1.0)
            if rng.random() < 0.3:
                end = nominal_end
                end_kind = "unvoiced"
                release = 0.0
            else:
                release = rng.uniform(0.06, 0.35)
                end = nominal_end - release * 0.5
                end_kind = "release"
        nt["end"] = end
        nt["release"] = release
        nt["end_kind"] = end_kind
        refs.append(CorpusNote(onset=nt["ann_onset"], end=end, release=release,
                               sung_pitch=float(nt["target"]), intended_pitch=int(nt["midi"]),
                               end_kind=end_kind, onset_kind=nt["kind"], glide=bool(nt["glide"])))

    # ---------------- render control curves ----------------
    dips = []   # (centre s, half-width s, depth dB): vowel re-articulations
    for i, nt in enumerate(notes):
        a = int(nt["onset"] * SR)
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        # Rendered phonation end: up to the next note's (rendered) start in
        # legato, which for a voiced consonant is before its annotated onset.
        if nt["end_kind"] == "legato" and nxt is not None:
            e_ph = nxt["onset"]
        else:
            e_ph = nt["end"] + nt["release"]
        b = int(e_ph * SR)
        if b <= a:
            continue
        seg_t = tt[a:b] - nt["onset"]
        target = nt["target"]
        p = np.full(b - a, target)
        prev = notes[i - 1] if i > 0 else None
        if nt["glide"] and prev is not None:
            g = rng.uniform(0.04, 0.16)
            gn = min(int(g * SR), b - a)
            r = np.linspace(0, 1, gn)
            r = r * r * (3 - 2 * r)
            p[:gn] = prev["target"] + (target - prev["target"]) * r
            refs[i].glide_start = nt["onset"]
            refs[i].glide_end = nt["onset"] + gn / SR
        elif rng.random() < p_scoop and nt["kind"] != "melisma":
            sc = rng.uniform(0.5, 2.0)
            sl = rng.uniform(0.05, 0.12)
            p -= sc * np.exp(-seg_t / (sl / 3))
            refs[i].scoop_end = nt["onset"] + sl
        # vibrato with delayed fade-in
        fade = np.clip((seg_t - vib_delay * 0.5) / vib_delay, 0, 1)
        p += (vib_depth / 100.0) * fade * np.sin(2 * np.pi * vib_rate * seg_t + rng.uniform(0, 6.28))
        # falls at release
        if nt["end_kind"] == "release" and rng.random() < p_fall:
            fall = rng.uniform(1.0, 4.0)
            rs = int((nt["end"] - nt["onset"]) * SR)
            if 0 < rs < b - a:
                ramp = np.linspace(0, 1, b - a - rs) ** 2
                p[rs:] -= fall * ramp
        pitch[a:b] = p

        # amplitude (dB)
        accent = rng.normal(0, 2.5)
        level = np.full(b - a, accent)
        # attack
        if nt["kind"] == "vowel" and (prev is None or prev["phrase"] != nt["phrase"]):
            atk = rng.uniform(0.02, 0.08)
        elif nt["kind"] in ("plosive", "fricative"):
            atk = rng.uniform(0.01, 0.03)
        else:
            atk = 0.0
        if atk > 0:
            an = min(int(atk * SR), b - a)
            level[:an] += 40 * (np.linspace(0, 1, an) - 1)
        # voiced consonant (m, n, l, w): a level dip, nasal formants
        if nt["kind"] == "voiced":
            cn = min(int(nt["cons_len"] * SR), b - a)
            dip = rng.uniform(8, 18)
            shape = np.ones(cn)
            ramp = min(int(0.010 * SR), cn // 2)
            if ramp > 0:
                shape[:ramp] = np.linspace(0, 1, ramp)
                shape[-ramp:] = np.linspace(1, 0, ramp)
            level[:cn] -= dip * shape
            vowel_id[a:a + cn] = -1
            vowel_id[a + cn:b] = vowel_choice[i]
        elif nt["kind"] == "vowel" and prev is not None and prev["phrase"] == nt["phrase"]:
            # a new vowel with no consonant: a glottal dip centred on the boundary
            dips.append((nt["onset"], rng.uniform(0.015, 0.04), rng.uniform(4, 15)))
            vowel_id[a:b] = vowel_choice[i]
        elif nt["kind"] == "melisma":
            vowel_id[a:b] = vowel_choice[i - 1] if i > 0 else vowel_choice[i]
            vowel_choice[i] = vowel_id[a]
        else:
            vowel_id[a:b] = vowel_choice[i]
        # diphthongs: a vowel change inside a held note, which is flux without a note
        if nt["kind"] != "melisma" and (b - a) > 0.35 * SR and rng.random() < 0.25:
            m = a + int((b - a) * rng.uniform(0.4, 0.7))
            vowel_id[m:b] = (vowel_choice[i] + int(rng.integers(1, 5))) % 5
            refs[i].diphthong_at = m / SR
        # release decay, 40 dB linear-in-dB over release length
        if nt["release"] > 0:
            rs = int((nt["end"] - nt["onset"]) * SR)
            rn = b - a - rs
            if rn > 0 and rs >= 0:
                level[rs:] -= np.linspace(0, 40, rn)
        elif nt["end_kind"] == "unvoiced":
            en = min(int(0.012 * SR), b - a)
            level[-en:] -= np.linspace(0, 30, en)
            # coda / next-syllable unvoiced consonant noise
            nxt = notes[i + 1] if i + 1 < len(notes) else None
            clen = nxt["cons_len"] if (nxt is not None and nxt["phrase"] == nt["phrase"]) else rng.uniform(0.06, 0.14)
            ckind = nxt["kind"] if (nxt is not None and nxt["phrase"] == nt["phrase"]) else "fricative"
            cs = b
            cn = min(int(clen * SR), n - cs)
            if cn > 0:
                burst = rng.normal(0, 1, cn)
                if ckind == "plosive":
                    env = np.zeros(cn)
                    bl = min(int(0.015 * SR), cn)
                    env[cn - bl:] = np.exp(-np.linspace(0, 4, bl))
                    lvl = rng.uniform(-14, -6)
                else:
                    env = np.hanning(cn)
                    lvl = rng.uniform(-24, -12)
                sos = butter(2, [2500, 7000], "band", fs=SR, output="sos")
                burst = sosfilt(sos, burst)
                burst = burst / (np.std(burst) + 1e-9) * env
                noise_sig[cs:cs + cn] += burst * 10 ** (lvl / 20) * 0.3
        amp_db[a:b] = np.maximum(amp_db[a:b], level)

    for centre, half, depth in dips:
        c = int(centre * SR)
        h = int(half * SR)
        lo, hi = max(c - h, 0), min(c + h, n)
        x = (np.arange(lo, hi) - c) / max(h, 1)
        amp_db[lo:hi] -= depth * 0.5 * (1 + np.cos(np.pi * x))

    # phrase arcs and shimmer
    arc = np.zeros(n)
    for ph in set(nt["phrase"] for nt in notes):
        idx = [nt for nt in notes if nt["phrase"] == ph]
        a = int(idx[0]["onset"] * SR)
        b = int((idx[-1]["end"] + idx[-1]["release"]) * SR)
        if b > a:
            shape = rng.choice(["cresc", "dim", "arch"])
            x = np.linspace(0, 1, b - a)
            depth = rng.uniform(2, 8)
            arc[a:b] = {"cresc": depth * (x - 1), "dim": -depth * x, "arch": -depth * (2 * x - 1) ** 2}[shape]
    sh = _pink(n, rng)
    sh = sh / (np.std(sh) + 1e-9)
    sos = butter(2, 8, "low", fs=SR, output="sos")
    sh = sosfilt(sos, sh)
    sh = sh / (np.std(sh) + 1e-9) * shimmer_db
    amp_db = amp_db + arc + sh

    # audible inhales before every phrase but the first
    for ph in sorted(set(nt["phrase"] for nt in notes))[1:]:
        first = [nt for nt in notes if nt["phrase"] == ph][0]
        end_t = first["onset"] - (first["cons_len"] if first["kind"] in ("plosive", "fricative") else 0) \
            - rng.uniform(0.05, 0.15)
        ln = rng.uniform(0.2, 0.4)
        s0, s1 = int(max(end_t - ln, 0) * SR), int(max(end_t, 0) * SR)
        if s1 > s0:
            br = sosfilt(butter(2, [500, 4000], "band", fs=SR, output="sos"), rng.normal(0, 1, s1 - s0))
            br = br / (np.std(br) + 1e-9) * np.hanning(s1 - s0)
            noise_sig[s0:s1] += br * 10 ** (rng.uniform(-35, -20) / 20) * 0.3

    # ---------------- additive harmonic synthesis ----------------
    voiced = np.isfinite(pitch)
    f0 = np.where(voiced, 440.0 * 2 ** ((np.nan_to_num(pitch, nan=60) - 69) / 12), 0.0)
    # jitter: small fast random perturbation
    jit = sosfilt(butter(2, 30, "low", fs=SR, output="sos"), rng.normal(0, 1, n))
    jit = jit / (np.std(jit) + 1e-9) * 0.003
    f0 = f0 * (1 + jit)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    amp = np.where(voiced, 10 ** (amp_db / 20), 0.0)

    # envelope evaluated every 5 ms, interpolated per sample
    hop = 80
    frames = np.arange(0, n, hop)
    out = np.zeros(n)
    vowel_frames = vowel_id[frames]
    # smooth formant transitions by averaging formant tables over ~40ms
    form_tab = np.array([VOWELS[v] for v in vowel_names] + [NASAL], dtype=float)
    ftrack = form_tab[np.where(vowel_frames < 0, 5, vowel_frames)]
    k = 8
    kernel = np.ones(k) / k
    ftrack = np.stack([np.convolve(np.pad(ftrack[:, j], (k // 2, k - 1 - k // 2), mode="edge"), kernel, mode="valid")
                       for j in range(4)], axis=1)
    f0_frames = np.maximum(f0[frames], 1.0)
    max_h = int(7600 / max(f0[voiced].min() if voiced.any() else 100, 50))
    # Spectral envelope per harmonic per frame, normalised so the harmonics'
    # total power is constant: loudness is set by amp_db alone and timbre by
    # the formants. Without this, a formant landing on f0 (a nasal's low F1
    # under a high voice) adds 15-20 dB and swallows the dip it was meant to make.
    env_db = np.full((max_h, frames.shape[0]), -200.0)
    for h in range(1, max_h + 1):
        fh = h * f0_frames
        e = np.full(frames.shape[0], -200.0)
        for j in range(4):
            fc = ftrack[:, j] * fscale
            bw = BANDWIDTHS[j] * fscale
            x = (fh - fc) / (bw / 2)
            comp = FORMANT_GAINS_DB[j] - 10 * np.log10(1 + x * x)
            e = 10 * np.log10(10 ** (e / 10) + 10 ** (comp / 10))
        e += -tilt * np.log2(h)
        env_db[h - 1] = np.where(fh < 7600, e, -200.0)
    power = np.sum(10 ** (env_db / 10), axis=0)
    env_db -= 10 * np.log10(np.maximum(power, 1e-20))[None, :]
    for h in range(1, max_h + 1):
        if env_db[h - 1].max() < -150:
            continue
        env_s = np.interp(np.arange(n), frames, 10 ** (env_db[h - 1] / 20))
        out += env_s * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
    out *= amp

    # aspiration noise, following the voiced amplitude
    asp = sosfilt(butter(2, [1500, 6000], "band", fs=SR, output="sos"), rng.normal(0, 1, n))
    asp = asp / (np.std(asp) + 1e-9)
    voiced_rms = np.sqrt(np.mean(out[voiced] ** 2)) if voiced.any() else 1.0
    out += asp * amp * voiced_rms / (np.sqrt(np.mean(amp[voiced] ** 2)) + 1e-12) * 10 ** (breath_db / 20)
    out += noise_sig * voiced_rms * 3

    # ---------------- room and microphone ----------------
    rt60 = rng.uniform(0.1, 0.8) if with_reverb else 0.0
    drr = rng.uniform(0, 15)
    if with_reverb and rt60 > 0:
        out = fftconvolve(out, _rir(rt60, drr, rng))[:n]
    peak = np.max(np.abs(out)) + 1e-12
    out = out / peak * 0.7
    noise_db = rng.uniform(-65, -42)
    floor = _pink(n, rng)
    floor = floor / (np.std(floor) + 1e-12) * 10 ** (noise_db / 20)
    out = out + floor
    hp = rng.uniform(70, 150)
    out = sosfilt(butter(2, hp, "high", fs=SR, output="sos"), out)

    meta = dict(voice=voice, vib_depth=float(vib_depth), tuning=float(tuning), drift=float(drift),
                scatter=float(scatter), rt60=float(rt60), drr=float(drr), noise_db=float(noise_db),
                p_glide=float(p_glide), bpm=float(bpm))
    return CorpusRecording(name=f"rs{seed:04d}", audio=out.astype(np.float32), notes=refs,
                           meta=meta)
