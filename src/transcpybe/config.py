"""Configuration for vocal melody transcription.

Every tunable lives here so that behaviour can be adjusted without touching
algorithm code. Defaults are tuned for source-separated, monophonic human
voice at typical pop/rock tessitura.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class TranscriptionConfig:
    """Parameters for the audio -> MIDI note transcription pipeline."""

    # ---------------- Signal / framing ----------------
    sample_rate: int = 16000
    """Analysis rate. CREPE is trained at 16 kHz; audio is resampled to this."""

    hop_ms: float = 10.0
    """Frame hop in milliseconds. 10 ms matches CREPE's native resolution."""

    # ---------------- Pitch range ----------------
    fmin_hz: float = 55.0
    """Lowest f0 considered (A1). Below this is almost never sung pitch."""

    fmax_hz: float = 1100.0
    """Highest f0 considered (~C#6), comfortably above soprano belt range."""

    # ---------------- f0 estimation ----------------
    f0_backend: str = "crepe"
    """'crepe' (torchcrepe, neural, most accurate) or 'pyin' (librosa, no torch)."""

    crepe_model: str = "full"
    """torchcrepe capacity: 'full' (best) or 'tiny' (fast)."""

    crepe_batch_size: int = 512
    """Frames per CREPE forward pass. Lower this if memory is tight."""

    device: str = "auto"
    """Torch device for CREPE. 'auto' prefers CUDA, then Apple MPS, then CPU.
    An explicit 'cpu', 'cuda', 'cuda:1' or 'mps' is honoured as given."""

    seed: Optional[int] = 0
    """Seed for CREPE's pitch dither, making transcription reproducible.

    torchcrepe deliberately adds random triangular dither of +/-20 cents to
    every f0 estimate, to trade bin-quantization error for noise. It is drawn
    from numpy's global RNG, so by default the same audio gives a slightly
    different transcription on every run. Seeding makes a run repeatable, which
    matters when transcriptions feed a dataset. Set to None to restore
    torchcrepe's stochastic behaviour. The RNG state is saved and restored, so
    seeding here never disturbs the caller's own random stream."""

    gpu_memory_log_every: int = 0
    """Log GPU memory every N CREPE batches (0 disables). Useful for watching
    headroom on a shared GPU before an out-of-memory failure, not after."""

    f0_median_filter_frames: int = 3
    """Median filter width on the f0 contour; suppresses isolated octave errors.
    Must be odd. Set to 1 to disable."""

    # ---------------- Voicing ----------------
    voicing_threshold: float = 0.30
    """Minimum combined voicing probability for a frame to inform note pitch."""

    loudness_dynamic_range_db: float = 45.0
    """Adaptive silence gate. Frames quieter than (loud percentile - this) are
    treated as unvoiced, which rejects the low-level burble that source
    separation leaves in silent passages."""

    loudness_percentile: float = 95.0
    """Percentile of frame energy taken as the 'loud' reference level."""

    loudness_soft_width_db: float = 10.0
    """Width of the soft ramp above the gate, so the gate is not a hard cliff."""

    # ---------------- Note HMM (Viterbi) ----------------
    pitch_sigma_semitones: float = 0.55
    """Std-dev of the Gaussian emission around each semitone state. Larger
    values absorb vibrato but blur genuine semitone steps."""

    p_self: float = 0.985
    """Probability of a pitch state persisting into the next frame. Drives how
    reluctant the decoder is to start a new note."""

    p_voiced_to_unvoiced: float = 0.005
    """Per-frame probability of a note ending into silence."""

    p_unvoiced_self: float = 0.97
    """Probability that silence persists into the next frame."""

    interval_tau_semitones: float = 7.0
    """Decay constant of the melodic interval prior; mildly favours small leaps
    over large ones, which suppresses spurious octave jumps."""

    unvoiced_offset: float = 0.0
    """Additive log-domain bias on the unvoiced state. Positive values produce
    more silence, negative values produce more notes."""

    # ---------------- Portamento / glides ----------------
    absorb_glides: bool = True
    """Recognise glides between notes: remove passing tones and pull legato
    note starts back to where the singer left the previous pitch."""

    max_passing_note_s: float = 0.12
    """Segments shorter than this, lying between their neighbours' pitches in a
    contiguous run, are treated as a glide crossing rather than a note."""

    glide_max_backtrack_s: float = 0.30
    """Ceiling on how far a legato note start may be moved back."""

    glide_edge_tolerance_semitones: float = 0.35
    """How close the contour must come to a note's pitch to count as arrived."""

    glide_max_gap_s: float = 0.20
    """Longest unvoiced gap a glide may be traced across. Pitch trackers lose
    confidence during fast slides, so the decoder often reports brief silence
    in the middle of one."""

    glide_min_periodicity: float = 0.05
    """A glide must still register as periodic. Pitch trackers keep emitting a
    smoothly drifting pitch estimate through unvoiced noise while reporting
    near-zero confidence, and without this the backtrack follows that drift and
    reports onsets far too early."""

    glide_monotonic_slack_semitones: float = 0.20
    """How far the contour may back-track the wrong way during a glide walk
    before the walk stops. Distinguishes a sung slide, which retreats steadily,
    from noise between notes, which wanders."""

    glide_max_level_drop_db: float = 20.0
    """A glide may not be more than this far below the level of the note it
    leads into. Keeps onset backtracking out of quiet residue between phrases."""

    glide_min_audibility: float = 0.05
    """Minimum loudness-gate value for a frame to count as part of a glide.
    Guards the backtrack against walking through genuine silence."""

    # ---------------- Edge fragments ----------------
    absorb_fragments: bool = True
    """Fold very short segments back into the much longer note beside them.
    Wide vibrato sheds such fragments at note ends, a semitone or so below."""

    fragment_max_duration_s: float = 0.12
    """Longest segment that can be treated as an edge fragment."""

    fragment_max_interval_semitones: int = 2
    """A fragment must lie within this interval of its neighbour to be absorbed."""

    fragment_dominance_ratio: float = 3.0
    """The neighbour must be at least this many times longer than the fragment,
    so that two genuinely short notes side by side are never merged."""

    # ---------------- Onset detection ----------------
    use_onsets: bool = True
    """Use spectral-flux onsets to split repeated same-pitch notes and to align
    note starts to the audible attack."""

    onset_n_fft: int = 512
    """FFT window for the onset envelope, in samples at ``sample_rate``.

    512 samples is 32 ms at the default 16 kHz. librosa's own default is 2048,
    which this deliberately overrides: 2048 samples is 46 ms at the 44.1 kHz
    SuperFlux was formulated for, but 128 ms here, and an attack smeared over
    128 ms cannot be localised to the 50 ms the scoring convention allows.
    Leaving it unset cost roughly two thirds of the detector's onset F."""

    onset_aggregate: str = "mean"
    """How the spectral flux is combined across frequency: 'mean' or 'median'.

    The median asks for rising energy in more than half of the mel bands at
    once. A sung syllable raises energy in a few formant regions, so broadband
    consonants clear that bar and vowel-initial onsets -- most of real singing
    -- do not. Vibrato suppression is ``onset_max_size``'s job, not this one."""

    onset_max_size: int = 9
    """Frequency-domain max-filter size for SuperFlux vibrato suppression.

    Three mel bands is far less than an operatic vibrato's excursion, so a
    steadily wobbling note still registers as flux. With ``onset_delta_ratio``
    scaling the peak threshold by the envelope's median, that costs recall
    rather than precision: vibrato flux raises the median, the bar rises with
    it, and genuine onsets end up underneath. At a width of 1 the
    re-articulation cases collapse from eight notes to one. 5 to 9 is a
    plateau; 9 leaves margin for wider vibrato than the suite contains."""

    onset_lag: int = 2
    """Frame lag for the SuperFlux difference."""

    onset_delta_ratio: float = 1.0
    """Peak-picking threshold, as a multiple of the envelope's own median.

    librosa's ``delta`` is an absolute margin above a local mean, and an
    absolute number cannot serve this envelope: its scale varies by two orders
    of magnitude across recordings. On a breathy take the envelope is close to
    flat noise -- its peak is seven times its median -- and an absolute
    threshold admits most of that noise as onsets; on a take with consonants
    the peak is seven hundred times the median and the same number sits far
    below every real onset. Scaling the threshold by the median asks each
    recording's peaks to stand out from that recording's own noise floor.

    Raise it if noisy input is over-segmented; lower it if quiet
    re-articulations are missed."""

    onset_split_strength_ratio: float = 0.70
    """How strong a detected onset must be, relative to the median detected
    onset in the same recording, to split an existing note.

    Measured against the other onsets rather than against the onset envelope.
    The envelope counts every frame between the peaks, and during continuous
    singing those frames are loud, so its upper percentiles sit above most
    genuine onsets: the gate then asks each peak to clear a bar set by the
    material between the peaks, and re-articulated notes never split at all.

    The bar moved up when the detector stopped being inert. A ratio is taken
    against the median detected onset, so making the detector find every real
    onset also changed what that median means: the same 0.35 now admits peaks
    that are weak against a much richer field, and on breathy input it split
    every note in two. 0.70 is where nothing in the suite regresses.

    That raise and the envelope's new sensitivity landed together and were
    never separated, which left it open whether the one was quietly cancelling
    the other -- whether the detector's extra reach was ever arriving at the
    notes. Separated since, by putting the ratio back to 0.35 with everything
    else as shipped: across all 22 cases only ``breathy`` moves, from 11 notes
    to 13 against a true 7, and every other case is identical to the digit. The
    bar is therefore doing one job, rejecting noise peaks on breathy input, and
    is not holding back anything that reaches a note.

    The same run says why that is worth little. Onset F is already 1.000 on
    every case here, so the suite cannot show a missed onset being found, and
    cannot say whether the sensitivity reaches notes on real audio -- only that
    this ratio is not what would be stopping it.

    Lower splits more. Raise it if the output is over-segmented."""

    onset_split_min_gap_s: float = 0.06
    """Minimum distance from a segment edge for an onset to split that segment.

    Set just above ``min_note_duration_s`` so the two agree: there is no point
    splitting off a piece the duration floor will then discard. At the previous
    0.10 a syllable shorter than 100 ms could not be separated at all, which
    put fast re-articulation out of reach however clear its onset."""

    onset_snap_window_s: float = 0.05
    """Note starts are pulled to a detected onset within this window."""

    # ---------------- Note post-processing ----------------
    min_note_duration_s: float = 0.055
    """Notes shorter than this are discarded as decoder noise."""

    attack_trim_frac: float = 0.20
    """Fraction of a note's head skipped when measuring its pitch, so scoops and
    consonant transients do not drag the estimate."""

    attack_trim_max_s: float = 0.06
    """Absolute cap on the attack trim."""

    reject_quiet_notes: bool = True
    """Drop notes far below the level of the rest of the performance. This is
    what removes phantom notes hallucinated from source-separation residue."""

    note_level_floor_db: float = 30.0
    """A note more than this far below the reference level is discarded. Lower
    it if genuinely quiet singing is being lost; raise it if artefacts survive."""

    note_level_reference_percentile: float = 75.0
    """Duration-weighted percentile of note levels used as the reference."""

    merge_same_pitch_gap_s: float = 0.035
    """Adjacent notes of equal pitch closer than this are merged, unless an
    onset separates them."""

    # ---------------- Note ends ----------------
    trim_note_releases: bool = False
    """End a note where its own level collapses, rather than where the decoder
    stops hearing pitch. Off by default, and the reason is worth the space.

    Everything else in the pipeline refines note *starts*; the end is left
    wherever the Viterbi path happened to leave the pitch state, which the
    adaptive gate holds open until the frame is ``loudness_dynamic_range_db``
    below the whole track. On synthetic signals whose notes keep sounding past
    their annotated end, that reports every note long, and trimming fixes it
    completely -- COnPOff 0.000 to 1.000 on those cases.

    On vocadito it made things worse on all forty recordings: COnPOff 0.410 to
    0.248, a paired difference of -0.161 with a 95% interval of [-0.191,
    -0.134], while COn and COnP did not move. So the defect the trim addresses
    is real and the trim's answer to it is wrong, at least for these
    annotators. It ends a note where the level departs from sustain -- the top
    of the decay -- and vocadito marks note ends later than that.

    Turning it off is therefore not a retraction of the finding, only of the
    constant. Where a note ends is a convention, and a convention has to be
    fitted rather than guessed: enable this and fit ``release_drop_db`` with
    ``transcpybe-tune --objective offset``, which is the machinery that had to
    exist before the question could be asked at all."""

    release_drop_db: float = 1.0
    """How far a frame must sit below the note's own sustained level to count
    as release rather than note.

    Small on purpose. This is not "how quiet is inaudible" -- it is "how far
    from sustain is no longer sustaining", so what it locates is the top of the
    decay, not a point part-way down it. That distinction decides whether the
    trim works at all: a threshold set well down the ramp lands at a fixed
    *fraction* of the release, so its error grows with the release and a long
    one overshoots however the threshold is set. Measured across releases from
    50 ms to 450 ms, every value from 3 dB up failed the longest of them and
    1 dB passed all five.

    Measured against the note, not the track, because a quiet note's release is
    quiet too. The floor under it is vibrato: singers ripple their level by a
    decibel or so, and at 0.5 dB the walk goes back through those dips and ends
    notes early -- it cost two cases in the suite. Raise it if note ends come
    out early, lower it if notes run on."""

    release_min_fraction: float = 0.5
    """A release trim may never take more than this fraction of a note.

    A note that fades throughout -- a deliberate diminuendo -- is below its own
    75th percentile for most of its length, and without this floor the trim
    would report it as a fraction of its true duration."""

    # ---------------- Tuning reference ----------------
    estimate_tuning: bool = True
    """Find the recording's own pitch centre before deciding what a note is.

    The decoder's pitch states and the rounding that produces an integer MIDI
    number both sit on the A440 semitone grid. A performance that sits well off
    that grid -- no reference pitch, an instrument tuned by ear, a tape running
    slow -- is then read against the wrong centres: notes land between two
    states rather than on one, and the ones nearest the boundary round to the
    neighbouring semitone. The transcription comes out a semitone wrong in
    places while the singing was perfectly in tune with itself.

    Measured on synthetic phrases with ordinary intonation scatter, a
    performance 35 cents off the grid loses a quarter of its note F-measure to
    this and nothing else. Correcting for it is not a preference about
    temperament: it is reading the recording in the tuning it was actually sung
    in. Turn it off to force the A440 grid."""

    tuning_min_concentration: float = 0.20
    """How concentrated the pitch deviations must be before the estimate is used.

    The resultant length of the circular mean, in [0, 1]. It answers "is there a
    pitch centre here at all" -- deviations spread evenly around the circle have
    none, and their mean angle is noise that would shift every note by an
    arbitrary amount. Ordinary singing measures 0.4 to 0.8 even with a quarter
    tone of scatter, so this floor rejects only the genuinely unidentifiable."""

    tuning_min_notes: int = 5
    """How many notes a tuning estimate needs before it is worth acting on.

    Below this the estimate rests on too little material and the A440 grid is
    the better guess."""

    tuning_max_offset_semitones: float = 0.40
    """Offsets beyond this are refused rather than applied.

    A quarter tone is where the question stops having an answer. Deviations
    spread symmetrically about the grid -- a singer who is simply scattered,
    landing 40 cents sharp as often as 40 cents flat -- sit, on a circle whose
    period is one semitone, nearest to the point half a semitone away. So the
    scattered performance is not merely mistaken for an offset one: it is
    genuinely better explained by one, because shifting everything by half a
    semitone really does pull those deviations into a tighter band. The suite's
    ``mistuned`` case is exactly this, and its notes are 49 cents from the grid
    by that reckoning while the pitches its singer intended are the ones on it.

    No statistic of the deviations can separate those two readings, because
    they fit the same numbers. What separates them is a prior: tuning
    references cluster near A440 -- the range in practical use spans about
    -40 to +25 cents -- whereas the artefact lands near 50 by construction,
    being the antipode of a distribution centred on zero. Refusing the far half
    of the circle therefore costs the offsets nobody has and rejects the
    artefact structurally rather than by a threshold fitted to one case."""

    tuning_min_offset_semitones: float = 0.05
    """Offsets smaller than this are treated as zero.

    A note is lost to rounding when it ends up more than half a semitone from
    the pitch intended, so a correction of five cents can only ever move a note
    that was already within five cents of the boundary -- one that was a coin
    toss whichever way it fell. Acting on estimates that small is therefore all
    risk and no reward, and it is where a noisy estimate lives: a performance
    already on the grid but scattered reports a few cents in one direction or
    the other, and this declines to chase it. Nearly every recording lands here
    and skips the second decoding pass entirely."""

    # ---------------- MIDI output ----------------
    midi_program: int = 0
    """General MIDI program for the output track (0 = Acoustic Grand Piano)."""

    midi_tempo: float = 120.0
    """Tempo written into the MIDI header. Note times are absolute seconds and
    are unaffected by this value; it only sets how a DAW displays the grid."""

    velocity_from_loudness: bool = True
    """Derive MIDI velocity from note loudness rather than using a constant."""

    velocity_min: int = 45
    velocity_max: int = 110
    velocity_constant: int = 80
    """Velocity used when velocity_from_loudness is False."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def validate(self) -> None:
        if self.f0_backend not in ("crepe", "pyin"):
            raise ValueError(f"f0_backend must be 'crepe' or 'pyin', got {self.f0_backend!r}")
        if self.crepe_model not in ("full", "tiny"):
            raise ValueError(f"crepe_model must be 'full' or 'tiny', got {self.crepe_model!r}")
        if self.fmin_hz <= 0 or self.fmax_hz <= self.fmin_hz:
            raise ValueError("Require 0 < fmin_hz < fmax_hz")
        if self.f0_median_filter_frames % 2 == 0:
            raise ValueError("f0_median_filter_frames must be odd")
        if self.onset_aggregate not in ("mean", "median"):
            raise ValueError("onset_aggregate must be 'mean' or 'median', "
                             f"got {self.onset_aggregate!r}")
        if self.onset_n_fft <= 0:
            raise ValueError("onset_n_fft must be positive")
        if not 0.0 <= self.tuning_min_concentration <= 1.0:
            raise ValueError("tuning_min_concentration must be in [0, 1]")
        if self.tuning_min_notes < 1:
            raise ValueError("tuning_min_notes must be at least 1")
        if self.tuning_min_offset_semitones < 0:
            raise ValueError("tuning_min_offset_semitones must be non-negative")
        if not 0.0 <= self.tuning_max_offset_semitones <= 0.5:
            raise ValueError("tuning_max_offset_semitones must be in [0, 0.5]")
        if not 0.0 < self.p_self < 1.0:
            raise ValueError("p_self must be in (0, 1)")
        if self.p_self + self.p_voiced_to_unvoiced >= 1.0:
            raise ValueError("p_self + p_voiced_to_unvoiced must be < 1")
        if not 0.0 < self.p_unvoiced_self < 1.0:
            raise ValueError("p_unvoiced_self must be in (0, 1)")
        if self.pitch_sigma_semitones <= 0:
            raise ValueError("pitch_sigma_semitones must be positive")

    @property
    def hop_length(self) -> int:
        """Frame hop in samples at the analysis rate."""
        return int(round(self.sample_rate * self.hop_ms / 1000.0))
