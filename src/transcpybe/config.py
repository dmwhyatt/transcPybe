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

    onset_max_size: int = 3
    """Frequency-domain max-filter size for SuperFlux vibrato suppression."""

    onset_lag: int = 2
    """Frame lag for the SuperFlux difference."""

    onset_split_strength_ratio: float = 0.35
    """How strong a detected onset must be, relative to the median detected
    onset in the same recording, to split an existing note.

    Measured against the other onsets rather than against the onset envelope.
    The envelope counts every frame between the peaks, and during continuous
    singing those frames are loud, so its upper percentiles sit above most
    genuine onsets: the gate then asks each peak to clear a bar set by the
    material between the peaks, and re-articulated notes never split at all.

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
