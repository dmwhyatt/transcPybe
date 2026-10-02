"""Frame-wise f0 estimation and voicing detection.

Two interchangeable backends:

* ``crepe`` - CREPE (Kim, Salamon, Li & Bello, ICASSP 2018) via ``torchcrepe``.
  A deep convolutional pitch tracker; the most accurate option for voice, and
  the default. Model weights ship inside the ``torchcrepe`` wheel.
* ``pyin`` - probabilistic YIN (Mauch & Dixon, ICASSP 2014) via ``librosa``.
  Pure signal processing, no torch dependency, a solid fallback.

Both produce the same ``F0Track`` so the rest of the pipeline is agnostic.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from .audio import frame_rms_db
from .config import TranscriptionConfig
from .progress import ProgressReporter, null_reporter


@dataclass
class F0Track:
    """Frame-synchronous pitch estimates.

    Attributes
    ----------
    times : (T,) float
        Frame centre times in seconds.
    f0_hz : (T,) float
        Estimated fundamental in Hz; NaN where no pitch was found.
    periodicity : (T,) float
        Backend confidence in [0, 1] that the frame is periodic.
    loudness_db : (T,) float
        Frame RMS in dBFS.
    loudness_gate : (T,) float
        Adaptive silence gate in [0, 1]. Zero means the frame is below the
        track's noise floor and cannot contain sung pitch, whatever the pitch
        tracker reports.
    voicing : (T,) float
        Combined probability in [0, 1] that the frame is sung pitch. This is
        ``periodicity`` attenuated by an adaptive loudness gate.
    """

    times: np.ndarray
    f0_hz: np.ndarray
    periodicity: np.ndarray
    loudness_db: np.ndarray
    loudness_gate: np.ndarray
    voicing: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.times.shape[0])

    @property
    def midi_float(self) -> np.ndarray:
        """f0 as fractional MIDI pitch; NaN where f0 is NaN."""
        return hz_to_midi(self.f0_hz)


def hz_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    """Convert Hz to fractional MIDI pitch, propagating NaN and rejecting <= 0."""
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    out = np.full(f0_hz.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(f0_hz) & (f0_hz > 0)
    out[valid] = 69.0 + 12.0 * np.log2(f0_hz[valid] / 440.0)
    return out


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray | float:
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def _median_filter_f0(f0_hz: np.ndarray, width: int) -> np.ndarray:
    """Median filter in the log-frequency domain, ignoring unvoiced frames.

    Working in cents means the filter treats a halving and a doubling of f0
    symmetrically, so it removes isolated octave errors cleanly.
    """
    if width <= 1:
        return f0_hz

    from scipy.ndimage import median_filter

    valid = np.isfinite(f0_hz) & (f0_hz > 0)
    if not valid.any():
        return f0_hz

    log_f0 = np.full(f0_hz.shape, np.nan)
    log_f0[valid] = np.log2(f0_hz[valid])

    # Interpolate across gaps so the filter is not dragged toward NaN, then
    # restore the original unvoiced mask afterwards.
    idx = np.arange(f0_hz.shape[0])
    filled = np.interp(idx, idx[valid], log_f0[valid])
    smoothed = median_filter(filled, size=width, mode="nearest")

    out = np.full(f0_hz.shape, np.nan)
    out[valid] = 2.0 ** smoothed[valid]
    return out


def _loudness_gate(loudness_db: np.ndarray, config: TranscriptionConfig) -> np.ndarray:
    """Soft gate in [0, 1] that suppresses frames well below the track's level.

    Source separation leaves low-level bleed and burble in passages where the
    singer is silent. CREPE reports confident pitch on that residue, so an
    energy gate relative to the track's own loud level is what keeps phantom
    notes out of the transcription. The reference is a high percentile rather
    than the maximum, so one clipped sample cannot move the gate.
    """
    finite = loudness_db[np.isfinite(loudness_db)]
    if finite.size == 0:
        return np.zeros_like(loudness_db)

    reference_db = float(np.percentile(finite, config.loudness_percentile))
    gate_db = reference_db - config.loudness_dynamic_range_db
    width = max(config.loudness_soft_width_db, 1e-6)

    # Linear ramp from 0 at the gate to 1 at (gate + width).
    gate = (loudness_db - gate_db) / width
    return np.clip(gate, 0.0, 1.0)


@contextmanager
def _seeded_numpy(seed: Optional[int]) -> Iterator[None]:
    """Temporarily seed numpy's global RNG, then restore it.

    torchcrepe's pitch dither draws from the global numpy RNG via scipy, so
    this is the only way to make CREPE reproducible. The previous state is put
    back on exit, so importing this module never changes the randomness of
    whatever else the caller is doing.
    """
    if seed is None:
        yield
        return

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


_CREPE_LOWEST_HZ = 32.0
"""Just above CREPE's lowest pitch bin, 31.7 Hz (1997.38 cents above 10 Hz)."""


def _crepe_batch_count(n_samples: int, config: TranscriptionConfig) -> int:
    """How many batches torchcrepe will run, mirroring its own framing.

    Needed up front so progress can be reported as a percentage rather than a
    running count with no end in sight.
    """
    import torchcrepe

    hop = config.hop_length
    length = n_samples
    if config.sample_rate != torchcrepe.SAMPLE_RATE:
        # preprocess() resamples first, then scales the hop to match.
        ratio = torchcrepe.SAMPLE_RATE / config.sample_rate
        length = int(round(length * ratio))
        hop = int(hop * ratio)
    hop = max(hop, 1)

    total_frames = 1 + int(length // hop)          # pad=True
    batch_size = config.crepe_batch_size or total_frames
    return max(1, -(-total_frames // batch_size))  # ceiling division


def _estimate_crepe(
    audio: np.ndarray,
    config: TranscriptionConfig,
    progress: Optional[ProgressReporter] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run CREPE, returning (f0_hz, periodicity) with NaN where unvoiced.

    This drives torchcrepe's ``preprocess``/``infer``/``postprocess`` directly
    rather than calling ``predict``, purely so that progress can be reported
    between batches -- ``predict`` runs the whole file in one opaque call, which
    on a long song means many minutes of silence. The loop is otherwise
    identical to ``predict``'s, and ``test_chunked_crepe_matches_torchcrepe_predict``
    asserts that the outputs agree exactly.
    """
    progress = progress or null_reporter()
    try:
        import torch
        import torchcrepe
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 'crepe' backend needs torch and torchcrepe:\n"
            "    pip install torch torchcrepe\n"
            "Or use the torch-free fallback: --f0-backend pyin"
        ) from exc

    from .device import memory_summary, synchronize

    # torchcrepe is trained at 16 kHz and clamps fmax; respect both. It does
    # not clamp fmin: below its lowest pitch bin (31.7 Hz) the bin index it
    # computes is negative, and probabilities[:, :minidx] then masks all but
    # the top few bins -- every frame comes back unvoiced, and an --fmin of 30
    # for a bass guitar transcribed nothing at all. Frames below the
    # configured fmin are still marked unvoiced below. A constant rather than
    # torchcrepe.convert.bins_to_frequency(0), which dithers its answer below
    # the bin and draws that dither from the caller's numpy RNG.
    fmax = min(config.fmax_hz, float(torchcrepe.MAX_FMAX))
    fmin = max(config.fmin_hz, _CREPE_LOWEST_HZ)

    audio_t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
    n_batches = _crepe_batch_count(audio_t.size(1), config)
    audio_seconds = audio_t.size(1) / config.sample_rate

    pitch_chunks: list = []
    periodicity_chunks: list = []

    with torch.no_grad(), _seeded_numpy(config.seed):
        generator = torchcrepe.preprocess(
            audio_t,
            config.sample_rate,
            hop_length=config.hop_length,
            batch_size=config.crepe_batch_size,
            device=config.device,
            pad=True,
        )

        with progress.stage(f"CREPE {config.crepe_model} on {config.device}",
                            total=n_batches) as stage:
            for index, frames in enumerate(generator):
                probabilities = torchcrepe.infer(
                    frames, config.crepe_model, config.device, embed=False
                )
                probabilities = probabilities.reshape(
                    audio_t.size(0), -1, torchcrepe.PITCH_BINS
                ).transpose(1, 2)

                pitch_chunk, periodicity_chunk = torchcrepe.postprocess(
                    probabilities, fmin, fmax,
                    torchcrepe.decode.viterbi,
                    False,      # return_harmonicity (deprecated)
                    True,       # return_periodicity
                )

                # Back to the audio's device so a very long file does not have
                # to fit on the GPU all at once.
                pitch_chunks.append(pitch_chunk.to(audio_t.device))
                periodicity_chunks.append(periodicity_chunk.to(audio_t.device))

                done_seconds = audio_seconds * (index + 1) / n_batches
                stage.advance(1, detail=f"{done_seconds:.0f}s / {audio_seconds:.0f}s audio")

                if config.gpu_memory_log_every and index and \
                        index % config.gpu_memory_log_every == 0:
                    summary = memory_summary(config.device)
                    if summary:
                        progress.log(f"    {summary}")

            synchronize(config.device)

    f0 = torch.cat(pitch_chunks, 1)
    periodicity_t = torch.cat(periodicity_chunks, 1)

    f0_hz = f0[0].cpu().numpy().astype(np.float64)
    periodicity = periodicity_t[0].cpu().numpy().astype(np.float64)

    # CREPE always emits its best guess; mark out-of-range guesses as unvoiced.
    out_of_range = (f0_hz < config.fmin_hz) | (f0_hz > config.fmax_hz)
    f0_hz = np.where(out_of_range, np.nan, f0_hz)
    periodicity = np.where(out_of_range, 0.0, periodicity)

    return f0_hz, np.clip(periodicity, 0.0, 1.0)


def _estimate_pyin(
    audio: np.ndarray,
    config: TranscriptionConfig,
    progress: Optional[ProgressReporter] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run probabilistic YIN, returning (f0_hz, voiced_probability).

    librosa runs pYIN as a single call with no progress hook, so this stage can
    only report its start and its elapsed time.
    """
    import librosa

    progress = progress or null_reporter()

    # frame_length must span at least two periods of the lowest pitch tracked.
    min_frame = int(2 * config.sample_rate / config.fmin_hz)
    frame_length = 1 << int(np.ceil(np.log2(max(min_frame, 1024))))

    with progress.stage("pYIN f0 estimation"):
        f0_hz, _voiced_flag, voiced_prob = librosa.pyin(
            y=audio,
            fmin=config.fmin_hz,
            fmax=config.fmax_hz,
            sr=config.sample_rate,
            frame_length=frame_length,
            hop_length=config.hop_length,
            center=True,
        )

    f0_hz = np.asarray(f0_hz, dtype=np.float64)          # NaN where unvoiced
    voiced_prob = np.nan_to_num(np.asarray(voiced_prob, dtype=np.float64))
    return f0_hz, np.clip(voiced_prob, 0.0, 1.0)


def estimate_f0(
    audio: np.ndarray,
    config: TranscriptionConfig,
    progress: Optional[ProgressReporter] = None,
) -> F0Track:
    """Estimate f0 and voicing for every frame of ``audio``.

    ``audio`` must already be mono at ``config.sample_rate``.
    """
    if config.f0_backend == "crepe":
        # estimate_f0 is public API, so 'auto' has to resolve here too rather
        # than only on the transcribe_audio path.
        if config.device == "auto":
            import dataclasses

            from .device import resolve_device

            config = dataclasses.replace(config, device=resolve_device("auto").device)
        f0_hz, periodicity = _estimate_crepe(audio, config, progress)
    elif config.f0_backend == "pyin":
        f0_hz, periodicity = _estimate_pyin(audio, config, progress)
    else:
        raise ValueError(f"Unknown f0_backend: {config.f0_backend!r}")

    f0_hz = _median_filter_f0(f0_hz, config.f0_median_filter_frames)

    n_frames = f0_hz.shape[0]
    times = np.arange(n_frames, dtype=np.float64) * (config.hop_length / config.sample_rate)
    loudness_db = frame_rms_db(audio, config.hop_length, n_frames)

    gate = _loudness_gate(loudness_db, config)
    voicing = periodicity * gate
    voicing = np.where(np.isfinite(f0_hz), voicing, 0.0)

    return F0Track(
        times=times,
        f0_hz=f0_hz,
        periodicity=periodicity,
        loudness_db=loudness_db,
        loudness_gate=gate,
        voicing=np.clip(voicing, 0.0, 1.0),
    )
