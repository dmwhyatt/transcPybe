"""Frame-wise f0 estimation and voicing detection.

Interchangeable backends, chosen by ``config.f0_backend``:

* ``crepe`` - CREPE (Kim, Salamon, Li & Bello, ICASSP 2018) via ``torchcrepe``.
  A deep convolutional pitch tracker; the most accurate option for voice, and
  the default. Model weights ship inside the ``torchcrepe`` wheel.
* ``pyin`` - probabilistic YIN (Mauch & Dixon, ICASSP 2014) via ``librosa``.
  Pure signal processing, no torch dependency, a solid fallback.
* ``pesto`` - PESTO (Riou, Lattner, Hadjeres & Peeters, ISMIR 2023) via
  ``pesto-pitch``. A small self-supervised network trained on MIR-1K's
  singing, many times faster than CREPE on a CPU. Its code is LGPL-3.0, so it
  is an optional install rather than a dependency.

All of them produce the same ``F0Track`` so the rest of the pipeline is
agnostic, and ``register_f0_backend`` plugs in any other tracker the same way.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

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
    backend : str
        Which tracker produced it, e.g. 'crepe-full', 'crepe-tiny', 'pyin' or
        'pesto'. Empty for a track built by hand, and for one pickled before
        this was recorded. Anything trained on one tracker's output can check
        it, because ``periodicity`` means a different thing for each.
    """

    times: np.ndarray
    f0_hz: np.ndarray
    periodicity: np.ndarray
    loudness_db: np.ndarray
    loudness_gate: np.ndarray
    voicing: np.ndarray
    backend: str = ""

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


def median_filter_contour(values: np.ndarray, width: int) -> np.ndarray:
    """Running median of a pitch contour, leaving its unvoiced gaps as gaps.

    NaN marks an unvoiced frame. The contour is interpolated across each gap
    before filtering, so the frames beside a gap are smoothed against the
    contour carried across it rather than dragged towards NaN, and the gaps
    are restored afterwards.

    The f0 track runs this on log2 Hz and the glide backtrack on MIDI. The two
    scales differ by a linear rescale, which a median commutes with, so it is
    the same filter on either.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if width <= 1 or not valid.any():
        return values

    from scipy.ndimage import median_filter

    idx = np.arange(values.shape[0])
    filled = np.interp(idx, idx[valid], values[valid])
    smoothed = median_filter(filled, size=width, mode="nearest")
    return np.where(valid, smoothed, np.nan)


def _median_filter_f0(f0_hz: np.ndarray, width: int) -> np.ndarray:
    """Median filter in the log-frequency domain, ignoring unvoiced frames.

    Working in cents means the filter treats a halving and a doubling of f0
    symmetrically, so it removes isolated octave errors cleanly.
    """
    if width <= 1:
        return f0_hz

    valid = np.isfinite(f0_hz) & (f0_hz > 0)
    if not valid.any():
        return f0_hz

    log_f0 = np.full(f0_hz.shape, np.nan)
    log_f0[valid] = np.log2(f0_hz[valid])
    return 2.0 ** median_filter_contour(log_f0, width)


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


PESTO_CHECKPOINT = "mir-1k_g7"
"""The checkpoint ``pesto-pitch`` ships and recommends, trained on MIR-1K."""

_pesto_models: Dict[Tuple[float, int, str], Any] = {}
"""Loaded PESTO models by (step, rate, device), so a batch loads one once."""


def _estimate_pesto(
    audio: np.ndarray,
    config: TranscriptionConfig,
    progress: Optional[ProgressReporter] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PESTO, returning (f0_hz, confidence) on this package's frame grid.

    PESTO centres its constant-Q frames on multiples of its step, as CREPE does
    with padding, so at 16 kHz and a 10 ms step its frames are already these.
    It rounds the step to whole samples, though, so at other rates its frames
    drift; the contour is interpolated onto ``k * hop`` either way, in log
    frequency, and cut or extended to the frame count the other backends give.

    PESTO has no voicing decision. Its confidence stands in for periodicity,
    which is the nearest thing it offers, not the same thing: thresholds set on
    CREPE's periodicity are not known to suit it.
    """
    progress = progress or null_reporter()
    try:
        import torch
        import pesto
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 'pesto' backend needs torch and pesto-pitch:\n"
            "    pip install torch pesto-pitch\n"
            "Or use the torch-free fallback: --f0-backend pyin"
        ) from exc

    key = (float(config.hop_ms), int(config.sample_rate), str(config.device))
    model = _pesto_models.get(key)
    if model is None:
        model = pesto.load_model(PESTO_CHECKPOINT, step_size=config.hop_ms,
                                 sampling_rate=config.sample_rate).to(config.device)
        _pesto_models[key] = model

    audio_t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(config.device)
    with progress.stage(f"PESTO on {config.device}"), torch.inference_mode():
        pitch, confidence, _volume, _activations = model(
            audio_t, sr=config.sample_rate, convert_to_freq=True, return_activations=True
        )

    pitch_hz = pitch.cpu().numpy().astype(np.float64)
    confidence = confidence.cpu().numpy().astype(np.float64)

    hop_s = config.hop_length / config.sample_rate
    their_times = np.arange(pitch_hz.shape[0]) * (model.hop_size / 1000.0)
    our_times = np.arange(1 + audio.shape[0] // config.hop_length) * hop_s
    log_f0 = np.interp(our_times, their_times, np.log2(np.maximum(pitch_hz, 1e-6)))
    f0_hz = 2.0 ** log_f0
    periodicity = np.interp(our_times, their_times, confidence)

    # PESTO always names a pitch; mark out-of-range guesses as unvoiced.
    out_of_range = (f0_hz < config.fmin_hz) | (f0_hz > config.fmax_hz)
    f0_hz = np.where(out_of_range, np.nan, f0_hz)
    periodicity = np.where(out_of_range, 0.0, periodicity)

    return f0_hz, np.clip(periodicity, 0.0, 1.0)


F0Estimator = Callable[
    [np.ndarray, TranscriptionConfig, Optional[ProgressReporter]],
    Tuple[np.ndarray, np.ndarray],
]
"""``(audio, config, progress) -> (f0_hz, periodicity)``, one value per frame.

Frame ``k`` is centred on sample ``k * config.hop_length`` of the mono audio
at ``config.sample_rate``. ``f0_hz`` is NaN where the tracker found no pitch;
``periodicity`` is its confidence in [0, 1] that the frame is pitched."""


@dataclass(frozen=True)
class _Backend:
    estimate: F0Estimator
    uses_device: bool


_BACKENDS: Dict[str, _Backend] = {
    "crepe": _Backend(_estimate_crepe, uses_device=True),
    "pyin": _Backend(_estimate_pyin, uses_device=False),
    "pesto": _Backend(_estimate_pesto, uses_device=True),
}


def register_f0_backend(name: str, estimate: F0Estimator, *, uses_device: bool = False) -> None:
    """Make another pitch tracker available as ``config.f0_backend = name``.

    ``estimate`` follows ``F0Estimator``. Set ``uses_device`` for a torch
    tracker, so that ``config.device`` is resolved before it runs. Registering
    a name that exists replaces it, which is how a test or a caller swaps in
    a different build of a built-in tracker.
    """
    if not name or not name.strip():
        raise ValueError("an f0 backend needs a name")
    _BACKENDS[name] = _Backend(estimate, uses_device=uses_device)


def f0_backends() -> Tuple[str, ...]:
    """The names ``config.f0_backend`` accepts, built-in ones first."""
    return tuple(_BACKENDS)


def backend_uses_device(name: str) -> bool:
    """Whether ``config.device`` matters to the named backend."""
    backend = _BACKENDS.get(name)
    return bool(backend and backend.uses_device)


def backend_label(config: TranscriptionConfig) -> str:
    """The tracker a config selects, as ``F0Track.backend`` records it."""
    if config.f0_backend == "crepe":
        return f"crepe-{config.crepe_model}"
    return config.f0_backend


def estimate_f0(
    audio: np.ndarray,
    config: TranscriptionConfig,
    progress: Optional[ProgressReporter] = None,
) -> F0Track:
    """Estimate f0 and voicing for every frame of ``audio``.

    ``audio`` must already be mono at ``config.sample_rate``.
    """
    backend = _BACKENDS.get(config.f0_backend)
    if backend is None:
        raise ValueError(f"Unknown f0_backend: {config.f0_backend!r}; "
                         f"available: {', '.join(f0_backends())}")
    if backend.uses_device and config.device == "auto":
        # estimate_f0 is public API, so 'auto' has to resolve here too rather
        # than only on the transcribe_audio path.
        import dataclasses

        from .device import resolve_device

        config = dataclasses.replace(config, device=resolve_device("auto").device)
    f0_hz, periodicity = backend.estimate(audio, config, progress)

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
        backend=backend_label(config),
    )
