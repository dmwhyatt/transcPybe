"""Torch device selection and GPU reporting.

Kept separate from the transcription code so that the pyin backend, which needs
no torch at all, never imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceInfo:
    """What was selected, and what the hardware actually is."""

    device: str
    """The resolved torch device string, e.g. 'cuda:0'."""

    kind: str
    """'cuda', 'mps' or 'cpu'."""

    name: str
    """Human-readable hardware name."""

    total_memory_mb: Optional[float] = None
    requested: str = "auto"
    """What the caller asked for, before resolution."""

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "mps")

    def describe(self) -> str:
        parts = [f"{self.device} ({self.name}"]
        if self.total_memory_mb:
            parts.append(f", {self.total_memory_mb / 1024:.1f} GiB")
        parts.append(")")
        text = "".join(parts)
        if self.requested == "auto":
            text += " [auto-selected]"
        return text


def resolve_device(requested: str = "auto") -> DeviceInfo:
    """Resolve a device request into a concrete device.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. An explicit request is
    honoured as given, so that a deliberate choice is never silently overridden
    -- if CUDA is asked for and is missing, torch raises, which is more useful
    than quietly running 30x slower on the CPU.
    """
    try:
        import torch
    except ImportError:
        return DeviceInfo(device="cpu", kind="cpu", name="CPU (torch not installed)",
                          requested=requested)

    if requested != "auto":
        kind = requested.split(":")[0]
        name = _hardware_name(torch, kind, requested)
        return DeviceInfo(device=requested, kind=kind, name=name,
                          total_memory_mb=_total_memory_mb(torch, kind, requested),
                          requested=requested)

    if torch.cuda.is_available():
        device = "cuda:0"
        return DeviceInfo(device=device, kind="cuda",
                          name=_hardware_name(torch, "cuda", device),
                          total_memory_mb=_total_memory_mb(torch, "cuda", device),
                          requested=requested)

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return DeviceInfo(device="mps", kind="mps", name="Apple Silicon GPU (MPS)",
                          requested=requested)

    return DeviceInfo(device="cpu", kind="cpu", name=_cpu_name(), requested=requested)


def _hardware_name(torch, kind: str, device: str) -> str:
    if kind == "cuda":
        try:
            return torch.cuda.get_device_name(device)
        except Exception:
            return "CUDA device"
    if kind == "mps":
        return "Apple Silicon GPU (MPS)"
    return _cpu_name()


def _cpu_name() -> str:
    import os
    import platform

    cores = os.cpu_count() or 1
    machine = platform.machine() or "CPU"
    return f"CPU ({cores} cores, {machine})"


def _total_memory_mb(torch, kind: str, device: str) -> Optional[float]:
    if kind != "cuda":
        return None
    try:
        return torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    except Exception:
        return None


def memory_summary(device: str) -> Optional[str]:
    """Current allocated/reserved GPU memory, or None off CUDA.

    Reported during long transcriptions so that an out-of-memory failure on a
    shared GPU box can be seen coming rather than diagnosed afterwards.
    """
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        return (f"GPU memory: {allocated:.0f} MiB allocated, "
                f"{reserved:.0f} MiB reserved, {total:.0f} MiB total")
    except Exception:
        return None


def synchronize(device: str) -> None:
    """Wait for queued GPU work, so timings measure compute and not queueing."""
    if not device.startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        pass
