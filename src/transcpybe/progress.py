"""Progress reporting for long transcriptions.

Transcribing a four-minute vocal takes minutes on a CPU, and a batch of songs
takes hours, so the run has to say what it is doing while it does it.

Output adapts to where it is going. On a terminal, progress rewrites one line in
place. When redirected to a file -- which is how anything long actually gets
run, under tmux or nohup -- carriage returns would produce an unreadable mess,
so progress instead becomes a periodic timestamped line. The same code serves
both, and neither needs a flag.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Optional, TextIO


def format_duration(seconds: float) -> str:
    """Render a duration as m:ss, or h:mm:ss once it runs past an hour."""
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_realtime_factor(elapsed: float, audio_seconds: Optional[float]) -> str:
    """Express speed as a multiple of realtime, the useful unit for audio."""
    if not audio_seconds or audio_seconds <= 0 or elapsed <= 0:
        return ""
    return f"{elapsed / audio_seconds:.2f}x realtime"


class StageProgress:
    """A single named stage whose completion can be advanced."""

    def __init__(self, reporter: "ProgressReporter", name: str, total: Optional[float]) -> None:
        self._reporter = reporter
        self.name = name
        self.total = total
        self.done = 0.0
        self.started = time.time()
        self._last_render = 0.0
        self._detail = ""

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def fraction(self) -> Optional[float]:
        if not self.total or self.total <= 0:
            return None
        return min(self.done / self.total, 1.0)

    @property
    def eta(self) -> Optional[float]:
        fraction = self.fraction
        if not fraction or fraction <= 0:
            return None
        return self.elapsed * (1.0 - fraction) / fraction

    def advance(self, amount: float = 1.0, detail: str = "") -> None:
        self.done += amount
        if detail:
            self._detail = detail
        self._reporter._maybe_render(self)

    def render(self) -> str:
        parts = [self.name]
        fraction = self.fraction
        if fraction is not None:
            parts.append(f"{fraction * 100:5.1f}%")
            eta = self.eta
            if eta is not None and fraction < 1.0:
                parts.append(f"ETA {format_duration(eta)}")
        else:
            parts.append(format_duration(self.elapsed))
        if self._detail:
            parts.append(self._detail)
        return "  ".join(parts)


class ProgressReporter:
    """Routes stage progress to a terminal, a log, or nowhere."""

    def __init__(
        self,
        enabled: bool = True,
        stream: Optional[TextIO] = None,
        logger: Optional[logging.Logger] = None,
        interval: float = 5.0,
        audio_seconds: Optional[float] = None,
    ) -> None:
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.logger = logger
        self.interval = interval
        self.audio_seconds = audio_seconds
        self._line_open = False

    @property
    def interactive(self) -> bool:
        """True when progress can safely rewrite a line in place."""
        if not self.enabled:
            return False
        try:
            return bool(self.stream.isatty())
        except Exception:
            return False

    # -- line handling -----------------------------------------------------

    def _write_transient(self, text: str) -> None:
        try:
            self.stream.write("\r\033[K" + text)
            self.stream.flush()
            self._line_open = True
        except Exception:
            pass

    def _clear_line(self) -> None:
        if not self._line_open:
            return
        try:
            self.stream.write("\r\033[K")
            self.stream.flush()
        except Exception:
            pass
        self._line_open = False

    def log(self, message: str) -> None:
        """Emit a message without disturbing an open progress line."""
        self._clear_line()
        if self.logger is not None:
            self.logger.info(message)
        elif self.enabled:
            try:
                self.stream.write(message + "\n")
                self.stream.flush()
            except Exception:
                pass

    # -- stages ------------------------------------------------------------

    def _maybe_render(self, stage: StageProgress) -> None:
        if not self.enabled:
            return
        now = time.time()
        if self.interactive:
            # Cap redraws so a fast inner loop cannot spend its time on I/O.
            if now - stage._last_render < 0.1:
                return
            stage._last_render = now
            self._write_transient("  " + stage.render())
        else:
            if now - stage._last_render < self.interval:
                return
            stage._last_render = now
            self.log("  " + stage.render())

    @contextmanager
    def stage(
        self, name: str, total: Optional[float] = None, announce: bool = True
    ) -> Iterator[StageProgress]:
        """Run a named stage, reporting its progress and final elapsed time.

        ``announce`` writes a line as the stage begins, so a slow stage is not
        silent while it works. Turn it off for stages that finish in
        milliseconds, where the extra line is pure noise -- across a batch of
        several hundred songs those add up to a log nobody can read.
        """
        progress = StageProgress(self, name, total)
        if self.enabled and announce and not self.interactive:
            self.log(f"  {name}: started")
        try:
            yield progress
        finally:
            self._clear_line()
            if self.enabled:
                elapsed = progress.elapsed
                # A realtime factor is only meaningful once a stage has run long
                # enough to measure; below that it just prints "0.00x".
                rate = (format_realtime_factor(elapsed, self.audio_seconds)
                        if elapsed >= 1.0 else "")
                suffix = f" ({rate})" if rate else ""
                self.log(f"  {name}: done in {format_duration(elapsed)}{suffix}")


def null_reporter() -> ProgressReporter:
    """A reporter that emits nothing, for library use and tests."""
    return ProgressReporter(enabled=False)


def setup_logging(
    verbose: bool = False,
    log_file: Optional[str] = None,
    stream: Optional[TextIO] = None,
) -> logging.Logger:
    """Configure timestamped logging for the transcriber.

    Messages go to ``stream`` (stderr by default) and, when given, are appended
    to ``log_file`` as well, so a long run under tmux leaves a record that
    survives the session.
    """
    logger = logging.getLogger("transcpybe")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(stream if stream is not None else sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(console)

    if log_file:
        from pathlib import Path

        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    return logger
