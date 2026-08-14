# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from typing import Optional

_CHUNK_MILESTONE_RE = re.compile(
    r"epoch (\d+)/(\d+).*chunk (\d+)/(\d+)", re.IGNORECASE
)
_EPOCH_ONLY_RE = re.compile(r"epoch (\d+)/(\d+)", re.IGNORECASE)
_ALWAYS_LOG_KEYWORDS = (
    "EXCEPTION",
    "SUBENV",
    "send_rollout_trajectories",
    "interact() returning",
    "interact finished",
    "recv_rollout_trajectories",
    "final bootstrap",
    "generate() finished",
    "generate() entered",
    "training step",
)


def get_logger():
    """Get the logger instance of the current worker."""
    from rlinf.scheduler.worker import Worker

    return Worker.logger


def should_log_progress_milestone(msg: str) -> bool:
    """Return True for coarse-grained progress worth printing (throttled chunk/epoch)."""
    upper = msg.upper()
    if any(keyword.upper() in upper for keyword in _ALWAYS_LOG_KEYWORDS):
        return True

    chunk_match = _CHUNK_MILESTONE_RE.search(msg)
    if chunk_match:
        chunk_cur = int(chunk_match.group(3))
        chunk_tot = int(chunk_match.group(4))
        from rlinf.utils.robotwin_hang_diagnostics import chunk_log_interval

        interval = chunk_log_interval()
        if chunk_cur == 1 or chunk_cur == chunk_tot:
            return True
        if chunk_cur % interval == 0:
            return True
        return False

    epoch_match = _EPOCH_ONLY_RE.search(msg)
    if epoch_match:
        epoch_cur = int(epoch_match.group(1))
        epoch_tot = int(epoch_match.group(2))
        if "done" in msg.lower() or "start" in msg.lower() or "entered" in msg.lower():
            return True
        if epoch_cur == 1 or epoch_cur == epoch_tot:
            return True
        return False

    if msg.startswith("predict:"):
        return False
    if msg.startswith("env_interact_step:"):
        return False
    if "send_to done" in msg or "got actions" in msg or "waiting recv" in msg:
        return False
    return False


def log_progress(
    component: str,
    msg: str,
    *,
    rank: int | None = 0,
    logger=None,
    all_ranks: bool = False,
) -> None:
    """Print a flushed progress line for hang diagnosis.

    Goes to stdout immediately (picked up by ``tee`` into ``run_embodiment.log``).
    Optionally also writes through ``logger``. When ``rank`` is not ``None`` and
    ``all_ranks`` is False, only rank 0 emits (pass ``rank=None`` to always print,
    e.g. from the driver). Set ``all_ranks=True`` to emit from every worker rank.

    No-op unless RoboTwin train hang diagnostics are enabled (see
    ``robotwin_hang_diagnostics.configure_from_cfg``).
    """
    from rlinf.utils.robotwin_hang_diagnostics import enabled as _hang_diag_enabled

    if not _hang_diag_enabled():
        return

    import time

    if not all_ranks and rank is not None and rank != 0:
        return
    rank_s = f"r{rank} " if rank is not None else ""
    line = f"[PROGRESS {time.strftime('%H:%M:%S')}] [{component}] {rank_s}{msg}"
    print(line, flush=True)
    if logger is not None:
        try:
            logger.info(line)
        except Exception:
            pass


# Per-process hang-diagnosis state + heartbeat.
_PROGRESS_STATE: dict = {
    "component": None,
    "msg": None,
    "rank": None,
    "ts": None,
    "extra": None,
}
_HEARTBEAT_STARTED = False
_HEARTBEAT_LOCK = None


def set_progress_state(
    component: str,
    msg: str,
    *,
    rank: int | None = None,
    extra: str | None = None,
    log: bool = False,
    all_ranks: bool = True,
) -> None:
    """Update this process's current phase; optionally print immediately.

    By default only updates in-memory state for HEARTBEAT. Pass ``log=True`` for
    an immediate line, or rely on throttled milestone logging for chunk/epoch
    progress without flooding the log file.
    """
    from rlinf.utils.robotwin_hang_diagnostics import (
        enabled as _hang_diag_enabled,
        milestone_all_ranks,
    )

    if not _hang_diag_enabled():
        return

    import threading
    import time

    global _HEARTBEAT_LOCK
    if _HEARTBEAT_LOCK is None:
        _HEARTBEAT_LOCK = threading.Lock()
    with _HEARTBEAT_LOCK:
        _PROGRESS_STATE["component"] = component
        _PROGRESS_STATE["msg"] = msg
        _PROGRESS_STATE["rank"] = rank
        _PROGRESS_STATE["ts"] = time.time()
        _PROGRESS_STATE["extra"] = extra

    should_print = log or should_log_progress_milestone(msg)
    if not should_print:
        return

    print_all_ranks = all_ranks if log else milestone_all_ranks()
    suffix = f" | {extra}" if extra and log else ""
    log_progress(
        component,
        f"{msg}{suffix}",
        rank=rank,
        all_ranks=print_all_ranks,
    )


def _cuda_mem_summary() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda=N/A"
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return (
            f"cuda_free={free / 1e9:.2f}G/{total / 1e9:.2f}G "
            f"alloc={alloc / 1e9:.2f}G reserved={reserved / 1e9:.2f}G"
        )
    except Exception as e:
        return f"cuda_err={type(e).__name__}:{e}"


def start_progress_heartbeat(
    *,
    component: str,
    rank: int | None = None,
    interval_sec: Optional[float] = None,
) -> None:
    """Start a daemon thread that dumps current state on a throttled schedule."""
    from rlinf.utils.robotwin_hang_diagnostics import (
        enabled as _hang_diag_enabled,
        heartbeat_interval_sec,
        heartbeat_stuck_threshold_sec,
    )

    if not _hang_diag_enabled():
        return

    import threading
    import time

    global _HEARTBEAT_STARTED, _HEARTBEAT_LOCK
    if _HEARTBEAT_STARTED:
        return
    _HEARTBEAT_STARTED = True
    if _HEARTBEAT_LOCK is None:
        _HEARTBEAT_LOCK = threading.Lock()

    effective_interval = (
        heartbeat_interval_sec() if interval_sec is None else interval_sec
    )
    stuck_threshold = heartbeat_stuck_threshold_sec()

    def _loop() -> None:
        while True:
            time.sleep(effective_interval)
            with _HEARTBEAT_LOCK:
                st = dict(_PROGRESS_STATE)
            if st.get("ts") is None:
                continue
            stuck_for = time.time() - st["ts"]
            is_stuck = stuck_for >= stuck_threshold
            # Rank 0: periodic alive line; other ranks: only when likely hung.
            if rank not in (None, 0) and not is_stuck:
                continue
            mem_suffix = f" { _cuda_mem_summary()}" if is_stuck else ""
            log_progress(
                component,
                f"HEARTBEAT stuck_for={stuck_for:.1f}s "
                f"state={st.get('msg')} extra={st.get('extra')}{mem_suffix}",
                rank=rank,
                all_ranks=is_stuck,
            )

    t = threading.Thread(
        target=_loop,
        name=f"progress-heartbeat-{component}-r{rank}",
        daemon=True,
    )
    t.start()


_LIBAV_LOGS_SILENCED = False


def silence_libav_logs() -> None:
    """Suppress the ``[libdav1d @ 0x..] libdav1d 0.9.2`` chatter that
    torchcodec (LeRobot's default video backend) emits every frame.

    The messages are written by libav* **directly to file-descriptor 2**
    — pyav's log level does not intercept them because torchcodec
    bypasses pyav. We splice our own ``fd=2`` through a long-lived
    ``grep -v '\\[libdav1d'`` subprocess so every write to stderr is
    filtered line-by-line; all other stderr output (our own logs,
    tracebacks, etc.) still reaches the terminal.

    Idempotent: once the fd=2 redirect is installed, repeated calls return
    immediately so we never stack multiple ``grep`` filter subprocesses or
    re-duplicate the file descriptor.

    Call this once, as early as possible, before the heavy ``torch`` /
    ``torchcodec`` imports so the redirect is installed before libav loads.
    """
    global _LIBAV_LOGS_SILENCED
    if _LIBAV_LOGS_SILENCED:
        return

    import atexit
    import os
    import shutil
    import subprocess
    import sys

    # pyav's log level still helps when pyav IS the backend (older lerobot
    # fallback); cheap to do alongside the fd-level filter.
    try:
        import av

        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass

    if not shutil.which("grep"):
        return
    # Save the original fd=2 BEFORE we redirect — we need it back at exit
    # so grep can see EOF on its stdin (otherwise grep blocks forever and
    # ``atexit`` deadlocks because our fd=2 is still a write-end of the pipe).
    saved_stderr_fd = os.dup(2)
    try:
        # grep's stdout points at the ORIGINAL stderr so filtered lines
        # keep showing. grep's stdin becomes our new fd=2.
        grep = subprocess.Popen(
            # Match any libdav1d / libav* chatter — both the `[libdav1d @
            # 0x..] libdav1d 0.9.2` form AND any continuation line that
            # contains just `libdav1d`. Cast a wider net so child workers
            # whose output formatting differs don't slip through.
            ["grep", "-v", "-E", "--line-buffered", r"libdav1d|libdav1d 0\.9"],
            stdin=subprocess.PIPE,
            stdout=saved_stderr_fd,
        )
    except Exception:
        os.close(saved_stderr_fd)
        return

    sys.stderr.flush()
    os.dup2(grep.stdin.fileno(), 2)

    def _restore_and_drain() -> None:
        # Restore fd=2 first so subsequent stderr writes (e.g. tracebacks)
        # still reach the terminal — and crucially so grep sees EOF on its
        # stdin instead of waiting on us forever.
        try:
            os.dup2(saved_stderr_fd, 2)
        except OSError:
            pass
        try:
            os.close(saved_stderr_fd)
        except OSError:
            pass
        try:
            grep.stdin.close()
        except Exception:
            pass
        try:
            grep.wait(timeout=3)
        except Exception:
            grep.kill()

    atexit.register(_restore_and_drain)
    _LIBAV_LOGS_SILENCED = True
