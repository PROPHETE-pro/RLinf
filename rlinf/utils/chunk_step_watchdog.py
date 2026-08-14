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

"""Process-level watchdog for RoboTwin/SAPIEN ``chunk_step`` hangs.

Why a separate OS process?
  Native hangs inside SAPIEN PhysX / RT render / mplib TOPP often hold the
  Python GIL. In-process threads (heartbeat, ``concurrent.futures`` timeouts)
  cannot interrupt them. An external process can still observe a stamp file
  written *before* entering the critical section and SIGKILL the hung worker.

Normal training is unaffected when ``timeout_sec`` is large vs typical
``chunk_step`` latency (~2–3s). Default recommended: 60s.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Iterator, Optional


_WATCHDOG_CHILD_CODE = r"""
import os, sys, time, signal

parent_pid = int(sys.argv[1])
stamp_path = sys.argv[2]
timeout_sec = float(sys.argv[3])
poll_sec = float(sys.argv[4])
rank = sys.argv[5]

def parent_alive():
    try:
        os.kill(parent_pid, 0)
        return True
    except OSError:
        return False

def read_stamp():
    try:
        with open(stamp_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if not lines:
            return None
        if lines[0] != "ARMED":
            return None
        deadline = float(lines[1])
        stage = lines[2] if len(lines) > 2 else "?"
        return deadline, stage
    except Exception:
        return None

while parent_alive():
    time.sleep(poll_sec)
    stamp = read_stamp()
    if stamp is None:
        continue
    deadline, stage = stamp
    now = time.time()
    if now < deadline:
        continue
    overdue = now - (deadline - timeout_sec)
    msg = (
        f"[CHUNK_STEP_WATCHDOG] rank={rank} stage={stage} "
        f"chunk_step exceeded {timeout_sec:.1f}s "
        f"(elapsed~{overdue:.1f}s). Killing EnvWorker pid={parent_pid} "
        f"to unblock training (SAPIEN/mplib hang suspected).\n"
    )
    try:
        sys.stderr.write(msg)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        os.kill(parent_pid, signal.SIGKILL)
    except OSError:
        pass
    break
"""


class ChunkStepWatchdog:
    """Long-lived child process that SIGKILLs this worker on ``chunk_step`` hang."""

    def __init__(
        self,
        timeout_sec: float,
        *,
        rank: int = 0,
        poll_sec: float = 0.5,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
        self.timeout_sec = float(timeout_sec)
        self.rank = int(rank)
        self.poll_sec = float(poll_sec)
        self._stamp_path = os.path.join(
            tempfile.gettempdir(),
            f"rlinf_chunk_step_watchdog_r{self.rank}_p{os.getpid()}.stamp",
        )
        self._proc: Optional[subprocess.Popen] = None
        self._disarmed()
        self._start_child()
        atexit.register(self.shutdown)

    def _disarmed(self) -> None:
        try:
            with open(self._stamp_path, "w", encoding="utf-8") as f:
                f.write("CLEAR\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass

    def _arm(self, stage_id: int) -> None:
        deadline = time.time() + self.timeout_sec
        payload = f"ARMED\n{deadline:.6f}\n{stage_id}\n"
        # Atomic-ish replace via temp + rename so the child never reads a partial write.
        tmp_path = self._stamp_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._stamp_path)

    def _start_child(self) -> None:
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WATCHDOG_CHILD_CODE,
                str(os.getpid()),
                self._stamp_path,
                str(self.timeout_sec),
                str(self.poll_sec),
                str(self.rank),
            ],
            # Inherit stderr so the kill message lands in run_embodiment.log via tee.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,
            start_new_session=True,
        )

    def shutdown(self) -> None:
        self._disarmed()
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            os.remove(self._stamp_path)
        except OSError:
            pass

    @contextmanager
    def guard(self, *, stage_id: int = 0) -> Iterator[None]:
        """Arm watchdog around a ``chunk_step`` call; always disarm on exit."""
        self._arm(stage_id)
        try:
            yield
        finally:
            self._disarmed()


def create_chunk_step_watchdog(
    timeout_sec: Optional[float],
    *,
    rank: int = 0,
) -> Optional[ChunkStepWatchdog]:
    """Factory: returns ``None`` when timeout is disabled (``<= 0`` or ``None``)."""
    if timeout_sec is None:
        return None
    timeout = float(timeout_sec)
    if timeout <= 0:
        return None
    return ChunkStepWatchdog(timeout_sec=timeout, rank=rank)
