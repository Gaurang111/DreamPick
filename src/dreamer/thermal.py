"""GPU temperature monitoring for long training runs on a laptop GPU.

Shells out to nvidia-smi rather than adding a pynvml dependency - it's
already proven to work in this project and needs nothing extra installed.
"""

import subprocess
import time


def get_gpu_temperature_c() -> int | None:
    """Current GPU temperature in Celsius, or None if it can't be read
    (missing nvidia-smi, parse failure, etc. - callers should treat that as
    "unknown, don't throttle" rather than crashing a training run over it)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


class ThermalGuard:
    """Call `check()` periodically during training. If the GPU is at or
    above `pause_c`, it blocks (sleeping, re-checking every few seconds,
    printing status) until the temperature drops back below `resume_c` -
    a hysteresis gap avoids rapidly flapping pause/resume right at the
    threshold. `check()` itself only actually queries nvidia-smi at most
    once every `poll_interval_s` seconds, so calling it every training
    step is cheap.
    """

    def __init__(self, pause_c=85, resume_c=80, poll_interval_s=30, sleep_s=10, logger=None):
        self.pause_c = pause_c
        self.resume_c = resume_c
        self.poll_interval_s = poll_interval_s
        self.sleep_s = sleep_s
        self._logger = logger
        self._last_poll = 0.0
        self.last_temp_c = None
        self.total_paused_s = 0.0

    def _log(self, msg):
        print(msg, flush=True)

    def check(self):
        now = time.time()
        if now - self._last_poll < self.poll_interval_s:
            return
        self._last_poll = now

        temp = get_gpu_temperature_c()
        self.last_temp_c = temp
        if temp is None or temp < self.pause_c:
            return

        self._log(f"[thermal] GPU at {temp}C (>= {self.pause_c}C) - pausing training to cool down.")
        pause_start = time.time()
        while True:
            time.sleep(self.sleep_s)
            temp = get_gpu_temperature_c()
            self.last_temp_c = temp
            if temp is None:
                self._log("[thermal] Could not read GPU temperature - resuming to be safe.")
                break
            if temp < self.resume_c:
                self._log(f"[thermal] GPU cooled to {temp}C (< {self.resume_c}C) - resuming training.")
                break
            self._log(f"[thermal] still at {temp}C, waiting...")
        self.total_paused_s += time.time() - pause_start
        self._last_poll = time.time()
