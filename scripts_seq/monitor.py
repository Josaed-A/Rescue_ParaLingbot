"""High-frequency external memory monitor for the sequential-campaign experiments.

Linux equivalent of the Windows `measure_ram.ps1` used in the earlier rounds of this
investigation (see CLAUDE.md). Runs as a background thread inside the SAME process
that does load_model()/inference_streaming() (there's nothing to attach to externally
without extra privileges the way the Windows ctypes/CIM approach did), sampling at a
fixed wall-clock interval so it captures memory during the monolithic
`inference_streaming()` call too, not just at stage boundaries.

Tracks a peak (max over all samples, not just at snapshot points) and enforces the
safety threshold requested by the user: if system-available RAM drops below
`safety_free_mb`, abort immediately via os._exit() rather than waiting for the kernel
OOM killer or an actual crash.
"""
import csv
import os
import threading
import time

import psutil


class MemoryMonitor:
    def __init__(self, csv_path, interval_s=0.5, safety_free_mb=2048):
        self.interval_s = interval_s
        self.safety_free_mb = safety_free_mb
        self.csv_path = csv_path
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self.peak_rss_mb = 0.0
        self.peak_uss_mb = 0.0
        self.min_sys_avail_mb = float("inf")
        self.peak_cpu_sys_pct = 0.0
        self.peak_cpu_proc_pct = 0.0
        self.n_samples = 0
        self.safety_aborted = False
        self._t0 = time.time()
        # Frame-index timeline, populated externally via note_frame(), so
        # memory/CPU samples can be correlated to "which frame was in flight at t".
        self.frame_events = []  # list of (t_s, frame_idx)

        # Pre-warm psutil.cpu_percent's internal baseline (first call always
        # returns 0.0 otherwise).
        psutil.cpu_percent(interval=None)
        self._proc.cpu_percent(interval=None)

        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["t_s", "rss_mb", "uss_mb", "sys_avail_mb", "sys_used_pct",
             "cpu_sys_pct", "cpu_proc_pct"]
        )

    def note_frame(self, frame_idx):
        """Called from the main thread when a frame finishes, to timestamp it."""
        self.frame_events.append((time.time() - self._t0, frame_idx))

    def _sample_once(self):
        try:
            mi = self._proc.memory_info()
            rss_mb = mi.rss / 1e6
            try:
                uss_mb = self._proc.memory_full_info().uss / 1e6
            except Exception:
                uss_mb = float("nan")
            vm = psutil.virtual_memory()
            sys_avail_mb = vm.available / 1e6
            sys_used_pct = vm.percent
            cpu_sys_pct = psutil.cpu_percent(interval=None)
            cpu_proc_pct = self._proc.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            return None

        t = time.time() - self._t0
        with self._lock:
            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
            if uss_mb == uss_mb:  # not NaN
                self.peak_uss_mb = max(self.peak_uss_mb, uss_mb)
            self.min_sys_avail_mb = min(self.min_sys_avail_mb, sys_avail_mb)
            self.peak_cpu_sys_pct = max(self.peak_cpu_sys_pct, cpu_sys_pct)
            self.peak_cpu_proc_pct = max(self.peak_cpu_proc_pct, cpu_proc_pct)
            self.n_samples += 1

        self._csv_writer.writerow([f"{t:.2f}", f"{rss_mb:.1f}", f"{uss_mb:.1f}",
                                    f"{sys_avail_mb:.1f}", f"{sys_used_pct:.1f}",
                                    f"{cpu_sys_pct:.1f}", f"{cpu_proc_pct:.1f}"])
        self._csv_file.flush()
        return sys_avail_mb

    def _run(self):
        while not self._stop.is_set():
            sys_avail_mb = self._sample_once()
            if sys_avail_mb is not None and sys_avail_mb < self.safety_free_mb:
                self.safety_aborted = True
                self._csv_writer.writerow(
                    [f"{time.time()-self._t0:.2f}", "SAFETY_ABORT", "", f"{sys_avail_mb:.1f}", "", "", ""]
                )
                self._csv_file.flush()
                self._csv_file.close()
                # Hard exit: a background thread can't safely unwind a torch call on
                # the main thread, and the whole point is to not wait around for the
                # kernel OOM killer (which is what produced the Windows APPCRASH
                # earlier in this investigation, see CLAUDE.md).
                os._exit(75)  # EX_TEMPFAIL-ish, distinct from a plain crash (137/-9)
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def snapshot(self, label):
        """Discrete stage-boundary snapshot, in addition to the continuous CSV."""
        mi = self._proc.memory_info()
        try:
            uss_mb = self._proc.memory_full_info().uss / 1e6
        except Exception:
            uss_mb = float("nan")
        vm = psutil.virtual_memory()
        return {
            "label": label,
            "t_s": round(time.time() - self._t0, 2),
            "rss_mb": round(mi.rss / 1e6, 1),
            "uss_mb": round(uss_mb, 1) if uss_mb == uss_mb else None,
            "sys_avail_mb": round(vm.available / 1e6, 1),
        }

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            result = {
                "peak_rss_mb": round(self.peak_rss_mb, 1),
                "peak_uss_mb": round(self.peak_uss_mb, 1),
                "min_sys_avail_mb": round(self.min_sys_avail_mb, 1),
                "peak_cpu_sys_pct": round(self.peak_cpu_sys_pct, 1),
                "peak_cpu_proc_pct": round(self.peak_cpu_proc_pct, 1),
                "n_samples": self.n_samples,
                "safety_aborted": self.safety_aborted,
            }
        try:
            self._csv_file.close()
        except Exception:
            pass
        return result
