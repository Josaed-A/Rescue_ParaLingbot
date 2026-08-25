"""High-frequency RAM+VRAM+GPU monitor for the Linux/NVIDIA GPU baseline run.

Extends the CPU-only approach used in scripts_seq/monitor.py (see CLAUDE.md,
"Campana de caracterizacion secuencial") with GPU-side metrics via pynvml
(low-overhead NVML bindings, no subprocess-per-sample) plus torch's own
process-precise allocator counters. Runs as a background thread in the SAME
process that does load_model()/inference_streaming(), same rationale as before:
nothing to attach to externally without extra privileges, and this way sampling
covers the monolithic inference_streaming() call too, not just stage boundaries.

Adds a VRAM safety threshold on top of the existing RAM one: an 8GB laptop GPU
is a real OOM risk for a 200-frame sequence, unlike the earlier CPU/RAM
campaign on a 30GB-RAM machine which never got close to its limit.
"""
import csv
import os
import threading
import time

import psutil
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


class GPUMemoryMonitor:
    def __init__(self, csv_path, interval_s=0.3, safety_free_ram_mb=2048,
                 safety_free_vram_mb=400, gpu_index=0):
        self.interval_s = interval_s
        self.safety_free_ram_mb = safety_free_ram_mb
        self.safety_free_vram_mb = safety_free_vram_mb
        self.csv_path = csv_path
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self._nvml_handle = None
        if _NVML_OK:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._nvml_handle = None

        self.peak_rss_mb = 0.0
        self.peak_uss_mb = 0.0
        self.min_sys_avail_mb = float("inf")
        self.peak_vram_alloc_mb = 0.0   # torch-reported, this-process-precise
        self.peak_vram_reserved_mb = 0.0
        self.peak_vram_used_nvml_mb = 0.0  # whole-GPU, from NVML
        self.min_vram_free_nvml_mb = float("inf")
        self.peak_gpu_util_pct = 0.0
        self.peak_gpu_temp_c = 0.0
        self.peak_gpu_power_w = 0.0
        self.n_samples = 0
        self.safety_aborted = False
        self.abort_reason = None
        self._t0 = time.time()
        # Frame-index timeline populated externally via note_frame(), so memory
        # samples can be correlated to "which frame was in flight at t".
        self.frame_events = []  # list of (t_s, frame_idx)

        # Pre-warm psutil.cpu_percent's internal baseline (first call always
        # returns 0.0 otherwise).
        psutil.cpu_percent(interval=None)
        self._proc.cpu_percent(interval=None)

        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "t_s", "rss_mb", "uss_mb", "sys_avail_mb", "cpu_sys_pct", "cpu_proc_pct",
            "vram_alloc_mb", "vram_reserved_mb", "vram_used_nvml_mb", "vram_free_nvml_mb",
            "gpu_util_pct", "gpu_temp_c", "gpu_power_w",
        ])

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
            cpu_sys_pct = psutil.cpu_percent(interval=None)
            cpu_proc_pct = self._proc.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            return None, None

        vram_alloc_mb = torch.cuda.memory_allocated() / 1e6
        vram_reserved_mb = torch.cuda.memory_reserved() / 1e6

        vram_used_nvml_mb = vram_free_nvml_mb = float("nan")
        gpu_util_pct = gpu_temp_c = gpu_power_w = float("nan")
        if self._nvml_handle is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                vram_used_nvml_mb = mem.used / 1e6
                vram_free_nvml_mb = mem.free / 1e6
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util_pct = util.gpu
                gpu_temp_c = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                try:
                    gpu_power_w = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                except Exception:
                    gpu_power_w = float("nan")
            except Exception:
                pass

        t = time.time() - self._t0
        with self._lock:
            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
            if uss_mb == uss_mb:
                self.peak_uss_mb = max(self.peak_uss_mb, uss_mb)
            self.min_sys_avail_mb = min(self.min_sys_avail_mb, sys_avail_mb)
            self.peak_vram_alloc_mb = max(self.peak_vram_alloc_mb, vram_alloc_mb)
            self.peak_vram_reserved_mb = max(self.peak_vram_reserved_mb, vram_reserved_mb)
            if vram_used_nvml_mb == vram_used_nvml_mb:
                self.peak_vram_used_nvml_mb = max(self.peak_vram_used_nvml_mb, vram_used_nvml_mb)
                self.min_vram_free_nvml_mb = min(self.min_vram_free_nvml_mb, vram_free_nvml_mb)
            if gpu_util_pct == gpu_util_pct:
                self.peak_gpu_util_pct = max(self.peak_gpu_util_pct, gpu_util_pct)
            if gpu_temp_c == gpu_temp_c:
                self.peak_gpu_temp_c = max(self.peak_gpu_temp_c, gpu_temp_c)
            if gpu_power_w == gpu_power_w:
                self.peak_gpu_power_w = max(self.peak_gpu_power_w, gpu_power_w)
            self.n_samples += 1

        self._csv_writer.writerow([
            f"{t:.2f}", f"{rss_mb:.1f}", f"{uss_mb:.1f}", f"{sys_avail_mb:.1f}",
            f"{cpu_sys_pct:.1f}", f"{cpu_proc_pct:.1f}",
            f"{vram_alloc_mb:.1f}", f"{vram_reserved_mb:.1f}",
            f"{vram_used_nvml_mb:.1f}", f"{vram_free_nvml_mb:.1f}",
            f"{gpu_util_pct:.1f}", f"{gpu_temp_c:.1f}", f"{gpu_power_w:.1f}",
        ])
        self._csv_file.flush()
        return sys_avail_mb, vram_free_nvml_mb

    def _run(self):
        while not self._stop.is_set():
            sys_avail_mb, vram_free_mb = self._sample_once()
            if sys_avail_mb is not None and sys_avail_mb < self.safety_free_ram_mb:
                self._abort("ram_safety_threshold_breached")
                return
            if (vram_free_mb is not None and vram_free_mb == vram_free_mb
                    and vram_free_mb < self.safety_free_vram_mb):
                self._abort("vram_safety_threshold_breached")
                return
            self._stop.wait(self.interval_s)

    def _abort(self, reason):
        self.safety_aborted = True
        self.abort_reason = reason
        self._csv_writer.writerow([f"{time.time()-self._t0:.2f}", "SAFETY_ABORT", reason]
                                   + [""] * 10)
        self._csv_file.flush()
        self._csv_file.close()
        os._exit(75)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def snapshot(self, label):
        mi = self._proc.memory_info()
        try:
            uss_mb = self._proc.memory_full_info().uss / 1e6
        except Exception:
            uss_mb = float("nan")
        vm = psutil.virtual_memory()
        vram_alloc_mb = torch.cuda.memory_allocated() / 1e6
        vram_reserved_mb = torch.cuda.memory_reserved() / 1e6
        vram_used_nvml_mb = None
        if self._nvml_handle is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                vram_used_nvml_mb = round(mem.used / 1e6, 1)
            except Exception:
                pass
        return {
            "label": label,
            "t_s": round(time.time() - self._t0, 2),
            "rss_mb": round(mi.rss / 1e6, 1),
            "uss_mb": round(uss_mb, 1) if uss_mb == uss_mb else None,
            "sys_avail_mb": round(vm.available / 1e6, 1),
            "vram_alloc_mb": round(vram_alloc_mb, 1),
            "vram_reserved_mb": round(vram_reserved_mb, 1),
            "vram_used_nvml_mb": vram_used_nvml_mb,
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
                "peak_vram_alloc_mb": round(self.peak_vram_alloc_mb, 1),
                "peak_vram_reserved_mb": round(self.peak_vram_reserved_mb, 1),
                "peak_vram_used_nvml_mb": round(self.peak_vram_used_nvml_mb, 1),
                "min_vram_free_nvml_mb": round(self.min_vram_free_nvml_mb, 1),
                "peak_gpu_util_pct": round(self.peak_gpu_util_pct, 1),
                "peak_gpu_temp_c": round(self.peak_gpu_temp_c, 1),
                "peak_gpu_power_w": round(self.peak_gpu_power_w, 1),
                "n_samples": self.n_samples,
                "safety_aborted": self.safety_aborted,
                "abort_reason": self.abort_reason,
            }
        try:
            self._csv_file.close()
        except Exception:
            pass
        return result
