"""Hardware detection and inference recommendations."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HardwareInfo:
    os_name: str
    os_version: str
    python_version: str
    cpu_count: int
    ram_mb: int | None
    cuda_available: bool
    gpu_name: str | None
    vram_mb: int | None
    gpus: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InferenceRecommendation:
    device: str  # "cuda" | "cpu"
    profile: str  # "gpu_8gb" | "cpu" | ...
    quantization: str
    context_length: int
    batch_size: int
    n_gpu_layers: int
    n_threads: int
    notes: list[str] = field(default_factory=list)


class HardwareDetector:
    """Detect CPU/RAM/GPU and recommend inference settings."""

    def detect(self) -> HardwareInfo:
        ram_mb = self._detect_ram_mb()
        cuda, gpu_name, vram_mb, gpus = self._detect_cuda()
        return HardwareInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            cpu_count=os.cpu_count() or 1,
            ram_mb=ram_mb,
            cuda_available=cuda,
            gpu_name=gpu_name,
            vram_mb=vram_mb,
            gpus=gpus,
        )

    def recommend(
        self,
        prefer_gpu: bool = True,
        min_vram_mb_for_gpu: int = 4096,
    ) -> tuple[HardwareInfo, InferenceRecommendation]:
        info = self.detect()
        notes: list[str] = []
        use_gpu = bool(
            prefer_gpu
            and info.cuda_available
            and info.vram_mb is not None
            and info.vram_mb >= min_vram_mb_for_gpu
        )
        if prefer_gpu and not use_gpu:
            notes.append("GPU unavailable or VRAM below threshold; using CPU.")

        if use_gpu:
            vram = info.vram_mb or 0
            if vram >= 7000:
                profile = "gpu_8gb"
                ctx = 4096
                batch = 8
                quant = "Q4_K_M"
            else:
                profile = "gpu_low_vram"
                ctx = 2048
                batch = 4
                quant = "Q4_K_M"
                notes.append("Limited VRAM: prefer 3B GGUF and lower context.")
            rec = InferenceRecommendation(
                device="cuda",
                profile=profile,
                quantization=quant,
                context_length=ctx,
                batch_size=batch,
                n_gpu_layers=-1,
                n_threads=max(1, (info.cpu_count or 4) // 2),
                notes=notes,
            )
        else:
            rec = InferenceRecommendation(
                device="cpu",
                profile="cpu",
                quantization="Q4_K_M",
                context_length=2048,
                batch_size=2,
                n_gpu_layers=0,
                n_threads=max(1, info.cpu_count or 4),
                notes=notes or ["CPU mode selected."],
            )
        return info, rec

    def _detect_ram_mb(self) -> int | None:
        try:
            if platform.system() == "Windows":
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return int(stat.ullTotalPhys // (1024 * 1024))
            # Unix
            pages = os.sysconf("SC_PAGE_SIZE")
            count = os.sysconf("SC_PHYS_PAGES")
            return int(pages * count // (1024 * 1024))
        except Exception:
            return None

    def _detect_cuda(
        self,
    ) -> tuple[bool, str | None, int | None, list[dict[str, Any]]]:
        gpus: list[dict[str, Any]] = []
        try:
            import torch

            if not torch.cuda.is_available():
                return False, None, None, []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "index": i,
                        "name": props.name,
                        "vram_mb": int(props.total_memory // (1024 * 1024)),
                    }
                )
            best = gpus[0]
            return True, best["name"], best["vram_mb"], gpus
        except Exception:
            pass
        # Fallback: nvidia-smi without torch
        try:
            import subprocess

            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            line = out.strip().splitlines()[0]
            name, mem = [x.strip() for x in line.split(",")]
            vram = int(float(mem))
            gpus = [{"index": 0, "name": name, "vram_mb": vram}]
            return True, name, vram, gpus
        except Exception:
            return False, None, None, []
