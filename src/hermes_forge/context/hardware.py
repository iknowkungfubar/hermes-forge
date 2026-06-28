"""Hardware detection utilities for VRAM-based budget estimation.

Detects GPU hardware (NVIDIA via nvidia-smi, AMD via rocminfo) to
estimate available VRAM for context budget decisions.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("forge.context.hardware")


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""

    vram_total_gb: float
    vram_free_gb: float
    gpu_count: int
    gpu_name: str = "unknown"
    backend: str = "unknown"  # "nvidia", "amd", "apple", "none"


def detect_hardware() -> HardwareProfile | None:
    """Detect GPU hardware and return a HardwareProfile.

    Tries NVIDIA first (nvidia-smi), then AMD (rocminfo), then Apple (ioreg).
    Returns None if no GPU is detected.
    """
    # Try NVIDIA
    nvidia = _detect_nvidia()
    if nvidia is not None:
        return nvidia

    # Try AMD
    amd = _detect_amd()
    if amd is not None:
        return amd

    logger.info("No supported GPU detected")
    return None


def _detect_nvidia() -> HardwareProfile | None:
    """Detect NVIDIA GPU via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n")
        if not lines:
            return None

        total_vram = 0.0
        free_vram = 0.0
        gpu_name = "unknown"
        gpu_count = len(lines)

        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpu_name = parts[0]
                total_vram += float(parts[1]) / 1024  # MiB to GB
                free_vram += float(parts[2]) / 1024

        return HardwareProfile(
            vram_total_gb=round(total_vram, 1),
            vram_free_gb=round(free_vram, 1),
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            backend="nvidia",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        logger.debug(f"nvidia-smi detection failed: {e}")
        return None


def _detect_amd() -> HardwareProfile | None:
    """Detect AMD GPU via rocminfo."""
    if not shutil.which("rocminfo"):
        return None
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        output = result.stdout
        if "gfx" not in output:
            return None

        # Extract GPU name
        gpu_name = "unknown"
        for line in output.split("\n"):
            if "Name:" in line and "AMD" in line:
                gpu_name = line.split("Name:")[-1].strip()
                break

        # Try to get VRAM from rocm-smi
        if shutil.which("rocm-smi"):
            smi = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if smi.returncode == 0:
                total_match = __import__("re").search(
                    r"Total Memory \(VRAM\): (\d+)", smi.stdout
                )
                if total_match:
                    total_gb = float(total_match.group(1)) / 1024
                    return HardwareProfile(
                        vram_total_gb=round(total_gb, 1),
                        vram_free_gb=round(total_gb * 0.8, 1),
                        gpu_count=1,
                        gpu_name=gpu_name,
                        backend="amd",
                    )

        # Fallback: known AMD GPUs
        known_vram = {
            "7900 GRE": 16,
            "7900 XT": 20,
            "7900 XTX": 24,
            "7800 XT": 16,
            "7700 XT": 12,
        }
        for name, vram in known_vram.items():
            if name in gpu_name:
                return HardwareProfile(
                    vram_total_gb=float(vram),
                    vram_free_gb=float(vram) * 0.8,
                    gpu_count=1,
                    gpu_name=gpu_name,
                    backend="amd",
                )

        return HardwareProfile(
            vram_total_gb=16.0,
            vram_free_gb=12.0,
            gpu_count=1,
            gpu_name=gpu_name,
            backend="amd",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"AMD detection failed: {e}")
        return None
