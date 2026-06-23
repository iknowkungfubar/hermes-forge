"""Server lifecycle management — start/stop backends, resolve budgets.

ServerManager owns backend lifecycle (start/stop processes, health polling)
and resolves context budgets based on BudgetMode. It is the single point
of truth for "how much context can I use?" — clients just send messages.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any

from hermes_forge.context.hardware import HardwareProfile, detect_hardware

logger = logging.getLogger("forge.server")


class BudgetMode(str, Enum):
    """How to determine the context budget for compaction."""

    BACKEND = "backend"  # Trust the backend's default
    MANUAL = "manual"  # User specifies exact token count
    FORGE_FULL = "forge-full"  # Max safe context (auto-tune)
    FORGE_FAST = "forge-fast"  # Half of FORGE_FULL


class ServerManager:
    """Manages LLM backend processes and resolves context budgets.

    Supports: ollama, llamaserver, llamafile, vllm backends.
    """

    def __init__(
        self,
        backend: str = "ollama",
        port: int = 8080,
        models_dir: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.port = port
        self.models_dir = Path(models_dir) if models_dir else None
        self._process: subprocess.Popen | None = None
        self._context_length: int | None = None

    async def start(
        self,
        model_path: str | Path | None = None,
        gguf_path: str | Path | None = None,
        ctx_override: int | None = None,
        mode: str = "native",
        extra_flags: list[str] | None = None,
    ) -> None:
        """Start the backend process.

        Args:
            model_path: Model path/directory (vLLM) or model name (Ollama).
            gguf_path: Path to .gguf file (llamaserver/llamafile).
            ctx_override: Override context length for the backend.
            mode: "native" or "prompt-injected" (for llamaserver).
            extra_flags: Additional flags for the backend.
        """
        if self.backend == "ollama":
            await self._start_ollama(model_path)
        elif self.backend in ("llamaserver", "llamafile"):
            await self._start_llama_server(gguf_path, ctx_override, mode, extra_flags)
        elif self.backend == "vllm":
            await self._start_vllm(model_path, ctx_override)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    async def _start_ollama(self, model_name: str | Path | None) -> None:
        """Ensure Ollama is running and the model is available."""
        logger.info(f"Starting Ollama with model: {model_name}")
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning("Ollama not running. Attempting to start...")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.sleep(3)

        if model_name:
            model_str = str(model_name)
            if model_str not in result.stdout:
                logger.info(f"Pulling model {model_str}...")
                subprocess.run(
                    ["ollama", "pull", model_str],
                    capture_output=True, text=True, timeout=300,
                )

    async def _start_llama_server(
        self,
        gguf_path: str | Path | None,
        ctx_override: int | None = None,
        mode: str = "native",
        extra_flags: list[str] | None = None,
    ) -> None:
        """Start llama-server or llamafile."""
        if not gguf_path:
            raise ValueError("gguf_path is required for llamaserver/llamafile backend")

        gguf = Path(gguf_path)
        if not gguf.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf}")

        cmd = [
            "llama-server",
            "-m", str(gguf),
            "-ngl", "999",
            "--port", str(self.port),
        ]
        if mode == "native":
            cmd.append("--jinja")
        if ctx_override is not None:
            cmd.extend(["-c", str(ctx_override)])
        if extra_flags:
            cmd.extend(extra_flags)

        logger.info(f"Starting llama-server: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await self._wait_healthy(timeout=180)

    async def _start_vllm(
        self,
        model_path: str | Path | None,
        ctx_override: int | None = None,
    ) -> None:
        """Start vLLM server."""
        if not model_path:
            raise ValueError("model_path is required for vLLM backend")

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_path),
            "--port", str(self.port),
        ]
        if ctx_override:
            cmd.extend(["--max-model-len", str(ctx_override)])

        logger.info(f"Starting vLLM: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await self._wait_healthy(timeout=300)

    async def stop(self) -> None:
        """Stop the backend process gracefully."""
        if self.backend == "ollama" and self._process is None:
            return  # Ollama managed externally

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            await asyncio.sleep(3)  # Let VRAM clear

    async def _wait_healthy(self, timeout: int = 180) -> None:
        """Wait for the backend to become healthy."""
        import httpx

        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                try:
                    if self.backend in ("llamaserver", "llamafile"):
                        resp = await client.get(
                            f"http://127.0.0.1:{self.port}/props",
                            timeout=5,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            if "default_generation_settings" in data:
                                self._context_length = data[
                                    "default_generation_settings"
                                ].get("n_ctx")
                                return
                    elif self.backend == "vllm":
                        resp = await client.get(
                            f"http://127.0.0.1:{self.port}/v1/models",
                            timeout=5,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            models = data.get("data", [])
                            if models:
                                self._context_length = models[0].get(
                                    "max_model_len"
                                )
                                return
                    else:
                        return  # Ollama — assume healthy
                except Exception:
                    pass
                await asyncio.sleep(2)

        raise TimeoutError(f"Backend did not become healthy within {timeout}s")

    def get_context_length(self) -> int | None:
        """Get the detected context length of the backend."""
        return self._context_length

    def resolve_budget(
        self,
        mode: BudgetMode = BudgetMode.BACKEND,
        manual_tokens: int | None = None,
    ) -> int:
        """Resolve the context budget based on mode.

        Returns the token budget for context compaction.
        """
        if mode == BudgetMode.MANUAL:
            if manual_tokens is not None:
                return manual_tokens
            raise ValueError("manual_tokens required for MANUAL budget mode")

        if mode in (BudgetMode.FORGE_FULL, BudgetMode.FORGE_FAST):
            detected = self._detect_vram_budget()
            if mode == BudgetMode.FORGE_FAST:
                return detected // 2
            return detected

        # BACKEND mode — use detected or default
        if self._context_length:
            return self._context_length
        return 4096

    def _detect_vram_budget(self) -> int:
        """Detect hardware and return appropriate context budget."""
        hw = detect_hardware()
        if hw is None:
            return 4096
        vram_gb = hw.vram_total_gb
        if vram_gb >= 48:
            return 262_144
        elif vram_gb >= 24:
            return 32_768
        elif vram_gb >= 16:
            return 16_384
        elif vram_gb >= 8:
            return 8_192
        else:
            return 4_096
