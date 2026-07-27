# Hermes Forge Repository Improvements

This run is cron job `2026-07-27T05:04:05Z`.

## Issues Fixed

### ASYNC220/ASYNC221 — Blocking subprocess calls in async functions (`server.py`)
- Replaced `subprocess.Popen()` with `asyncio.create_subprocess_exec()` in `_start_llama_server()`, `_start_vllm()`, and `_start_ollama()`
- Replaced `subprocess.run()` with `asyncio.create_subprocess_exec()` + `communicate()` in `_start_ollama()`
- Updated `stop()` method to use `await asyncio.wait_for(self._process.wait())` instead of blocking `self._process.wait(timeout=10)`
- Changed type annotation from `subprocess.Popen | None` to `asyncio.subprocess.Process | None`

### S110/BLE001 — Bare `except Exception: pass` in `server.py`
- Replaced `except Exception: pass` in `_wait_healthy()` with `logger.debug("Backend not ready yet", exc_info=True)`

### UP041 — Use builtin `TimeoutError` instead of `asyncio.TimeoutError`
- Replaced 3 occurrences of `except asyncio.TimeoutError` with `except TimeoutError` across `server.py`

### F401 — Unused import
- Removed unused `import subprocess` from `server.py`

### CI Status
- All 19 lint errors in `server.py` eliminated
- CI lint job now passes on server.py
- All 149 tests pass (49.67% coverage, above 40% threshold)

### Pre-existing issues (in other files — not addressed in this run)
- BLE001 in: openai_compat.py, vllm.py, slot_worker.py, handler.py, server.py (proxy)
- S110/S112 in: slot_worker.py, proxy/server.py