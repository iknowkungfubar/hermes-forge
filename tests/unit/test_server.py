"""
Tests for hermes-forge ServerManager — budget resolution and lifecycle logic.

Focuses on pure-logic functions first: BudgetMode enum, resolve_budget()
branching, and _detect_vram_budget() VRAM mapping.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_forge.server import BudgetMode, ServerManager


class TestBudgetMode:
    def test_budget_mode_values(self):
        assert BudgetMode.BACKEND == "backend"
        assert BudgetMode.MANUAL == "manual"
        assert BudgetMode.FORGE_FULL == "forge-full"
        assert BudgetMode.FORGE_FAST == "forge-fast"

    def test_budget_mode_is_enum(self):
        assert issubclass(BudgetMode, str)


class TestServerManagerInit:
    def test_default_init(self):
        mgr = ServerManager()
        assert mgr.backend == "ollama"
        assert mgr.port == 8080
        assert mgr.models_dir is None
        assert mgr._process is None
        assert mgr._context_length is None

    def test_custom_init(self):
        mgr = ServerManager(backend="vllm", port=8000, models_dir="/models")
        assert mgr.backend == "vllm"
        assert mgr.port == 8000
        assert str(mgr.models_dir) == "/models"


class TestServerManagerGetContextLength:
    def test_context_length_none_by_default(self):
        mgr = ServerManager()
        assert mgr.get_context_length() is None

    def test_context_length_after_setting(self):
        mgr = ServerManager()
        mgr._context_length = 8192
        assert mgr.get_context_length() == 8192


class TestServerManagerResolveBudget:
    def test_backend_mode_with_context(self):
        mgr = ServerManager()
        mgr._context_length = 16384
        assert mgr.resolve_budget(BudgetMode.BACKEND) == 16384

    def test_backend_mode_without_context_fallback(self):
        mgr = ServerManager()
        assert mgr.resolve_budget(BudgetMode.BACKEND) == 4096

    def test_manual_mode_with_tokens(self):
        mgr = ServerManager()
        assert mgr.resolve_budget(BudgetMode.MANUAL, manual_tokens=2048) == 2048

    def test_manual_mode_without_tokens_raises(self):
        mgr = ServerManager()
        with pytest.raises(ValueError, match="manual_tokens required"):
            mgr.resolve_budget(BudgetMode.MANUAL)

    def test_forge_full_huge_vram(self):
        with patch(
            "hermes_forge.server.detect_hardware"
        ) as mock_detect:
            mock_detect.return_value.vram_total_gb = 64
            mgr = ServerManager()
            assert mgr.resolve_budget(BudgetMode.FORGE_FULL) == 262_144

    def test_forge_full_large_vram(self):
        with patch(
            "hermes_forge.server.detect_hardware"
        ) as mock_detect:
            mock_detect.return_value.vram_total_gb = 32
            mgr = ServerManager()
            assert mgr.resolve_budget(BudgetMode.FORGE_FULL) == 32_768

    def test_forge_full_medium_vram(self):
        with patch(
            "hermes_forge.server.detect_hardware"
        ) as mock_detect:
            mock_detect.return_value.vram_total_gb = 16
            mgr = ServerManager()
            assert mgr.resolve_budget(BudgetMode.FORGE_FULL) == 16_384

    def test_forge_fast_half_budget(self):
        with patch(
            "hermes_forge.server.detect_hardware"
        ) as mock_detect:
            mock_detect.return_value.vram_total_gb = 32
            mgr = ServerManager()
            assert mgr.resolve_budget(BudgetMode.FORGE_FAST) == 16_384

    def test_forge_full_small_vram(self):
        with patch(
            "hermes_forge.server.detect_hardware"
        ) as mock_detect:
            mock_detect.return_value.vram_total_gb = 4
            mgr = ServerManager()
            assert mgr.resolve_budget(BudgetMode.FORGE_FULL) == 4_096

    def test_forge_full_no_hardware(self):
        with patch(
            "hermes_forge.server.detect_hardware"
        ) as mock_detect:
            mock_detect.return_value = None
            mgr = ServerManager()
            assert mgr.resolve_budget(BudgetMode.FORGE_FULL) == 4_096

    def test_resolve_budget_with_missing_context(self):
        mgr = ServerManager()
        mgr._context_length = None
        result = mgr.resolve_budget(BudgetMode.BACKEND)
        assert result == 4096

    def test_resolve_budget_with_zero_context_fallback(self):
        mgr = ServerManager()
        mgr._context_length = 0
        result = mgr.resolve_budget(BudgetMode.BACKEND)
        assert result == 4096
