"""
Tests for the Click CLI (pokemon_stencil.cli).

These tests exercise command parsing, option validation, and error paths
using Click's CliRunner – no actual pipeline execution or GPU required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from pokemon_stencil.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ─────────────────────────────────────────────────────────────────────────────
# Top-level
# ─────────────────────────────────────────────────────────────────────────────

class TestMainGroup:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Pokemon Stencil Art Factory" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_unknown_command(self, runner):
        result = runner.invoke(main, ["nonexistent"])
        assert result.exit_code != 0


# ─────────────────────────────────────────────────────────────────────────────
# generate command
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateCommand:
    def test_help(self, runner):
        result = runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0
        assert "POKEMON_NAME" in result.output

    def test_missing_pokemon_name_fails(self, runner):
        result = runner.invoke(main, ["generate"])
        assert result.exit_code != 0

    def test_skip_generation_without_refs_fails(self, runner, tmp_path):
        result = runner.invoke(
            main,
            ["generate", "Pikachu", "--skip-generation"],
        )
        assert result.exit_code != 0
        assert "refs" in result.output.lower() or "refs" in (result.exception and str(result.exception) or "")

    def test_refs_must_exist(self, runner):
        result = runner.invoke(
            main,
            ["generate", "Pikachu", "--refs", "/nonexistent/path/"],
        )
        assert result.exit_code != 0

    def test_n_colors_range_lower(self, runner):
        result = runner.invoke(
            main,
            ["generate", "Pikachu", "--n-colors", "1"],
        )
        assert result.exit_code != 0

    def test_n_colors_range_upper(self, runner):
        result = runner.invoke(
            main,
            ["generate", "Pikachu", "--n-colors", "99"],
        )
        assert result.exit_code != 0

    def test_steps_range(self, runner):
        result = runner.invoke(
            main,
            ["generate", "Pikachu", "--steps", "0"],
        )
        assert result.exit_code != 0


# ─────────────────────────────────────────────────────────────────────────────
# process command
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessCommand:
    def test_help(self, runner):
        result = runner.invoke(main, ["process", "--help"])
        assert result.exit_code == 0
        assert "IMAGE_PATH" in result.output

    def test_missing_image_path_fails(self, runner):
        result = runner.invoke(main, ["process"])
        assert result.exit_code != 0

    def test_nonexistent_image_fails(self, runner):
        result = runner.invoke(main, ["process", "/nonexistent/img.png"])
        assert result.exit_code != 0


# ─────────────────────────────────────────────────────────────────────────────
# batch command
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchCommand:
    def test_help(self, runner):
        result = runner.invoke(main, ["batch", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_FILE" in result.output
        assert "--workers" in result.output

    def test_missing_config_file_fails(self, runner):
        result = runner.invoke(main, ["batch"])
        assert result.exit_code != 0

    def test_nonexistent_config_file_fails(self, runner):
        result = runner.invoke(main, ["batch", "/nonexistent/config.json"])
        assert result.exit_code != 0

    def test_invalid_json_config_exits_nonzero(self, runner, tmp_path):
        bad_cfg = tmp_path / "bad.json"
        bad_cfg.write_text("not json", encoding="utf-8")
        result = runner.invoke(main, ["batch", str(bad_cfg)])
        assert result.exit_code != 0

    def test_config_missing_name_key_exits_nonzero(self, runner, tmp_path):
        bad_cfg = tmp_path / "bad.json"
        bad_cfg.write_text(
            json.dumps([{"refs": "refs/pikachu/"}]), encoding="utf-8"
        )
        result = runner.invoke(main, ["batch", str(bad_cfg)])
        assert result.exit_code != 0

    def test_workers_option_accepted(self, runner, tmp_path):
        """Verify --workers is a recognised option (parsing only)."""
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps([{"name": "Pikachu"}]), encoding="utf-8")
        # The command will fail at pipeline execution, but option parsing
        # must succeed (no "No such option" error).
        result = runner.invoke(main, ["batch", str(cfg), "--workers", "2"])
        assert "No such option" not in result.output


# ─────────────────────────────────────────────────────────────────────────────
# inspect command
# ─────────────────────────────────────────────────────────────────────────────

class TestInspectCommand:
    def test_help(self, runner):
        result = runner.invoke(main, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "IMAGE_PATH" in result.output

    def test_missing_image_path_fails(self, runner):
        result = runner.invoke(main, ["inspect"])
        assert result.exit_code != 0

    def test_nonexistent_image_fails(self, runner):
        result = runner.invoke(main, ["inspect", "/nonexistent/img.png"])
        assert result.exit_code != 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: _load_batch_config
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadBatchConfig:
    def test_valid_config(self, tmp_path):
        from pokemon_stencil.cli import _load_batch_config

        cfg = tmp_path / "cfg.json"
        data = [{"name": "Pikachu", "refs": "refs/pikachu/"}]
        cfg.write_text(json.dumps(data), encoding="utf-8")
        result = _load_batch_config(cfg)
        assert len(result) == 1
        assert result[0]["name"] == "Pikachu"

    def test_not_a_list_raises(self, tmp_path):
        from pokemon_stencil.cli import _load_batch_config

        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"name": "Pikachu"}), encoding="utf-8")
        with pytest.raises(ValueError, match="array"):
            _load_batch_config(cfg)

    def test_missing_name_raises(self, tmp_path):
        from pokemon_stencil.cli import _load_batch_config

        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps([{"refs": "refs/p/"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="name"):
            _load_batch_config(cfg)
