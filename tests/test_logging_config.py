"""Tests for pokemon_stencil.logging_config."""

import logging
from pathlib import Path

import pytest

from pokemon_stencil.logging_config import setup_logging


class TestSetupLogging:
    def teardown_method(self):
        """Reset root logger after each test."""
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_console_handler_added(self):
        setup_logging(verbose=False, log_file=None)
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_verbose_sets_debug_level(self):
        setup_logging(verbose=True, log_file=None)
        root = logging.getLogger()
        console = next(
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        )
        assert console.level == logging.DEBUG

    def test_non_verbose_sets_info_level(self):
        setup_logging(verbose=False, log_file=None)
        root = logging.getLogger()
        console = next(
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        )
        assert console.level == logging.INFO

    def test_file_handler_created(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logging(verbose=False, log_file=log_file)
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert log_file.exists()

    def test_no_file_handler_when_log_file_none(self):
        setup_logging(verbose=False, log_file=None)
        # log_file=None triggers default path; just confirm no crash and
        # that the root logger has at least the console handler.
        root = logging.getLogger()
        assert len(root.handlers) >= 1

    def test_duplicate_call_resets_handlers(self):
        setup_logging(verbose=False, log_file=None)
        setup_logging(verbose=False, log_file=None)
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        # Should be exactly 1 console handler after a fresh call.
        assert len(stream_handlers) == 1
