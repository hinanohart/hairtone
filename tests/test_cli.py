from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hairtone.cli import (
    _build_parser,
    _default_out_dir_for_all,
    _default_out_for_single,
    main,
)


def test_parser_basic() -> None:
    ns = _build_parser().parse_args(["photo.jpg", "blue"])
    assert str(ns.src) == "photo.jpg"
    assert ns.preset == "blue"
    assert ns.strength == 0.85
    # Default must point at the vendored BiSeNet so --bisenet-weights
    # works out of the box.
    assert ns.bisenet_module == "hairtone._vendor.bisenet"


def test_parser_rejects_unknown_preset() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["photo.jpg", "nonsense-preset"])


def test_list_presets_without_positional_args(capsys) -> None:
    rc = main(["--list-presets"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "blue" in out


def test_main_missing_positional_returns_2(capsys) -> None:
    rc = main([])
    err = capsys.readouterr().err
    assert rc == 2
    assert "required" in err.lower() or "error" in err.lower()


def test_default_out_single() -> None:
    assert _default_out_for_single(Path("/tmp/a.jpg"), "blue") == Path("/tmp/a_blue.jpg")


def test_default_out_all() -> None:
    assert _default_out_dir_for_all(Path("/tmp/a.png")) == Path("/tmp/a_hairtone")


def test_main_missing_source_returns_1(tmp_path, capsys) -> None:
    rc = main([str(tmp_path / "nope.jpg"), "blue"])
    out = capsys.readouterr()
    assert rc == 1
    assert "error" in out.err.lower()


def test_list_presets_shortcircuits(capsys) -> None:
    rc = main(["irrelevant.jpg", "blue", "--list-presets"])
    stdout = capsys.readouterr().out
    assert rc == 0
    assert "blonde" in stdout
    assert "blue" in stdout
    assert "purple" in stdout


def test_console_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hairtone.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "hairtone" in result.stdout


def test_version_flag() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hairtone.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "hairtone" in result.stdout
