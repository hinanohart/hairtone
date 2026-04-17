from __future__ import annotations

import pytest

from hairtone.presets import PRESETS, Preset, get_preset, list_preset_names


def test_nineteen_presets() -> None:
    assert len(PRESETS) == 19


def test_all_presets_have_valid_lab() -> None:
    for p in PRESETS.values():
        L, A, B = p.lab
        assert 0 <= L <= 255, (p.name, L)
        assert 0 <= A <= 255, (p.name, A)
        assert 0 <= B <= 255, (p.name, B)
        assert p.hex_reference.startswith("#")
        assert len(p.hex_reference) == 7


def test_get_preset_known() -> None:
    preset = get_preset("blue")
    assert isinstance(preset, Preset)
    assert preset.pretty_name == "Blue"


def test_get_preset_unknown_lists_choices() -> None:
    with pytest.raises(KeyError) as exc:
        get_preset("totally-not-real")
    # Error message must hint available keys so users can self-correct.
    assert "blue" in str(exc.value)
    assert "blonde" in str(exc.value)


def test_list_preset_names_matches_dict_order() -> None:
    assert list_preset_names() == list(PRESETS.keys())


def test_preset_names_are_unique_lowercase_identifiers() -> None:
    names = list_preset_names()
    assert len(names) == len(set(names))
    for n in names:
        assert n == n.lower()
        assert n.replace("_", "").isalnum()
