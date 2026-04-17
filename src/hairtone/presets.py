"""Declarative colour presets.

Each preset stores a target colour in OpenCV's LAB space (L in 0..255, A/B
centred at 128). Values were tuned empirically against a portrait dataset
and matched against the hex reference printed below for reviewability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Preset:
    """A named LAB colour target."""

    name: str
    pretty_name: str
    lab: tuple[int, int, int]
    hex_reference: str


PRESETS: Final[dict[str, Preset]] = {
    p.name: p
    for p in (
        Preset("blonde",      "Blonde",      (200, 125, 165), "#d9c277"),
        Preset("honey",       "Honey",       (185, 128, 170), "#c5a85e"),
        Preset("strawberry",  "Strawberry",  (165, 155, 150), "#b07c74"),
        Preset("pastel_pink", "Pastel Pink", (190, 145, 130), "#d9a8b6"),
        Preset("coral",       "Coral",       (170, 155, 160), "#b8756a"),
        Preset("lavender",    "Lavender",    (175, 145, 115), "#a094c9"),
        Preset("pink",        "Pink",        (155, 160, 130), "#a86a86"),
        Preset("hotpink",     "Hot Pink",    (130, 175, 145), "#95285b"),
        Preset("red",         "Red",         (120, 170, 155), "#8a2a2a"),
        Preset("orange",      "Orange",      (165, 150, 180), "#c07038"),
        Preset("blue",        "Blue",        (155, 120, 95),  "#4864a8"),
        Preset("cyan",        "Cyan",        (175, 105, 110), "#3a9fb4"),
        Preset("teal",        "Teal",        (155, 108, 108), "#2f8a8c"),
        Preset("turquoise",   "Turquoise",   (170, 108, 112), "#38a0a8"),
        Preset("green",       "Green",       (160, 100, 155), "#3a9a60"),
        Preset("purple",      "Purple",      (130, 155, 100), "#5a3088"),
        Preset("silver",      "Silver",      (195, 128, 127), "#b8b8b8"),
        Preset("ash",         "Ash",         (170, 128, 126), "#989890"),
        Preset("mint",        "Mint",        (195, 110, 130), "#95c8a8"),
    )
}


def list_preset_names() -> list[str]:
    """Return preset keys in declaration order."""
    return list(PRESETS.keys())


def get_preset(name: str) -> Preset:
    """Look up a preset by its key. Raises ``KeyError`` with a helpful list."""
    try:
        return PRESETS[name]
    except KeyError as err:
        known = ", ".join(PRESETS.keys())
        raise KeyError(f"unknown preset {name!r}; available: {known}") from err
