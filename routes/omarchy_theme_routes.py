"""Omarchy desktop theme bridge.

An Omarchy theme-set hook on the host writes the current palette to
DATA_DIR/omarchy-theme.json. The frontend's "omarchy" theme swatch only
appears when this endpoint reports a valid palette, so deployments without
the hook see nothing new.
"""
import json
import os
import re

from fastapi import APIRouter

from src.constants import DATA_DIR

OMARCHY_THEME_FILE = os.path.join(DATA_DIR, "omarchy-theme.json")
_COLOR_KEYS = ("bg", "fg", "panel", "border", "red")
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def load_omarchy_theme(path: str | None = None) -> dict | None:
    """Return the validated palette, or None if missing or malformed."""
    try:
        with open(path or OMARCHY_THEME_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    colors = {}
    for key in _COLOR_KEYS:
        value = data.get(key)
        if not isinstance(value, str) or not _HEX.match(value):
            return None
        colors[key] = value.lower()
    return colors


def setup_omarchy_theme_routes():
    router = APIRouter(prefix="/api/omarchy-theme", tags=["preferences"])

    @router.get("")
    async def get_omarchy_theme():
        colors = load_omarchy_theme()
        return {"available": colors is not None, "colors": colors}

    return router
