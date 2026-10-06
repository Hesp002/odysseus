import json

import routes.omarchy_theme_routes as omarchy_theme_routes

PALETTE = {"mode": "dark", "bg": "#131515", "fg": "#F1E4C2", "panel": "#101212",
           "border": "#3b3a34", "red": "#ef944d"}


def test_loads_valid_palette_lowercased(tmp_path):
    path = tmp_path / "omarchy-theme.json"
    path.write_text(json.dumps(PALETTE), encoding="utf-8")
    assert omarchy_theme_routes.load_omarchy_theme(str(path)) == {
        "bg": "#131515", "fg": "#f1e4c2", "panel": "#101212",
        "border": "#3b3a34", "red": "#ef944d",
    }


def test_missing_file_is_unavailable(tmp_path):
    assert omarchy_theme_routes.load_omarchy_theme(str(tmp_path / "nope.json")) is None


def test_rejects_unrendered_or_bad_colors(tmp_path):
    path = tmp_path / "omarchy-theme.json"
    for bad in ({**PALETTE, "red": "{{ accent }}"}, {**PALETTE, "bg": "red"},
                {k: v for k, v in PALETTE.items() if k != "border"}, ["#131515"]):
        path.write_text(json.dumps(bad), encoding="utf-8")
        assert omarchy_theme_routes.load_omarchy_theme(str(path)) is None
    path.write_text("{not json", encoding="utf-8")
    assert omarchy_theme_routes.load_omarchy_theme(str(path)) is None
