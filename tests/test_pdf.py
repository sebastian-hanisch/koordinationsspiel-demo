"""PDF-Export: gültige Datei für jede Spielklasse, Kernschriften kennen nur Latin-1 (Sonderzeichen werden ersetzt)."""

import pytest

import kos_evaluation as V
from kos_pdf_export import _latin, build_pdf


@pytest.mark.parametrize("game", [(6, 4, 0, 4), (-1, 2, 4, 0), (3, 1, 2, 0), (3, 2, 1, 3), (0, 0, 0, 0)])
def test_pdf_is_built_for_every_class(game):
    s = V.Settings(*game)
    gv = V.analyse_game(game)
    data = build_pdf(s, gv, [V.verdict_game(gv)])
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_replaces_special_characters_and_markdown():
    assert _latin("β = 2 → „x“ ≥ 2 · 3 – 4 **fett** `c` Φ σ μ ≠ ×") == 'beta = 2 -> "x" >= 2 * 3 - 4 fett c Phi sigma mu != x'
    assert _latin("Größe ä ö ü ß") == "Größe ä ö ü ß"
