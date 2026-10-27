"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Erwartungswerte stehen als Rechnung dabei."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Spiele (a, b, c, d): a beide bündeln, b beide direkt, c ich bündele und der andere fährt direkt, d ich fahre direkt und der andere bündelt.
HIRSCH = (5, 2, 1, 2)       # a - d = 3 > b - c = 1: p* = 1/(3 + 1) = 1/4, Bündeln risikodominant, Einzugsbereich von 'alle bündeln' 3/4
BUENDELN = (6, 4, 0, 4)     # a - d = 2 < b - c = 4: p* = 4/(2 + 4) = 2/3; a > b (Bündeln auszahlungsdominant), aber b - c > a - d (Direkt risikodominant); Phi(S,S) = 2, Phi(H,H) = 4
GLEICHAUF = (6, 4, 2, 4)    # a - d = 2 = b - c: p* = 2/4 = 1/2, kein Gleichgewicht risikodominant
FEIGLING = (-1, 2, 4, 0)    # V = 4, C = 6: a = (V - C)/2 = -1, c = V = 4, d = 0, b = V/2 = 2; s = (-1 - 0) + (2 - 4) = -3, x* = (2 - 4)/(-3) = 2/3
DOMINANT = (3, 1, 2, 0)     # a - d = 3 > 0, b - c = -1 < 0: Bündeln dominiert, genau ein Gleichgewicht
ENTARTET = (3, 2, 1, 3)     # a = d: Gleichstand, entartet


@pytest.fixture
def games():
    return {"hirsch": HIRSCH, "buendeln": BUENDELN, "gleichauf": GLEICHAUF, "feigling": FEIGLING, "dominant": DOMINANT, "entartet": ENTARTET}


@pytest.fixture
def bos():
    """Kampf der Geschlechter: A = [[2, 0], [0, 1]] (Zeile will Option 0), B = [[1, 0], [0, 2]] (Spalte will Option 1). Reine Gleichgewichte (0,0) und (1,1);
    gemischt: p* = P(Zeile spielt 0) macht die Spalte indifferent: p·1 + (1 - p)·0 = p·0 + (1 - p)·2, also p = 2 - 2p, p = 2/3; q* = 1/3 (Zeile: 2q = 1 - q)."""
    return [[2, 0], [0, 1]], [[1, 0], [0, 2]], (Fraction(2, 3), Fraction(1, 3))
