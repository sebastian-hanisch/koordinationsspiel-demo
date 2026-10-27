"""Szenario: Zufallsspiele (E1), Anfangszustände der Läufe und Schwellen. Aller Zufall kommt aus dem SplitMix64-Strom (kos_rng.py): gleiche Eingabe, gleiche Zahlen auf jeder Plattform."""

from __future__ import annotations

import numpy as np

from kos_rng import SplitMix64, uniform_block

MODE_GENERAL = "allgemein"
MODE_SAFE_H = "sicherer Hase"
MODES = (MODE_GENERAL, MODE_SAFE_H)


def draw_games(seed: int, draws: int, mode: str = MODE_GENERAL):
    """`draws` Spiele (a, b, c, d) mit unabhängigen Auszahlungen gleichverteilt auf [0, 10). Modus 'sicherer Hase': d := b (direkt fahren zahlt unabhängig vom anderen).
    Gibt vier Felder der Länge `draws`."""
    if mode not in MODES:
        raise ValueError(f"unbekannter Modus: {mode}")
    u = uniform_block(seed, 4 * draws).reshape(draws, 4) * 10.0
    a, b, c, d = u[:, 0], u[:, 1], u[:, 2], u[:, 3]
    if mode == MODE_SAFE_H:
        d = b.copy()
    return a, b, c, d


def coordination_subset(a, b, c, d):
    """Maske der Koordinationsspiele mit eindeutig auszahlungsdominantem Gleichgewicht: a > d, b > c, a != b."""
    return (a > d) & (b > c) & (a != b)


def start_count(n: int, start_percent: int, rng: SplitMix64) -> int:
    """Zahl der S-Spieler beim Start ohne Absprache: jeder der n Spieler wählt unabhängig S mit Wahrscheinlichkeit start_percent/100.
    Vergleich ganzzahlig: Zufallszahl auf 10^6 Stufen gegen start_percent·10^4."""
    return sum(1 for _ in range(n) if int(rng.uniform() * 1_000_000) < start_percent * 10_000)


def role_matrices(role1: int, role2: int):
    """Kampf der Geschlechter: Rolle 1 zieht Option 0 vor (Wert role1 gegen 1), Rolle 2 zieht Option 1 vor (Wert role2 gegen 1); keine Einigung zahlt 0.
    A (Zeilenrolle) = [[role1, 0], [0, 1]], B (Spaltenrolle) = [[1, 0], [0, role2]]."""
    return [[role1, 0], [0, 1]], [[1, 0], [0, role2]]


def threshold_normals(seed: int, draws: int, n: int) -> np.ndarray:
    """Feld (draws, n) von Standardnormalzahlen; die Schwellen für verschiedene sigma entstehen aus denselben Zahlen (gemeinsame Zufallszahlen)."""
    from kos_threshold import standard_normals
    return standard_normals(seed, draws * n).reshape(draws, n)
