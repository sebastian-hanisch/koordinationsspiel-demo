"""Granovetters Schwellenmodell im vollständig durchmischten Fall: Person i macht mit, wenn mindestens so viele andere mitmachen, wie ihre Schwelle verlangt.

Schwellen werden in Personen gezählt (`count_final`) oder als Anteile gezogen (`agents_final`, Schwelle theta·n Personen). Der Vergleich 'Zahl der Mitmachenden >= Schwelle'
läuft über ganze Zahlen und die Schwelle selbst, nicht über Anteile: 10 von 30 ist genau 1/3, in Gleitkomma aber nicht (3·(10/30) - 1 ist nicht 0).
Die Verteilung der Schwellen im Kontinuum: Endanteil als Fixpunkt von F(x) = Phi((x - mu)/sigma), gerechnet mit math.erf (kein scipy zur Laufzeit)."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def count_final(thresholds, n0: int = 0) -> int:
    """Endzahl der Mitmachenden. Schwellen in Personen (reelle Zahlen erlaubt); n0 Personen machen von sich aus mit. Iteration n -> max(n0, #{Schwelle <= n}) bis zum Stillstand
    (steigt monoton, endet nach höchstens n Schritten)."""
    thr = np.sort(np.asarray(thresholds, dtype=float))
    n = int(n0)
    while True:
        new = max(int(n0), int(np.searchsorted(thr, n, side="right")))
        if new <= n:
            return n
        n = new


def agents_final(theta, s0_percent: int = 0) -> float:
    """Endanteil bei n Personen mit Schwellen theta (Anteile der Gesamtheit); s0_percent Prozent machen von sich aus mit (n0 = s0_percent·n/100, abgerundet)."""
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    persons = np.round(theta * n, 9)          # 0,07·100 ist in Gleitkomma 7,000000000000001: auf 9 Stellen runden, damit eine Schwelle von genau 7 Personen auch bei 7 Mitmachenden greift
    return count_final(persons, (s0_percent * n) // 100) / n


def fixed_point_final(mu: float, sigma: float, s0: float = 0.0, tol: float = 1e-12, max_iter: int = 200000) -> float:
    """Endanteil der Dichte-Fassung: x -> max(s0, F(x)), F(x) = Phi((x - mu)/sigma), vom Start x = s0 aus bis zum Stillstand."""
    x = s0
    for _ in range(max_iter):
        new = max(s0, normal_cdf(x, mu, sigma))
        if new <= x + tol:
            return x
        x = new
    return x


def count_stable_fixed(mu: float, sigma: float, grid: int = 20001) -> int:
    """Zahl der stabilen Fixpunkte von F(x) = Phi((x - mu)/sigma) auf [0, 1]: Stellen auf dem Gitter, an denen F(x) - x von nicht negativ auf nicht positiv wechselt
    (F schneidet die Diagonale von oben; ein Wert genau 0 zählt einmal). F ist positiv und bleibt unter 1, also kommt kein Randpunkt dazu: es sind 1 oder 2
    (bei drei Schnittpunkten zwei stabile, dazwischen ein instabiler). Schnittpunkte enger als das Gitter (5·10^-5) würden übersehen."""
    xs = np.linspace(0.0, 1.0, grid)
    g = np.array([normal_cdf(float(x), mu, sigma) for x in xs]) - xs
    start_ok = np.concatenate(([g[0] >= 0], g[1:-1] > 0))
    return int(np.sum(start_ok & (g[1:] <= 0)))


def standard_normals(seed: int, count: int) -> np.ndarray:
    """count Standardnormalzahlen aus dem SplitMix64-Strom (Inversion der Normalverteilung auf offenem Intervall)."""
    from kos_rng import uniform_block
    u = uniform_block(seed, count) + 2.0 ** -54          # [0, 1) -> (0, 1), exakt darstellbar
    inv = NormalDist().inv_cdf
    return np.array([inv(float(v)) for v in u])


def draw_thresholds(mu: float, sigma: float, z: np.ndarray) -> np.ndarray:
    """Schwellen mu + sigma·z, bei 0 gekappt (wer ohne andere mitmacht, hat Schwelle 0). Gleiche z für verschiedene sigma: gemeinsame Zufallszahlen."""
    return np.maximum(mu + sigma * z, 0.0)
