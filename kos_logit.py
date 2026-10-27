"""Logit-Lernen (log-lineares Lernen) im vollständig durchmischten Spiel: stationäre Verteilung der Kette und das Gibbs-Maß des Potentials.

Ein zufällig gezogener Spieler wählt S mit Wahrscheinlichkeit sigma(beta·(u_S - u_H)); u sind die mittleren Auszahlungen gegen die n - 1 anderen (Summe geteilt durch n - 1),
so dass beta nicht von der Größe n abhängt. Zustand ist die Zahl k der S-Spieler. `stationary` rechnet die Geburts-Sterbe-Kette im Logarithmus, `gibbs` die geschlossene Formel
pi(k) ~ C(n, k)·exp(beta·Phi(k)); beides wird in tests/test_oracle_koordinationsspiel.py gegen den Eigenvektor der vollen Übergangsmatrix gehalten."""

from __future__ import annotations

import math

import numpy as np


def sigma_log(x: float) -> float:
    """log sigma(x) = -log(1 + exp(-x)), numerisch stabil für große |x|."""
    return -float(np.logaddexp(0.0, -x))


def advantage(n: int, k: int, game) -> float:
    """u_S - u_H eines Spielers, gegen den k der n - 1 anderen S spielen (Mittel über die anderen)."""
    a, b, c, d = game
    return (k * (a - d) - (n - 1 - k) * (b - c)) / (n - 1)


def stationary(n: int, game, beta: float) -> np.ndarray:
    """Stationäre Verteilung pi(0..n) der Logit-Kette. Aufwärts von k: (n - k)/n · sigma(beta·Delta_k), abwärts von k + 1: (k + 1)/n · sigma(-beta·Delta_k);
    Delta_k = advantage(n, k, game). Detailed Balance (Geburts-Sterbe-Kette), im Logarithmus aufsummiert, damit n = 200 und beta = 10 nicht über- oder unterlaufen."""
    logp = np.zeros(n + 1)
    for k in range(n):
        delta = advantage(n, k, game)
        up = math.log((n - k) / n) + sigma_log(beta * delta)
        down = math.log((k + 1) / n) + sigma_log(-beta * delta)
        logp[k + 1] = logp[k] + up - down
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def potential_of_state(n: int, k: int, game) -> float:
    """Phi(k) = [(a - d)·k(k - 1)/2 + (b - c)·(n - k)(n - k - 1)/2] / (n - 1): das Potential, wenn k Spieler S und n - k Spieler H spielen (alle Paare gezählt)."""
    a, b, c, d = game
    return ((a - d) * k * (k - 1) / 2 + (b - c) * (n - k) * (n - k - 1) / 2) / (n - 1)


def gibbs(n: int, game, beta: float) -> np.ndarray:
    """Gibbs-Maß pi(k) ~ C(n, k)·exp(beta·Phi(k)) (Binomialkoeffizient über lgamma)."""
    k = np.arange(n + 1)
    phi = np.array([potential_of_state(n, int(i), game) for i in k])
    logc = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
    logp = logc + beta * phi
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def mass_in_range(pi: np.ndarray, lo_share: float, hi_share: float) -> float:
    """Masse der Zustände, in denen der Anteil k/n der S-Spieler zwischen lo_share und hi_share liegt (beide eingeschlossen; Vergleich ganzzahlig auf k·1000)."""
    n = len(pi) - 1
    lo, hi = round(lo_share * 1000), round(hi_share * 1000)
    return float(sum(p for k, p in enumerate(pi) if lo <= k * 1000 / n <= hi))


def mean_share(pi: np.ndarray) -> float:
    """Erwarteter Anteil der S-Spieler unter pi."""
    n = len(pi) - 1
    return float(np.dot(pi, np.arange(n + 1)) / n)


def logit_run(n: int, k0: int, game, beta: float, rng, steps: int, record: bool = False) -> tuple[int, tuple]:
    """Simulation der Logit-Kette: je Schritt ein Spieler (Zahl 1 des Stroms), seine Wahl mit einer zweiten Zahl. Gibt (Endzustand, Pfad). Der Pfad hält k nach jedem Schritt."""
    k = k0
    path = [k] if record else []
    for _ in range(steps):
        i = int(rng.uniform() * n)
        is_s = i < k
        others = k - 1 if is_s else k
        p_s = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, beta * advantage(n, others, game)))))
        choose_s = rng.uniform() < p_s
        k += int(choose_s) - int(is_s)
        if record:
            path.append(k)
    return k, tuple(path)
