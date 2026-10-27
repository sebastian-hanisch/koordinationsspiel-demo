"""Die sechs Experimente: Eingaben fest, Ergebnis ein Wörterbuch aus Zeilen. Die Zahlen in README und App kommen von hier (tests/test_claims.py rechnet sie nach).

E1 Zufallsspiele und Start ohne Absprache, E2 Einzugsbereich gegen p*, E3 Logit gegen Gibbs-Maß, E4 Kampf der Geschlechter, E5 Schwellen, E6 Anti-Koordination."""

from __future__ import annotations

from fractions import Fraction
from math import comb

import numpy as np

import kos_constants as C
import kos_dynamics as D
import kos_games as G
import kos_logit as L
import kos_scenario as S
import kos_threshold as T

# --------------------------------------------------------------------------------------------------------------------------------------- E1

E1_START_GAMES = ((5, 2, 1, 2), (6, 4, 0, 4), (6, 4, 2, 4), (8, 4, 0, 4), (6, 4, 3, 4))
E1_START_N = 50


def random_game_stats(seed: int, draws: int, mode: str) -> dict:
    """Ein Strom: unter den Koordinationsspielen mit a != b der Anteil, in dem das auszahlungsdominante Gleichgewicht den kleineren Einzugsbereich hat (= risikodominiert),
    und sein mittlerer Einzugsbereich. Einzugsbereich von 'alle bündeln' ist 1 - p*, von 'alle direkt' p*."""
    a, b, c, d = S.draw_games(seed, draws, mode)
    m = S.coordination_subset(a, b, c, d)
    a, b, c, d = a[m], b[m], c[m], d[m]
    p = (b - c) / ((a - d) + (b - c))
    basin = np.where(a > b, 1 - p, p)
    return {"games": int(m.sum()), "risk_dominated": float(np.mean(basin < 0.5)), "mean_basin": float(np.mean(basin))}


def e1_random_games(draws: int = C.RANDOM_GAME_DRAWS, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """E1a: je Modus Mittel und Standardabweichung (über die Ströme) von Anteil und mittlerem Einzugsbereich."""
    rows = []
    for mode in S.MODES:
        stats = [random_game_stats(s, draws, mode) for s in seeds]
        rows.append({
            "mode": mode, "draws": draws, "seeds": len(seeds), "games": float(np.mean([s["games"] for s in stats])),
            "risk_dominated": float(np.mean([s["risk_dominated"] for s in stats])), "risk_dominated_std": float(np.std([s["risk_dominated"] for s in stats])),
            "mean_basin": float(np.mean([s["mean_basin"] for s in stats])), "mean_basin_std": float(np.std([s["mean_basin"] for s in stats])),
        })
    return rows


def start_without_agreement(n: int, start_percent: int, game) -> Fraction:
    """Exakte Wahrscheinlichkeit, dass die beste Antwort bei unabhängigem Start (jeder bündelt mit Wahrscheinlichkeit start_percent/100) bei 'alle bündeln' endet:
    Summe über den Startzustand k (Binomial) mal Absorptionswahrscheinlichkeit aus k."""
    h = D.br_absorption_probabilities(n, game)
    p = Fraction(start_percent, 100)
    return sum(Fraction(comb(n, k)) * p ** k * (1 - p) ** (n - k) * h[k] for k in range(n + 1))


def e1_start_without_agreement(games=E1_START_GAMES, n: int = E1_START_N, start_percent: int = 50) -> list[dict]:
    rows = []
    for game in games:
        ps = G.p_star(*game)
        rows.append({"game": game, "p_star": float(ps), "payoff_dominant": G.payoff_dominant(*game), "risk_dominant": G.risk_dominant(*game),
                     "all_s": float(start_without_agreement(n, start_percent, game)), "n": n, "start": start_percent})
    return rows


# --------------------------------------------------------------------------------------------------------------------------------------- E2

E2_GAMES = ((5, 2, 1, 2), (4, 3, 1, 2), (6, 4, 0, 4), (3, 1, 0, 0))
E2_N = 100


def br_boundary(n: int, game) -> Fraction:
    """Kleinster Start k0, von dem die beste Antwort mit Wahrscheinlichkeit mindestens 1/2 bei 'alle bündeln' endet, als Anteil k0/n (exakt, aus der Absorptionswahrscheinlichkeit)."""
    h = D.br_absorption_probabilities(n, game)
    k0 = next(k for k in range(n + 1) if h[k] >= Fraction(1, 2))
    return Fraction(k0, n)


def e2_basins(games=E2_GAMES, n: int = E2_N) -> list[dict]:
    rows = []
    for game in games:
        rows.append({"game": game, "p_star": float(G.p_star(*game)), "replicator": D.basin_boundary(game), "best_response": float(br_boundary(n, game)), "n": n})
    return rows


# --------------------------------------------------------------------------------------------------------------------------------------- E3

E3_GAMES = ((5, 2, 1, 2), (6, 4, 0, 4), (4, 3, 1, 2), (-1, 2, 4, 0))


def e3_logit_vs_gibbs(games=E3_GAMES) -> list[dict]:
    """Größte Abweichung |pi_Kette - pi_Gibbs| über das Gitter n x beta je Spiel (die Kette rechnet über Übergangswahrscheinlichkeiten, das Gibbs-Maß über das Potential)."""
    rows = []
    for game in games:
        worst = 0.0
        for n in C.LOGIT_GRID_N:
            for beta in C.LOGIT_GRID_BETA:
                worst = max(worst, float(np.abs(L.stationary(n, game, beta) - L.gibbs(n, game, beta)).max()))
        rows.append({"game": game, "worst": worst, "cells": len(C.LOGIT_GRID_N) * len(C.LOGIT_GRID_BETA)})
    return rows


def e3_masses(n: int, game, betas=(0.5, 1.0, 2.0)) -> list[dict]:
    """Masse bei höchstens 10 % und bei mindestens 90 % Bündelnden für einige beta."""
    rows = []
    for beta in betas:
        pi = L.stationary(n, game, beta)
        rows.append({"beta": beta, "low": L.mass_in_range(pi, 0.0, 0.1), "high": L.mass_in_range(pi, 0.9, 1.0), "mean": L.mean_share(pi)})
    return rows


# --------------------------------------------------------------------------------------------------------------------------------------- E4

E4_ROLES = ((2, 2), (3, 2), (4, 4))


def two_role_shares(role1: int, role2: int, grid: int = C.TWO_ROLE_GRID) -> dict:
    """Replikatordynamik von grid x grid Anfangspunkten: Anteil der Ziele (0,0) unter den erreichten Zielen und Anteil, der nirgends ankommt.
    Die Zeilenrolle startet bei (i + 1/2)/grid, die Spaltenrolle bei (j + 1/4)/grid: im symmetrischen Spiel liegt die Trennlinie der Einzugsbereiche auf p + q = 1,
    und ein Gitter mit Mittelpunkten träfe sie 50-mal (dort entscheidet Rundungsrauschen)."""
    A, B = S.role_matrices(role1, role2)
    gp = (np.arange(grid) + 0.5) / grid
    gq = (np.arange(grid) + 0.25) / grid
    z0 = np.stack(np.meshgrid(gp, gq, indexing="ij"), -1)
    goal = D.two_role_goal(D.two_role_replicator(z0, A, B, C.TWO_ROLE_T, C.TWO_ROLE_STEPS))
    reached = goal >= 0
    p, q = D.two_role_mixed_point(A, B)
    return {"roles": (role1, role2), "share_00": float((goal == 0).sum() / max(1, reached.sum())), "unreached": float(1 - reached.mean()),
            "p_star": float(p), "q_star": float(q), "direct_00": float((1 - p) * (1 - q)), "direct_11": float(p * q)}


def two_role_br_share(role1: int, role2: int, runs: int, seed: int, n_each: int = 30) -> dict:
    """Sequentielle beste Antwort mit n_each Spielern je Rolle, Start unabhängig 1/2 je Spieler: Anteil der Läufe, die bei (0,0) enden, unter den Läufen, die in einem der beiden Ziele enden."""
    from kos_rng import SplitMix64
    A, B = S.role_matrices(role1, role2)
    rng = SplitMix64(seed)
    at00 = at11 = 0
    for _ in range(runs):
        kr0, kc0 = S.start_count(n_each, 50, rng), S.start_count(n_each, 50, rng)
        kr, kc, _ = D.two_role_br_run(n_each, n_each, kr0, kc0, A, B, rng)
        at00 += int(kr == n_each and kc == n_each)
        at11 += int(kr == 0 and kc == 0)
    return {"roles": (role1, role2), "at00": at00, "at11": at11, "share_00": at00 / max(1, at00 + at11), "runs": runs}


def e4_two_roles(roles=E4_ROLES, runs: int = 200, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for r1, r2 in roles:
        row = two_role_shares(r1, r2)
        br = [two_role_br_share(r1, r2, runs, s)["share_00"] for s in seeds]
        row.update({"br_share_00": float(np.mean(br)), "br_std": float(np.std(br)), "br_runs": runs * len(seeds)})
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------------------------------------------------- E5

E5_SIGMAS = (5, 10, 15, 20, 30, 50)
E5_CELLS = ((30, 0), (50, 0), (50, 10))      # (mu, s0) in Prozent
E5_DRAWS = 25


def threshold_cell(mu: int, sigma: int, s0: int, draws: int, seed: int, n: int = C.N_AGENTS) -> dict:
    """Endanteile von `draws` Ziehungen aus n Personen mit Normalschwellen (Mittel mu %, Streuung sigma %), s0 % machen von sich aus mit; dazu der Fixpunkt der Dichte-Fassung."""
    z = S.threshold_normals(seed, draws, n)
    finals = np.array([T.agents_final(T.draw_thresholds(mu / 100, sigma / 100, z[i]), s0) for i in range(draws)])
    return {"mu": mu, "sigma": sigma, "s0": s0, "mean": float(finals.mean()), "std": float(finals.std()), "finals": finals,
            "fixed_point": T.fixed_point_final(mu / 100, sigma / 100, s0 / 100)}


def e5_thresholds(cells=E5_CELLS, sigmas=E5_SIGMAS, draws: int = E5_DRAWS, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for mu, s0 in cells:
        for sigma in sigmas:
            per_seed = [threshold_cell(mu, sigma, s0, draws, s) for s in seeds]
            rows.append({"mu": mu, "sigma": sigma, "s0": s0, "mean": float(np.mean([c["mean"] for c in per_seed])), "seed_std": float(np.std([c["mean"] for c in per_seed])),
                         "inner_std": float(np.mean([c["std"] for c in per_seed])), "fixed_point": per_seed[0]["fixed_point"], "draws": draws * len(seeds)})
    return rows


def classic_example() -> dict:
    """Granovetters Beispiel mit 100 Personen: Schwellen 0, 1, ..., 99 (je eine Person) gegen dieselben mit einer Person von Schwelle 1 auf Schwelle 2 gesetzt."""
    full = list(range(100))
    changed = list(range(100))
    changed[1] = 2
    return {"full": T.count_final(full), "changed": T.count_final(changed), "mean_full": sum(full) / 100, "mean_changed": sum(changed) / 100}


# --------------------------------------------------------------------------------------------------------------------------------------- E6

E6_GAMES = ((5, 2, 1, 2), (6, 4, 0, 4), (-1, 2, 4, 0), (0, 3, 4, 2))


def anti_row(game, grid: int = C.ANTI_GRID) -> dict:
    """Stabilität des gemischten Gleichgewichts: Replikator von grid gleichverteilten Anfangsanteilen; Anteil der Läufe nahe p* (Abstand < 0,01) gegen Anteil an einem Rand."""
    ps = G.p_star(*game)
    x0 = (np.arange(grid) + 0.5) / grid
    xf = D.replicator_final(x0, game, C.ANTI_T, C.ANTI_STEPS)
    near = float(np.mean(np.abs(xf - float(ps)) < 0.01)) if ps is not None else 0.0
    return {"game": game, "class": G.classify(*game), "slope": G.slope(*game), "p_star": None if ps is None else float(ps), "near": near,
            "edge": float(np.mean((xf < 0.01) | (xf > 0.99)))}


def e6_anti(games=E6_GAMES) -> list[dict]:
    return [anti_row(g) for g in games]


def anti_welfare(game) -> dict:
    """Mittlere Auszahlung je Spieler im gemischten Gleichgewicht gegen die beste Absprache (Brüche)."""
    return {"mixed": G.mixed_welfare(*game), "agreed": G.best_agreed_welfare(*game)}
