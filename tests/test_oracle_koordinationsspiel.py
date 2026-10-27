"""Orakel-Tests koordinationsspiel-demo: jede Kernroutine gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2 'Spieltheorie').

1. Gleichgewichte: Support-Enumeration für allgemeine Bimatrix-Spiele (hier im Test, mit Brüchen, ohne p*) gegen `enumerate_ne` und gegen die Formel p* / Klasse.
2. Replikator: scipy solve_ivp (DOP853, enge Toleranz); Grenze des Einzugsbereichs gegen die Formel p*.
3. Logit-Kette: stationäre Verteilung der vollen (n+1)x(n+1)-Übergangsmatrix nach Grassmann-Taksar-Heyman (numpy) gegen `stationary` und `gibbs`; kleine Simulation.
4. Beste Antwort: Absorptionswahrscheinlichkeit gegen die lineare Lösung der vollen Kette (numpy) und gegen eine Simulation mit einzelnen Spielern (numpy-Zufall, anderer Strom).
5. Schwellenmodell: Personen mit Quantil-Schwellen gegen den Fixpunkt; Zahl stabiler Fixpunkte gegen Iterationen von vielen Startwerten.
6. Zwei Rollen: Replikator gegen scipy, gemischter Punkt gegen die Aufzählung.
7. Zufallsspiele (E1): vektorisierte Auswertung gegen Spiel-für-Spiel in Brüchen.
Die große Fassung (40 000 Spiele, größere Gitter) lief im Scratch-Ordner, hier die kleine, schnelle (< 10 s)."""

import itertools
import statistics
from fractions import Fraction as F

import numpy as np
import pytest

import kos_dynamics as D
import kos_experiments as X
import kos_games as G
import kos_logit as L
import kos_scenario as S
import kos_threshold as T
from kos_rng import SplitMix64

# ------------------------------------------------------------------------------------------------------------ 1. Gleichgewichte


def _solve(M, rhs):
    """Gauß-Elimination mit Brüchen; None bei singulärer Matrix."""
    n = len(M)
    aug = [list(map(F, row)) + [F(r)] for row, r in zip(M, rhs)]
    for col in range(n):
        piv = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if piv is None:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        aug[col] = [v / aug[col][col] for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [v - f * w for v, w in zip(aug[r], aug[col])]
    return [aug[r][n] for r in range(n)]


def oracle_support_enumeration(A, B):
    """Alle Gleichgewichte eines nicht entarteten m x n Bimatrix-Spiels mit Trägern gleicher Größe (Standardverfahren): (x, y) als Bruchlisten."""
    m, n = len(A), len(A[0])
    found = []
    for s in range(1, min(m, n) + 1):
        for I in itertools.combinations(range(m), s):
            for J in itertools.combinations(range(n), s):
                # y auf J: Zeilenspieler indifferent über I: sum_j A[i][j] y_j = v für i in I, sum y = 1
                My = [[A[i][j] for j in J] + [-1] for i in I] + [[1] * s + [0]]
                sol = _solve(My, [0] * s + [1])
                Mx = [[B[i][j] for i in I] + [-1] for j in J] + [[1] * s + [0]]
                solx = _solve(Mx, [0] * s + [1])
                if sol is None or solx is None:
                    continue
                y_part, v = sol[:s], sol[s]
                x_part, w = solx[:s], solx[s]
                if any(t <= 0 for t in y_part) or any(t <= 0 for t in x_part):
                    continue
                x = [F(0)] * m
                y = [F(0)] * n
                for i, t in zip(I, x_part):
                    x[i] = t
                for j, t in zip(J, y_part):
                    y[j] = t
                row_pay = [sum(A[i][j] * y[j] for j in range(n)) for i in range(m)]
                col_pay = [sum(B[i][j] * x[i] for i in range(m)) for j in range(n)]
                if max(row_pay) == v and max(col_pay) == w:
                    found.append((tuple(x), tuple(y)))
    return found


def _random_int_games(seed, count, low=0, high=9):
    rng = np.random.default_rng(seed)
    return [tuple(int(v) for v in rng.integers(low, high + 1, 4)) for _ in range(count)]


def test_oracle_handbeispiel_support_enumeration_gives_bos_equilibria(bos):
    """Das Orakel selbst an der Handrechnung: Kampf der Geschlechter hat reine (1,0)/(1,0), (0,1)/(0,1) und gemischt (2/3,1/3)/(1/3,2/3)."""
    A, B, _ = bos
    found = oracle_support_enumeration(A, B)
    assert sorted(found) == sorted([((F(1), F(0)), (F(1), F(0))), ((F(0), F(1)), (F(0), F(1))), ((F(2, 3), F(1, 3)), (F(1, 3), F(2, 3)))])


def test_enumeration_matches_oracle_and_formula_on_random_games():
    counts = {G.KOORDINATION: 0, G.ANTI: 0, G.DOMINANT: 0, G.ENTARTET: 0}
    for game in _random_int_games(11, 1500):
        a, b, c, d = game
        A, B = G.payoff_matrices(*game)
        cls = G.classify(*game)
        counts[cls] += 1
        mine = G.enumerate_ne(A, B)
        if cls == G.ENTARTET:
            assert G.is_degenerate(A, B)
            for p, q in mine:                                     # jede gemeldete reine Strategie ist ein Gleichgewicht (schwache Ungleichung nachgerechnet)
                i, j = int(1 - p), int(1 - q)
                assert A[i][j] >= A[1 - i][j] and B[i][j] >= B[i][1 - j]
            continue
        assert not G.is_degenerate(A, B)
        oracle = [(x[0], y[0]) for x, y in oracle_support_enumeration(A, B)]          # P(Option 0) je Seite
        assert sorted(mine) == sorted(oracle), game
        assert len(mine) == (1 if cls == G.DOMINANT else 3), game
        if cls in (G.KOORDINATION, G.ANTI):
            ps = G.p_star(*game)
            assert (ps, ps) in mine and 0 < ps < 1, game
    assert min(counts[k] for k in (G.KOORDINATION, G.ANTI, G.DOMINANT, G.ENTARTET)) > 100       # alle vier Zweige wurden getroffen


def test_no_even_number_of_equilibria_in_generic_games():
    """Nicht entartete Bimatrix-Spiele haben eine ungerade Zahl von Gleichgewichten (Wilson); allgemeine 2x2-Spiele ohne Symmetrie."""
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(800):
        A = rng.integers(0, 10, (2, 2)).tolist()
        B = rng.integers(0, 10, (2, 2)).tolist()
        if G.is_degenerate(A, B) or (A[0][0] - A[1][0]) - (A[0][1] - A[1][1]) == 0 or (B[0][0] - B[0][1]) - (B[1][0] - B[1][1]) == 0:
            continue
        checked += 1
        assert len(G.enumerate_ne(A, B)) % 2 == 1
        assert len(G.enumerate_ne(A, B)) == len(oracle_support_enumeration(A, B))
    assert checked > 400


# ------------------------------------------------------------------------------------------------------------ 2. Replikator


def test_replicator_matches_scipy_solve_ivp(games):
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp
    for name in ("hirsch", "buendeln", "feigling", "gleichauf"):
        g = games[name]
        for x0 in (0.05, 0.3, 0.5, 0.7, 0.95):
            sol = solve_ivp(lambda t, x: D.replicator_rhs(x, g), (0, 8), [x0], method="DOP853", rtol=1e-12, atol=1e-14, t_eval=[8.0])
            mine = D.replicator_path(x0, g, 8.0, 800)[-1]
            assert mine == pytest.approx(sol.y[0, 0], abs=1e-7), (name, x0)


def test_basin_boundary_matches_formula_and_scipy_bisection(games):
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp
    g = games["buendeln"]
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = (lo + hi) / 2
        end = solve_ivp(lambda t, x: D.replicator_rhs(x, g), (0, 60), [mid], rtol=1e-10, atol=1e-12).y[0, -1]
        lo, hi = (mid, hi) if end < 0.5 else (lo, mid)
    assert (lo + hi) / 2 == pytest.approx(2 / 3, abs=1e-4)
    for name in ("hirsch", "gleichauf", "buendeln"):
        assert D.basin_boundary(games[name]) == pytest.approx(float(G.p_star(*games[name])), abs=1e-4)


# ------------------------------------------------------------------------------------------------------------ 3. Logit-Kette


def full_transition_matrix(n, game, beta):
    """Übergangsmatrix der Logit-Kette auf k = 0..n, aus der Wahl der Spieler hergeleitet (logistische Wahrscheinlichkeit, eigene Rechnung mit exp)."""
    a, b, c, d = game
    P = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        if k < n:        # ein H-Spieler (andere S: k) wird gezogen und wählt S
            diff = (k * a + (n - 1 - k) * c - k * d - (n - 1 - k) * b) / (n - 1)
            P[k, k + 1] = (n - k) / n / (1 + np.exp(-beta * diff))
        if k > 0:        # ein S-Spieler (andere S: k - 1) wird gezogen und wählt H
            diff = ((k - 1) * a + (n - k) * c - (k - 1) * d - (n - k) * b) / (n - 1)
            P[k, k - 1] = k / n / (1 + np.exp(beta * diff))
        P[k, k] = 1 - P[k].sum()
    return P


def gth_stationary(P):
    """Stationäre Verteilung nach Grassmann-Taksar-Heyman (Zustandsreduktion ohne Subtraktion, daher auch bei Einträgen von 1e-14 neben 0,5 genau).
    np.linalg.eig und ein lineares System mit ersetzter Zeile waren hier ungenauer als der Prüfling (Fehler 5e-7 bei n = 25, obwohl das Spiel symmetrisch ist und pi es auch sein muss;
    ORACLE-PLAYBOOK §2: Die Referenz darf nicht ungenauer sein als die Toleranz)."""
    P = np.array(P, dtype=float)
    n1 = P.shape[0]
    for k in range(n1 - 1, 0, -1):
        s = P[k, :k].sum()
        P[:k, k] /= s
        P[:k, :k] += np.outer(P[:k, k], P[k, :k])
    pi = np.zeros(n1)
    pi[0] = 1.0
    for j in range(1, n1):
        pi[j] = pi[:j] @ P[:j, j]
    return pi / pi.sum()


def test_stationary_and_gibbs_match_gth_of_the_full_matrix(games):
    for name in ("hirsch", "buendeln", "gleichauf", "feigling"):
        for n in (2, 3, 8, 25):
            for beta in (0.1, 1.0, 3.0):
                ev = gth_stationary(full_transition_matrix(n, games[name], beta))
                assert np.abs(L.stationary(n, games[name], beta) - ev).max() < 1e-10, (name, n, beta)
                assert np.abs(L.gibbs(n, games[name], beta) - ev).max() < 1e-10, (name, n, beta)


def test_gibbs_equals_chain_on_random_games():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(300):
        game = tuple(int(v) for v in rng.integers(-9, 10, 4))
        n = int(rng.integers(2, 60))
        beta = float(rng.uniform(0.1, 10))
        worst = max(worst, float(np.abs(L.stationary(n, game, beta) - L.gibbs(n, game, beta)).max()))
    assert worst < 1e-12


def test_logit_simulation_visits_states_like_the_stationary_distribution(games):
    n, beta = 6, 1.0
    g = games["feigling"]
    pi = L.stationary(n, g, beta)
    rng = SplitMix64(9)
    k, counts = 3, np.zeros(n + 1)
    for _ in range(60000):
        k, _ = L.logit_run(n, k, g, beta, rng, 1)
        counts[k] += 1
    assert np.abs(counts / counts.sum() - pi).max() < 0.02


# ------------------------------------------------------------------------------------------------------------ 4. Beste Antwort


def numpy_absorption(n, game):
    """Wahrscheinlichkeit, bei k = n zu enden, aus der vollen Kette: (I - Q) h = r über die Zustände, die nicht festhängen (numpy, Gleitkomma)."""
    up, down = D.br_chain_probs(n, game)
    P = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        if up[k]:
            P[k, k + 1] = (n - k) / n
        if down[k]:
            P[k, k - 1] = k / n
    stuck = [k for k in range(n + 1) if P[k].sum() == 0]
    trans = [k for k in range(n + 1) if k not in stuck]
    h = np.zeros(n + 1)
    for k in stuck:
        h[k] = 1.0 if k == n else 0.0
    if trans:
        Q = P[np.ix_(trans, trans)]
        R = P[np.ix_(trans, stuck)] @ h[stuck]
        norm = P[trans].sum(axis=1)
        h[trans] = np.linalg.solve(np.diag(norm) - Q, R)
    return h


def test_absorption_matches_numpy_solution_of_the_full_chain(games):
    for name in ("hirsch", "buendeln", "gleichauf", "feigling", "dominant"):
        for n in (2, 3, 7, 20, 31):
            exact = D.br_absorption_probabilities(n, games[name])
            assert [float(v) for v in exact] == pytest.approx(numpy_absorption(n, games[name]).tolist(), abs=1e-9), (name, n)


def test_absorption_matches_numpy_on_random_games():
    rng = np.random.default_rng(8)
    for _ in range(150):
        game = tuple(int(v) for v in rng.integers(-6, 7, 4))
        n = int(rng.integers(2, 25))
        exact = D.br_absorption_probabilities(n, game)
        assert [float(v) for v in exact] == pytest.approx(numpy_absorption(n, game).tolist(), abs=1e-9), (game, n)


def individual_br_run(n, k0, game, rng):
    """Beste Antwort mit einzelnen Spielern (Feld), numpy-Zufall; Auszahlungen aus der Matrix gegen die tatsächlich gezählten anderen, ganzzahlig. Gibt die Endzahl."""
    a, b, c, d = game
    s = np.zeros(n, dtype=int)
    s[:k0] = 1
    for _ in range(5000):
        k = int(s.sum())
        # irgendein Spieler, der strikt wechseln will?
        movers = []
        for i in range(n):
            others = k - s[i]
            u_s = others * a + (n - 1 - others) * c
            u_h = others * d + (n - 1 - others) * b
            if (s[i] == 0 and u_s > u_h) or (s[i] == 1 and u_h > u_s):
                movers.append(i)
        if not movers:
            return k
        i = int(rng.integers(0, n))
        if i in movers:
            s[i] = 1 - s[i]
    return int(s.sum())


def test_br_chain_matches_simulation_with_individual_players():
    """Mit einzelnen Spielern und anderem Zufallsstrom: Anteil der Läufe, die bei 'alle bündeln' enden, gegen die exakte Absorptionswahrscheinlichkeit (3 Standardfehler)."""
    rng = np.random.default_rng(1)
    for game, n, k0 in (((6, 4, 2, 4), 9, 4), ((6, 4, 0, 4), 9, 6), ((5, 2, 1, 2), 9, 3), ((6, 4, 0, 4), 7, 5), ((6, 4, 0, 4), 8, 5)):
        exact = float(D.br_absorption_probabilities(n, game)[k0])
        runs = 400
        wins = sum(individual_br_run(n, k0, game, rng) == n for _ in range(runs))
        assert abs(wins / runs - exact) < 3 * np.sqrt(max(exact * (1 - exact), 0.01) / runs) + 0.01, (game, n, k0, exact, wins / runs)


def test_br_run_simulation_matches_exact_with_splitmix_stream():
    for game, n in (((6, 4, 2, 4), 50), ((6, 4, 0, 4), 50), ((6, 4, 3, 4), 50)):
        exact = float(X.start_without_agreement(n, 50, game))
        rng = SplitMix64(21)
        runs = 2000
        wins = 0
        for _ in range(runs):
            wins += D.br_run(n, S.start_count(n, 50, rng), game, rng).final == n
        assert abs(wins / runs - exact) < 3 * np.sqrt(max(exact * (1 - exact), 0.0025) / runs) + 0.005, (game, exact, wins / runs)


# ------------------------------------------------------------------------------------------------------------ 5. Schwellenmodell


def test_agents_with_quantile_thresholds_reach_the_fixed_point():
    """Sehr viele Personen mit Quantil-Schwellen mu + sigma·z_i (z_i = Normal-Quantile von (i + 0,5)/n) ersetzen die Dichte: Endanteil gleich dem Fixpunkt bis auf 1/n-Größe."""
    n = 20000
    inv = statistics.NormalDist().inv_cdf
    z = np.array([inv((i + 0.5) / n) for i in range(n)])
    for mu, sigma in ((0.3, 0.2), (0.3, 0.1), (0.5, 0.3), (0.5, 0.5), (0.4, 0.25)):
        theta = np.maximum(mu + sigma * z, 0.0)
        assert T.agents_final(theta, 0) == pytest.approx(T.fixed_point_final(mu, sigma, 0.0), abs=2e-3), (mu, sigma)
    for mu, sigma, s0 in ((0.5, 0.2, 10), (0.3, 0.1, 5)):
        theta = np.maximum(mu + sigma * z, 0.0)
        assert T.agents_final(theta, s0) == pytest.approx(T.fixed_point_final(mu, sigma, s0 / 100), abs=2e-3), (mu, sigma, s0)


def attractor_count(mu, sigma):
    """Stabile Fixpunkte als Grenzwerte der Iteration x -> F(x) von 81 Startwerten (scipy.special.ndtr statt math.erf, auf 1e-3 zusammengefasst): anderer Weg als das Vorzeichengitter."""
    ndtr = pytest.importorskip("scipy.special").ndtr
    x = (np.arange(81) + 0.37) / 81            # versetzt, damit kein Start genau auf dem instabilen Fixpunkt 0,5 liegt
    for _ in range(6000):
        x = ndtr((x - mu) / sigma)
    return len(set(np.round(x, 3).tolist()))


def test_stable_fixed_point_count_matches_iteration_attractors_on_the_whole_grid():
    for mu in (0.2, 0.3, 0.5, 0.7):
        for sigma in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
            assert T.count_stable_fixed(mu, sigma) == attractor_count(mu, sigma), (mu, sigma)


# ------------------------------------------------------------------------------------------------------------ 6. Zwei Rollen


def test_two_role_replicator_matches_scipy(bos):
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp
    A, B, _ = bos
    for z0 in ([0.2, 0.7], [0.8, 0.4], [0.5, 0.5], [0.9, 0.05]):
        sol = solve_ivp(lambda t, z: D.two_role_rhs(np.array(z), A, B), (0, 6), z0, method="DOP853", rtol=1e-12, atol=1e-14)
        mine = D.two_role_replicator(np.array(z0), A, B, 6.0, 600)
        assert mine == pytest.approx(sol.y[:, -1], abs=1e-6), z0


def test_two_role_mixed_point_matches_enumeration():
    for r1, r2 in ((2, 2), (3, 2), (4, 4), (1, 5), (7, 3)):
        A, B = S.role_matrices(r1, r2)
        mixed = [ne for ne in G.enumerate_ne(A, B) if 0 < ne[0] < 1]
        assert mixed == [D.two_role_mixed_point(A, B)], (r1, r2)


# ------------------------------------------------------------------------------------------------------------ 7. Zufallsspiele (E1)


def test_random_game_statistics_match_game_by_game_exact_evaluation():
    a, b, c, d = S.draw_games(100001, 3000, S.MODE_GENERAL)
    m = S.coordination_subset(a, b, c, d)
    stats = X.random_game_stats(100001, 3000, S.MODE_GENERAL)
    assert stats["games"] == int(m.sum())
    risky = total = 0
    basin_sum = F(0)
    for i in np.flatnonzero(m):
        game = tuple(F(float(v)) for v in (a[i], b[i], c[i], d[i]))
        assert G.classify(*game) == G.KOORDINATION
        basin = G.payoff_dominant_basin(*game)
        total += 1
        risky += int(basin < F(1, 2))
        basin_sum += basin
        assert (G.risk_dominant(*game) != G.payoff_dominant(*game)) == (basin < F(1, 2))      # auszahlungsdominiert gegen risikodominiert
    assert stats["risk_dominated"] == pytest.approx(risky / total, abs=1e-12)
    assert stats["mean_basin"] == pytest.approx(float(basin_sum / total), abs=1e-12)


def test_safe_hare_draws_have_d_equal_b():
    a, b, c, d = S.draw_games(100002, 500, S.MODE_SAFE_H)
    assert (d == b).all()
    assert (S.draw_games(100002, 500, S.MODE_GENERAL)[3] != b).all()


def test_gth_reference_matches_the_hand_ratio_for_n2():
    """Das Orakel selbst an der Handrechnung (ORACLE-PLAYBOOK §1): Hirsch (5,2,1,2), n = 2: pi(1)/pi(0) = 2·exp(-beta), pi(2)/pi(1) = exp(3·beta)/2."""
    for beta in (0.5, 2.0):
        pi = gth_stationary(full_transition_matrix(2, (5, 2, 1, 2), beta))
        assert pi[1] / pi[0] == pytest.approx(2 * np.exp(-beta), rel=1e-12)
        assert pi[2] / pi[1] == pytest.approx(np.exp(3 * beta) / 2, rel=1e-12)
