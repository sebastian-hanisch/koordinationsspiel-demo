"""kos_dynamics gegen Handrechnung: Replikator, ganzzahlige beste Antwort (inklusive Schwellenwert genau), Absorptionswahrscheinlichkeiten, Zwei-Rollen-Dynamik."""

from fractions import Fraction as F

import numpy as np
import pytest

import kos_dynamics as D
import kos_games as G
import kos_threshold as T
from kos_rng import SplitMix64


def test_replicator_rhs_by_hand():
    """Hirsch (5,2,1,2), x = 0,5: u_S = 0,5·5 + 0,5·1 = 3, u_H = 0,5·2 + 0,5·2 = 2, also dx/dt = 0,25·(3 - 2) = 0,25."""
    assert D.replicator_rhs(0.5, (5, 2, 1, 2)) == pytest.approx(0.25)
    assert D.replicator_rhs(0.0, (5, 2, 1, 2)) == 0.0 and D.replicator_rhs(1.0, (5, 2, 1, 2)) == 0.0     # die Ränder stehen
    assert D.replicator_rhs(0.25, (5, 2, 1, 2)) == pytest.approx(0.0)                                    # p* ist Ruhepunkt (u_S = u_H = 2)


def test_replicator_runs_to_the_nearer_edge(games):
    path = D.replicator_path(np.array([0.2, 0.3]), games["hirsch"], 60.0, 600)           # p* = 1/4: 0,2 darunter, 0,3 darüber
    assert path.shape == (601, 2)
    assert path[0].tolist() == [0.2, 0.3]
    assert path[-1, 0] < 1e-6 and path[-1, 1] > 1 - 1e-6


def test_replicator_fixed_at_the_edges_and_at_p_star(games):
    end = D.replicator_final(np.array([0.0, 1.0, 0.25]), games["hirsch"], 30.0, 300)
    assert end.tolist() == pytest.approx([0.0, 1.0, 0.25], abs=1e-9)


def test_replicator_anti_game_goes_to_the_mixed_point(games):
    end = D.replicator_final(np.array([0.05, 0.95]), games["feigling"], 60.0, 600)       # x* = 2/3 stabil
    assert end == pytest.approx([2 / 3, 2 / 3], abs=1e-6)


def test_basin_boundary_is_p_star(games):
    for name in ("hirsch", "buendeln", "gleichauf"):
        g = games[name]
        assert D.basin_boundary(g) == pytest.approx(float(G.p_star(*g)), abs=1e-4)


def test_basin_boundary_without_second_equilibrium(games):
    assert D.basin_boundary(games["dominant"]) == 0.0                 # Bündeln dominiert: schon 0,001 läuft nach S
    assert D.basin_boundary((1, 4, 0, 3)) == 1.0                      # Direkt dominiert (a - d = -2 < 0, b - c = 4 > 0): auch 0,999 läuft nach H


def test_switch_gain_by_hand():
    """Hirsch, n = 3: gain(m) = m·(a - d) - (n - 1 - m)·(b - c) = 3m - (2 - m) = 4m - 2."""
    assert [D.switch_gain(m, 3, (5, 2, 1, 2)) for m in range(3)] == [-2, 2, 6]


def test_wants_switch_by_hand():
    """Hirsch, n = 3: H-Spieler wechselt bei k S-Spielern, wenn 4k - 2 > 0 (k >= 1); S-Spieler wechselt, wenn 4(k - 1) - 2 < 0 (k <= 1)."""
    g = (5, 2, 1, 2)
    assert [D.wants_switch(3, k, g) for k in range(4)] == [(False, False), (True, True), (True, False), (False, False)]


def test_threshold_tie_does_not_switch():
    """Gleichauf (6,4,2,4), n = 31: (n - 1)·p* = 15 genau. Bei k = 15 anderen S: gain = 15·2 - 15·2 = 0, kein Wechsel (strikt). Mit Anteil in Gleitkomma wäre 15/30 - 0,5 verdächtig."""
    g = (6, 4, 2, 4)
    assert D.switch_gain(15, 31, g) == 0
    assert D.wants_switch(31, 15, g) == (False, True)       # H bei k = 15: gain(15) = 0, bleibt; S bei k = 15: gain(14) = -2 < 0 wechselt
    assert D.wants_switch(31, 16, g) == (True, False)       # H bei k = 16: gain(16) = 2 > 0 wechselt; S bei k = 16: gain(15) = 0, bleibt


def test_exact_threshold_one_third_with_31_players():
    """Der Fehler der Vormessung, ganzzahlig gerechnet: Spiel (6,4,3,4) mit p* = 1/3 und n = 31: (n - 1)·p* = 10 genau. Bei 10 anderen S ist gain = 10·2 - 20·1 = 0:
    kein strikter Vorteil, der Spieler bleibt. In Gleitkomma wäre 3·(10/30) - 1 nicht genau 0."""
    g = (6, 4, 3, 4)
    assert D.switch_gain(10, 31, g) == 0
    assert D.wants_switch(31, 10, g) == (False, True)       # H bei k = 10 (andere = 10): gain 0, bleibt. S bei k = 10 (andere = 9): gain = 18 - 22 = -4 wechselt
    assert D.wants_switch(31, 11, g) == (True, False)       # H bei k = 11: gain = 22 - 20 = 2 wechselt. S bei k = 11 (andere = 10): gain 0, bleibt


def test_br_run_stops_when_nobody_wants_to_switch(games):
    run = D.br_run(10, 0, games["hirsch"], SplitMix64(1))
    assert run.final == 0 and run.steps == 0 and run.converged
    run = D.br_run(10, 10, games["hirsch"], SplitMix64(1))
    assert run.final == 10 and run.steps == 0 and run.converged


def test_br_run_goes_to_the_side_of_p_star(games):
    for seed in range(5):
        assert D.br_run(20, 12, games["buendeln"], SplitMix64(seed)).final == 0      # 12/20 = 0,6 < p* = 2/3: alle direkt
        assert D.br_run(20, 16, games["buendeln"], SplitMix64(seed)).final == 20     # 0,8 > 2/3: alle bündeln


def test_br_run_records_the_path_and_is_deterministic(games):
    r1 = D.br_run(10, 5, games["hirsch"], SplitMix64(7), record=True)
    r2 = D.br_run(10, 5, games["hirsch"], SplitMix64(7), record=True)
    assert r1 == r2 and r1.path[0] == 5 and r1.path[-1] == r1.final and len(r1.path) == r1.steps + 1
    assert all(abs(x - y) <= 1 for x, y in zip(r1.path, r1.path[1:]))               # je Schritt höchstens ein Wechsel


def test_br_run_budget_exhausted_is_reported():
    run = D.br_run(50, 25, (6, 4, 2, 4), SplitMix64(3), max_steps=0)
    assert run.steps == 0 and not run.converged and run.final == 25


def test_absorption_by_hand_hirsch_n3():
    """Hirsch, n = 3 (siehe test_wants_switch_by_hand): k=0 hält, k=1 beide Richtungen (aufwärts 2/3, abwärts 1/3), k=2 nur aufwärts, k=3 hält.
    h(1) = (2/3)·h(2) / (2/3 + 1/3) = 2/3, h(2) = 1."""
    assert D.br_absorption_probabilities(3, (5, 2, 1, 2)) == [0, F(2, 3), 1, 1]


def test_absorption_by_hand_buendeln_n3():
    """Bündeln (6,4,0,4), n = 3: gain(m) = 6m - 8. H wechselt bei k >= 2, S wechselt bei k <= 2. k=1: nur abwärts (-> 0). k=2: aufwärts 1/3, abwärts 2/3: h(2) = 1/3."""
    assert D.br_absorption_probabilities(3, (6, 4, 0, 4)) == [0, 0, F(1, 3), 1]


def test_absorption_for_dominant_game_is_one_everywhere(games):
    assert D.br_absorption_probabilities(6, games["dominant"]) == [1] * 7


def test_absorption_for_anti_game_is_zero_except_at_all_s(games):
    h = D.br_absorption_probabilities(10, games["feigling"])
    assert all(v == 0 for v in h)                  # die Kette endet im inneren stabilen Zustand, nie bei k = n


def test_stable_state_in_anti_game_n50():
    """Feigling (-1,2,4,0), n = 50: gain(m) = m·(a - d) - (49 - m)·(b - c) = -m + 2·(49 - m) = 98 - 3m. Ein H-Spieler wechselt bei gain(k) > 0, also k <= 32;
    ein S-Spieler bei gain(k - 1) < 0, also k >= 34. Niemand wechselt nur bei k = 33 (33/50 = 0,66, nahe V/C = 2/3)."""
    g = (-1, 2, 4, 0)
    assert [D.switch_gain(m, 50, g) for m in (32, 33)] == [2, -1]
    assert [k for k in range(51) if D.stable_state(50, k, g)] == [33]


def test_two_role_mixed_point_of_bos(bos):
    A, B, (p, q) = bos
    assert D.two_role_mixed_point(A, B) == (p, q)


def test_two_role_rhs_vanishes_at_the_mixed_point_and_at_corners(bos):
    A, B, (p, q) = bos
    z = np.array([float(p), float(q)])
    assert D.two_role_rhs(z, A, B) == pytest.approx([0.0, 0.0], abs=1e-12)
    for corner in ([0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]):
        assert D.two_role_rhs(np.array(corner), A, B).tolist() == [0.0, 0.0]


def test_two_role_replicator_goes_to_the_corner_of_the_quadrant(bos):
    A, B, _ = bos
    far = D.two_role_replicator(np.array([[0.9, 0.9], [0.1, 0.1]]), A, B, 60.0, 600)
    assert D.two_role_goal(far).tolist() == [0, 1]
    assert D.two_role_goal(np.array([0.5, 0.5])) == -1                   # noch unterwegs


def test_two_role_br_run_ends_in_a_corner(bos):
    A, B, _ = bos
    kr, kc, converged = D.two_role_br_run(10, 10, 8, 8, A, B, SplitMix64(1))
    assert converged and (kr, kc) in ((10, 10), (0, 0))


# ---------------------------------------------------------------------------------------------------------------- Lücken aus der Mutationsprüfung


class _Fixed:
    """Zufallsquelle, die eine feste Folge von Zahlen liefert (wählt gezielt einen Spieler)."""

    def __init__(self, *values):
        self.values = list(values)

    def uniform(self):
        return self.values.pop(0)


def test_chain_probs_have_one_entry_per_state():
    up, down = D.br_chain_probs(7, (6, 4, 2, 4))
    assert len(up) == len(down) == 8


def test_replicator_final_equals_the_last_point_of_the_path(games):
    for name in ("hirsch", "feigling"):
        x0 = np.array([0.1, 0.4, 0.8])
        assert D.replicator_final(x0, games[name], 5.0, 50) == pytest.approx(D.replicator_path(x0, games[name], 5.0, 50)[-1], abs=1e-15)


def test_replicator_path_one_step_by_hand():
    """Hirsch, x = 0,5, Schrittweite 1 und ein Schritt: RK4 mit f(x) = x(1 - x)(4x - 1) (u_S - u_H = 5x + (1 - x) - 2 = 4x - 1)."""
    f = lambda x: x * (1 - x) * (4 * x - 1)                                   # noqa: E731
    k1 = f(0.5); k2 = f(0.5 + 0.5 * k1); k3 = f(0.5 + 0.5 * k2); k4 = f(0.5 + k3)
    assert D.replicator_path(0.5, (5, 2, 1, 2), 1.0, 1)[-1] == pytest.approx(min(1.0, 0.5 + (k1 + 2 * k2 + 2 * k3 + k4) / 6))


# Zwei Rollen mit allgemeinen Matrizen (die Rollenmatrizen des Kampfs der Geschlechter haben Nullen außerhalb der Diagonale und verdecken Vorzeichenfehler dort)
GA = [[4, 1], [3, 2]]
GB = [[1, 3], [0, 5]]


def test_two_role_br_one_forced_step_by_hand_general_matrices():
    """n = 2 + 2, kr = 1, kc = 2. gain_row = kc·(A00 - A10) + (2 - kc)·(A01 - A11) = 2 > 0, gain_col = kr·(B00 - B01) + (2 - kr)·(B10 - B11) = -2 - 5 = -7 < 0.
    Zufallszahl 0,3 wählt Spieler 1 (Zeile, Option 1): wechselt zu Option 0, kr = 2. Zahl 0,9 wählt Spaltenspieler 1 (Option 0): wechselt zu Option 1, kc = 1."""
    assert D.two_role_br_run(2, 2, 1, 2, GA, GB, _Fixed(0.3), max_steps=1) == (2, 2, False)
    assert D.two_role_br_run(2, 2, 1, 2, GA, GB, _Fixed(0.9), max_steps=1) == (1, 1, False)


def test_two_role_br_stable_state_with_nobody_on_option_zero():
    """kr = kc = 0: gain_row = 0 + 2·(0 - 1)... = -2 < 0, gain_col = 0 + 2·(0 - 5) = -10 < 0: niemand will zu Option 0, also gleich fertig (0 Schritte)."""
    assert D.two_role_br_run(2, 2, 0, 0, GA, GB, _Fixed(), max_steps=50) == (0, 0, True)


def test_two_role_br_stops_when_everyone_is_on_option_zero_and_wants_to_stay():
    A = [[4, 1], [3, 2]]
    assert D.two_role_br_run(3, 3, 3, 0, A, [[5, 0], [0, 1]], _Fixed(), max_steps=0)[2] is False        # ohne Schritte: nicht als fertig gemeldet, wenn jemand will
    assert D.two_role_br_run(3, 3, 3, 3, A, [[5, 0], [0, 1]], _Fixed())[:2] == (3, 3)


def test_two_role_goal_margin_is_exclusive_and_needs_both_roles():
    assert D.two_role_goal(np.array([0.95, 0.95])) == 0 and D.two_role_goal(np.array([0.05, 0.05])) == 1
    assert D.two_role_goal(np.array([0.95, 0.5])) == -1 and D.two_role_goal(np.array([0.5, 0.95])) == -1
    assert D.two_role_goal(np.array([0.05, 0.5])) == -1 and D.two_role_goal(np.array([0.5, 0.05])) == -1
    assert D.two_role_goal(np.array([0.1, 0.1])) == -1 and D.two_role_goal(np.array([0.9, 0.9])) == -1      # genau auf dem Rand: noch nicht angekommen


def test_two_role_rhs_general_matrices_against_expected_payoffs():
    rng = np.random.default_rng(2)
    for _ in range(40):
        A = rng.integers(-5, 6, (2, 2)).tolist()
        B = rng.integers(-5, 6, (2, 2)).tolist()
        p, q = rng.uniform(0.05, 0.95, 2)
        # Zeile: Erwartungswerte ihrer Optionen gegen die Spalte (q = Anteil Option 0); Spalte: gegen die Zeile (p)
        u0 = q * A[0][0] + (1 - q) * A[0][1]
        u1 = q * A[1][0] + (1 - q) * A[1][1]
        v0 = p * B[0][0] + (1 - p) * B[1][0]
        v1 = p * B[0][1] + (1 - p) * B[1][1]
        assert D.two_role_rhs(np.array([p, q]), A, B) == pytest.approx([p * (1 - p) * (u0 - u1), q * (1 - q) * (v0 - v1)], abs=1e-12)


def test_two_role_mixed_point_general_matrices_against_enumeration():
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(300):
        A = rng.integers(-5, 6, (2, 2)).tolist()
        B = rng.integers(-5, 6, (2, 2)).tolist()
        mixed = [ne for ne in G.enumerate_ne(A, B) if 0 < ne[0] < 1 and 0 < ne[1] < 1]
        if not mixed:
            continue
        checked += 1
        assert mixed == [D.two_role_mixed_point(A, B)]
    assert checked > 20


def test_best_agreed_welfare_alternating_is_the_mean_of_c_and_d():
    assert G.best_agreed_welfare(-1, 2, 6, 2) == F(4)                   # (6 + 2)/2 gegen a = -1 und b = 2


def test_two_role_br_column_gain_uses_the_difference_of_the_first_row():
    """n_row = 3, n_col = 2, kr = 2, kc = 2, B = [[1, 3], [0, 5]]: gain_col = kr·(B00 - B01) + (n_row - kr)·(B10 - B11) = 2·(-2) + 1·(-5) = -9 < 0, die Spalte will zu Option 1.
    Zufallszahl 0,85 wählt von 5 Spielern den Spaltenspieler 1 (Option 0): er wechselt, kc = 1."""
    assert D.two_role_br_run(3, 2, 2, 2, GA, GB, _Fixed(0.85), max_steps=1) == (2, 1, False)


def test_two_role_goal_needs_both_roles_strictly_inside_the_margin():
    for point, goal in (([0.9, 0.95], -1), ([0.95, 0.9], -1), ([0.1, 0.05], -1), ([0.05, 0.1], -1), ([0.91, 0.91], 0), ([0.09, 0.09], 1)):
        assert D.two_role_goal(np.array(point)) == goal, point


def test_fixed_point_exhausted_iterations_still_returns_a_number():
    x = T.fixed_point_final(0.5, 0.3, 0.0, max_iter=3)
    assert isinstance(x, float) and 0 <= x < T.fixed_point_final(0.5, 0.3, 0.0) + 1e-9


# Vier Richtungen (Zeile auf/ab, Spalte auf/ab), je mit Vorteil, Nachteil und Gleichstand; n = 2 + 2 Spieler, echter Zufallsstrom. Die Matrizen sind so gewählt, dass die andere Rolle gleichgültig ist.
POS_A, TIE_A, NEG_A = [[3, 2], [1, 0]], [[1, 2], [1, 2]], [[0, 0], [2, 2]]        # gain_row = +4 / 0 / -4 unabhängig von kc
POS_B, TIE_B, NEG_B = [[3, 1], [2, 0]], [[1, 1], [2, 2]], [[0, 2], [0, 2]]        # gain_col = +4 / 0 / -4 unabhängig von kr


@pytest.mark.parametrize("A,B,start,expected", [
    (POS_A, TIE_B, (0, 1), (2, 1, True)),        # Zeile will Option 0: alle Zeilen wechseln, Spalte gleichgültig
    (POS_A, TIE_B, (2, 1), (2, 1, True)),        # schon alle auf Option 0 und zufrieden: nichts passiert
    (NEG_A, TIE_B, (2, 1), (0, 1, True)),        # Zeile will Option 1: alle Zeilen wechseln herunter
    (NEG_A, TIE_B, (0, 1), (0, 1, True)),        # schon alle auf Option 1: nicht wechseln, kein Weiterlaufen
    (NEG_A, TIE_B, (1, 1), (0, 1, True)),        # kr = 1 geht auch auf 0 (Grenze kr > 0 inklusive)
    (TIE_A, TIE_B, (1, 1), (1, 1, True)),        # Gleichstand: niemand wechselt (strikte Verbesserung)
    (TIE_A, POS_B, (1, 0), (1, 2, True)),        # Spalte will Option 0
    (TIE_A, POS_B, (1, 2), (1, 2, True)),
    (TIE_A, NEG_B, (1, 2), (1, 0, True)),        # Spalte will Option 1
    (TIE_A, NEG_B, (1, 1), (1, 0, True)),
    (TIE_A, NEG_B, (1, 0), (1, 0, True)),
])
def test_two_role_br_each_direction_ends_where_the_stronger_side_wants(A, B, start, expected):
    assert D.two_role_br_run(2, 2, start[0], start[1], A, B, SplitMix64(5), max_steps=200) == expected


def test_two_role_br_gain_signs_with_asymmetric_off_diagonal_entries():
    """Die Differenz A00 - A10 (und B00 - B01) geht mit ihrem Vorzeichen ein: A = [[3, 0], [0, 1]] gegen A = [[3, 0], [4, 1]] ändern das Urteil der Zeile bei kc = 2."""
    assert D.two_role_br_run(2, 2, 0, 2, [[3, 0], [0, 1]], TIE_B, SplitMix64(1), max_steps=200)[:2] == (2, 2)      # gain = 2·3 + 0 = 6 > 0: alle zu Option 0
    assert D.two_role_br_run(2, 2, 0, 2, [[3, 0], [4, 1]], TIE_B, SplitMix64(1), max_steps=200)[:2] == (0, 2)      # gain = 2·(3 - 4) = -2 < 0: keiner wechselt


def test_two_role_br_acts_on_a_gain_of_exactly_one_in_every_direction():
    """Kleinster Vorteil (Gewinn 1 bei Zählung): Zeile hinauf [n_row = 2, n_col = 1, A = [[1, 0], [0, 0]], gain = kc = 1], Zeile herab [A = [[0, 0], [1, 0]], gain = -1],
    Spalte hinauf [n_row = 1, n_col = 2, B = [[1, 0], [0, 0]], gain = kr = 1], Spalte herab [B = [[0, 1], [0, 0]], gain = -kr = -1]."""
    tie_b = [[1, 1], [2, 2]]
    tie_a = [[1, 2], [1, 2]]
    assert D.two_role_br_run(2, 1, 0, 1, [[1, 0], [0, 0]], tie_b, SplitMix64(3), max_steps=200) == (2, 1, True)
    assert D.two_role_br_run(2, 1, 2, 1, [[0, 0], [1, 0]], tie_b, SplitMix64(3), max_steps=200) == (0, 1, True)
    assert D.two_role_br_run(1, 2, 1, 0, tie_a, [[1, 0], [0, 0]], SplitMix64(3), max_steps=200) == (1, 2, True)
    assert D.two_role_br_run(1, 2, 1, 2, tie_a, [[0, 1], [0, 0]], SplitMix64(3), max_steps=200) == (1, 0, True)
