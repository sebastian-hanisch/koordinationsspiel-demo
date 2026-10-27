"""Jede Zahl der README und der App-Texte, nachgerechnet (DEMO-PLAYBOOK §4, §11a). Gegliedert nach den Hypothesen H1 bis H7 und den Korrekturen gegenüber dem Plan.

Zählaussagen stehen als Zählung (k von m), Stichprobenanteile mit Marge, exakte Größen exakt (Brüche). Plattformabhängiges (Gleitkomma-Summen) mit Toleranz."""

import itertools
from fractions import Fraction as F

import numpy as np
import pytest

import kos_constants as C
import kos_dynamics as D
import kos_evaluation as V
import kos_experiments as X
import kos_games as G
import kos_logit as L
import kos_scenario as S
import kos_threshold as T
from kos_rng import SplitMix64

# ---------------------------------------------------------------------------------------------------------------------------- H1


def _integer_games():
    return itertools.product(range(10), repeat=4)


def test_h1_class_counts_over_all_10000_integer_games():
    """Alle 10 000 ganzzahligen Spiele mit Auszahlungen 0 bis 9: 2 025 Koordination, 2 025 Anti-Koordination, 4 050 dominant, 1 900 entartet."""
    counts = {}
    for game in _integer_games():
        counts[G.classify(*game)] = counts.get(G.classify(*game), 0) + 1
    assert counts == {G.KOORDINATION: 2025, G.ANTI: 2025, G.DOMINANT: 4050, G.ENTARTET: 1900}


def test_h1_enumeration_gives_three_or_one_equilibria_and_mixed_at_p_star_in_all_8100_nondegenerate_games():
    """Aufzählung ohne die Formel: 3 Gleichgewichte (Koordination, Anti), 1 (dominant); das gemischte liegt bei (p*, p*). Keine Ausnahme in 8 100 Spielen."""
    seen = 0
    for game in _integer_games():
        cls = G.classify(*game)
        if cls == G.ENTARTET:
            continue
        ne = G.enumerate_ne(*G.payoff_matrices(*game))
        seen += 1
        assert len(ne) == (1 if cls == G.DOMINANT else 3), game
        if cls != G.DOMINANT:
            assert (G.p_star(*game), G.p_star(*game)) in ne, game
    assert seen == 8100


# ---------------------------------------------------------------------------------------------------------------------------- H2


def test_h2_basin_boundaries_replicator_and_best_response_n100():
    rows = X.e2_basins()
    assert [r["game"] for r in rows] == [(5, 2, 1, 2), (4, 3, 1, 2), (6, 4, 0, 4), (3, 1, 0, 0)]
    assert [r["p_star"] for r in rows] == pytest.approx([0.25, 0.5, 2 / 3, 0.25])
    assert [r["replicator"] for r in rows] == pytest.approx([0.25, 0.5, 2 / 3, 0.25], abs=1e-4)
    assert [r["best_response"] for r in rows] == [0.25, 0.5, 0.67, 0.25]                  # exakt: kleinster Start mit Gewinnwahrscheinlichkeit >= 1/2, als Anteil von 100


def test_h2_best_response_boundary_is_exact_for_gleichauf_n100():
    """p* = 1/2, n = 100: (n - 1)·p* = 49,5, der Zustand k = 50 ist der Wechselzustand (aufwärts 50/100, abwärts 50/100): h = 1/2, also Grenze 50/100."""
    assert X.br_boundary(100, (4, 3, 1, 2)) == F(1, 2)
    assert D.br_absorption_probabilities(100, (4, 3, 1, 2))[50] == F(1, 2)


# ---------------------------------------------------------------------------------------------------------------------------- H3


def test_h3_random_games_payoff_dominant_equilibrium_is_risk_dominated_one_third_of_the_time():
    rows = {r["mode"]: r for r in X.e1_random_games()}
    g, h = rows[S.MODE_GENERAL], rows[S.MODE_SAFE_H]
    assert g["risk_dominated"] == pytest.approx(0.334, abs=0.0006) and g["risk_dominated_std"] == pytest.approx(0.0026, abs=0.001) and g["mean_basin"] == pytest.approx(0.590, abs=0.0006)
    assert h["risk_dominated"] == pytest.approx(0.4995, abs=0.0006) and h["mean_basin"] == pytest.approx(0.5005, abs=0.0006)
    assert 25000 < g["games"] < 25100 and 16600 < h["games"] < 16800 and g["seeds"] == 4 and g["draws"] == 100000


def test_h3_the_one_third_holds_with_other_streams_and_more_draws():
    """Gegenprobe (DEMO-PLAYBOOK §11a.5): vier andere Ströme zu 400 000 Ziehungen."""
    general = [X.random_game_stats(s, 400000, S.MODE_GENERAL)["risk_dominated"] for s in (200001, 200002, 200003, 200004)]
    safe = [X.random_game_stats(s, 400000, S.MODE_SAFE_H)["risk_dominated"] for s in (200001, 200002, 200003, 200004)]
    assert np.mean(general) == pytest.approx(0.3324, abs=0.001) and np.mean(safe) == pytest.approx(0.5018, abs=0.001)


def test_h3_the_share_depends_on_the_distribution_of_payoffs():
    """Dieselbe Frage mit ganzzahligen Auszahlungen 0 bis 9, jede Kombination einmal: 420 von 1 500 Koordinationsspielen (a != b, a - d != b - c) = 28,0 %, nicht 33,4 %."""
    total = risky = 0
    for game in _integer_games():
        a, b, c, d = game
        if G.classify(*game) != G.KOORDINATION or a == b or a - d == b - c:
            continue
        total += 1
        risky += G.risk_dominant(*game) != G.payoff_dominant(*game)
    assert (risky, total) == (420, 1500)


def test_h3_start_without_agreement_exact_values_n50_start_50_percent():
    rows = {r["game"]: r["all_s"] for r in X.e1_start_without_agreement()}
    assert rows[(5, 2, 1, 2)] == pytest.approx(0.9998, abs=5e-5)
    assert rows[(6, 4, 0, 4)] == pytest.approx(0.0106, abs=5e-5)
    assert rows[(6, 4, 3, 4)] == pytest.approx(0.9894, abs=5e-5)
    assert X.start_without_agreement(50, 50, (6, 4, 2, 4)) == F(1, 2) and X.start_without_agreement(50, 50, (8, 4, 0, 4)) == F(1, 2)


def test_h3_simulation_of_20000_runs_matches_the_exact_value_for_buendeln():
    """Die Simulation (SplitMix64, beste Antwort): Anteil 'alle bündeln' im Bündeln-Spiel bei Start 50 %, n = 50; exakt 0,0106."""
    rng = SplitMix64(1234)
    runs = 20000
    wins = sum(D.br_run(50, S.start_count(50, 50, rng), (6, 4, 0, 4), rng).final == 50 for _ in range(runs))
    assert abs(wins / runs - float(X.start_without_agreement(50, 50, (6, 4, 0, 4)))) < 0.004


def test_h3_gleichauf_start_correction_of_the_pre_measurement():
    """Die Vormessung hatte 0,484 (4 Ströme zu 400 Läufen); exakt ist es 1/2, und 8 000 Läufe mit anderem Strom liegen nahe 0,5 (3 Standardfehler = 0,017)."""
    rng = SplitMix64(77)
    runs = 8000
    wins = sum(D.br_run(50, S.start_count(50, 50, rng), (6, 4, 2, 4), rng).final == 50 for _ in range(runs))
    assert abs(wins / runs - 0.5) < 0.017


# ---------------------------------------------------------------------------------------------------------------------------- H4


def test_h4_gibbs_matches_the_chain_over_120_cells_and_300_random_games():
    rows = X.e3_logit_vs_gibbs()
    assert len(rows) == 4 and sum(r["cells"] for r in rows) == 120
    assert max(r["worst"] for r in rows) < 1e-12


def test_h4_logit_mass_in_the_buendeln_game_n50():
    """(6, 4, 0, 4), n = 50: Masse bei höchstens 10 % Bündelnden 0,114 / 0,998 / mehr als 99,9 % für β = 0,5 / 1 / 2; bei mindestens 90 % unter 10^-9."""
    rows = X.e3_masses(50, (6, 4, 0, 4))
    assert [r["beta"] for r in rows] == [0.5, 1.0, 2.0]
    assert rows[0]["low"] == pytest.approx(0.114, abs=5e-4) and rows[1]["low"] == pytest.approx(0.998, abs=5e-4) and rows[2]["low"] > 0.999
    assert all(r["high"] < 1e-9 for r in rows)


def test_h4_logit_mass_is_mirrored_for_p_star_one_third_and_split_for_gleichauf():
    rows = X.e3_masses(50, (6, 4, 3, 4))
    assert rows[2]["high"] > 0.99 and rows[2]["high"] == pytest.approx(0.998, abs=1e-3)
    mid = X.e3_masses(50, (6, 4, 2, 4))
    assert mid[2]["low"] == pytest.approx(0.498, abs=1e-3) and mid[2]["high"] == pytest.approx(0.498, abs=1e-3)


def test_h4_larger_logit_corner_is_the_risk_dominant_one_in_1500_of_1500_integer_games():
    """Alle Koordinationsspiele mit ganzzahligen Auszahlungen 0 bis 9, a != b, a - d != b - c (1 500): bei n = 50 und β = 2 hat der risikodominante Rand die größere Masse in allen 1 500,
    der auszahlungsdominante in 1 080 (72,0 %)."""
    total = risk = pay = 0
    for game in _integer_games():
        a, b, c, d = game
        if G.classify(*game) != G.KOORDINATION or a == b or a - d == b - c:
            continue
        pi = L.stationary(50, game, 2.0)
        bigger = "S" if L.mass_in_range(pi, 0.9, 1.0) > L.mass_in_range(pi, 0.0, 0.1) else "H"
        total += 1
        risk += bigger == G.risk_dominant(*game)
        pay += bigger == G.payoff_dominant(*game)
    assert (total, risk, pay) == (1500, 1500, 1080)


# ---------------------------------------------------------------------------------------------------------------------------- H5


def test_h5_two_role_replicator_shares():
    rows = {r["roles"]: r for r in X.e4_two_roles(runs=100)}
    assert rows[(2, 2)]["share_00"] == pytest.approx(0.49, abs=0.011) and rows[(4, 4)]["share_00"] == pytest.approx(0.49, abs=0.011)
    assert rows[(3, 2)]["share_00"] == pytest.approx(0.582, abs=0.006)
    assert all(r["unreached"] == 0.0 for r in rows.values())
    assert rows[(2, 2)]["direct_00"] == pytest.approx(2 / 9) and rows[(3, 2)]["direct_00"] == pytest.approx(0.25) and rows[(3, 2)]["direct_11"] == pytest.approx(1 / 6)
    assert rows[(2, 2)]["p_star"] == pytest.approx(2 / 3) and rows[(2, 2)]["q_star"] == pytest.approx(1 / 3)


def test_h5_two_role_best_response_shares_with_n30_each():
    """Beste Antwort mit 30 + 30 Spielern, Start 1/2 je Spieler: gleich starke Rollen nahe der Hälfte, 3 gegen 2 deutlich darüber (Band 0,76 bis 0,86)."""
    sym = np.mean([X.two_role_br_share(2, 2, 400, s)["share_00"] for s in (400001, 400002)])
    asym = np.mean([X.two_role_br_share(3, 2, 400, s)["share_00"] for s in (400001, 400002)])
    strong = np.mean([X.two_role_br_share(4, 4, 400, s)["share_00"] for s in (400001, 400002)])
    assert 0.45 < sym < 0.56 and 0.45 < strong < 0.56 and 0.76 < asym < 0.86


def test_h5_quadrants_explain_less_than_half_of_the_starts_in_the_symmetric_game():
    s = X.two_role_shares(2, 2)
    assert s["direct_00"] + s["direct_11"] == pytest.approx(4 / 9)                       # 44,4 %: die vier Quadranten laufen nur zur Hälfte direkt


# ---------------------------------------------------------------------------------------------------------------------------- H6


def test_h6_more_spread_opens_the_tipping_for_mean_threshold_30():
    rows = {(r["mu"], r["sigma"], r["s0"]): r for r in X.e5_thresholds()}
    assert rows[(30, 5, 0)]["mean"] == 0.0 and rows[(30, 10, 0)]["mean"] == pytest.approx(0.0014, abs=3e-4)
    assert rows[(30, 15, 0)]["mean"] == pytest.approx(0.092, abs=0.004) and rows[(30, 15, 0)]["fixed_point"] == pytest.approx(0.0438, abs=5e-4)
    assert rows[(30, 20, 0)]["mean"] == pytest.approx(0.9997, abs=5e-4) and rows[(30, 30, 0)]["mean"] == pytest.approx(0.989, abs=0.003) and rows[(30, 50, 0)]["mean"] == pytest.approx(0.874, abs=0.01)
    assert rows[(30, 15, 0)]["inner_std"] == pytest.approx(0.175, abs=0.04)


def test_h6_near_the_boundary_the_draw_decides():
    """μ = 30 %, σ = 15 %: in 400 Ziehungen kippt ein kleiner Teil (mindestens die Hälfte, 3 bis 8 % je Strom), der Fixpunkt liegt bei 4,4 %; die meisten Läufe enden bei wenigen Prozent."""
    for seed in (300001, 300002, 300003):
        cell = X.threshold_cell(30, 15, 0, 400, seed)
        tipped = float(np.mean(cell["finals"] >= 0.5))
        assert 0.035 <= tipped <= 0.075, seed
        assert float(np.median(cell["finals"])) < 0.06
        assert 0.079 <= cell["mean"] <= 0.111                                                  # das Mittel liegt über dem Fixpunkt 0,044, weil die gekippten Läufe es heben


def test_h6_mean_threshold_50_percent_needs_a_wide_spread_or_a_starting_share():
    rows = {(r["mu"], r["sigma"], r["s0"]): r for r in X.e5_thresholds()}
    assert rows[(50, 15, 0)]["mean"] < 0.001 and rows[(50, 30, 0)]["mean"] == pytest.approx(0.082, abs=0.01) and rows[(50, 50, 0)]["mean"] == pytest.approx(0.484, abs=0.03)
    assert rows[(50, 20, 10)]["mean"] == pytest.approx(0.1, abs=1e-12)                    # 10 % machen mit, sonst niemand


def test_h6_classic_example():
    ex = X.classic_example()
    assert (ex["full"], ex["changed"]) == (100, 1) and ex["mean_full"] == 49.5 and ex["mean_changed"] == pytest.approx(49.51)


def test_h6_stable_fixed_point_counts_are_one_or_two_never_three():
    """Korrektur der Vormessung: 'drei stabile Fixpunkte' war eine Doppelzählung (Randpunkt und Vorzeichenwechsel). Auf dem Raster mu 0,2/0,3/0,5/0,7 und sigma 0,02 bis 1: 1 oder 2."""
    counts = [T.count_stable_fixed(mu, sigma) for mu in (0.2, 0.3, 0.5, 0.7) for sigma in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)]
    assert set(counts) == {1, 2} and counts.count(2) == 16 and len(counts) == 32


# ---------------------------------------------------------------------------------------------------------------------------- H7


def test_h7_mixed_equilibrium_is_unstable_for_positive_slope_and_stable_for_negative():
    rows = {r["game"]: r for r in X.e6_anti()}
    for game in ((5, 2, 1, 2), (6, 4, 0, 4)):
        assert rows[game]["slope"] > 0 and rows[game]["near"] == 0.0 and rows[game]["edge"] == 1.0
    for game in ((-1, 2, 4, 0), (0, 3, 4, 2)):
        assert rows[game]["slope"] == -3 and rows[game]["near"] == 1.0 and rows[game]["edge"] == 0.0


def test_h7_feigling_welfare_mixed_one_third_of_agreed():
    wf = X.anti_welfare((-1, 2, 4, 0))
    assert wf["mixed"] == F(2, 3) and wf["agreed"] == 2 and wf["mixed"] / wf["agreed"] == F(1, 3)


def test_h7_the_agreement_gain_depends_on_the_payoffs():
    wf = X.anti_welfare((0, 3, 4, 2))
    assert wf["mixed"] == F(8, 3) and wf["agreed"] == 3 and wf["mixed"] / wf["agreed"] == F(8, 9)


def test_h7_best_response_in_the_feigling_game_stops_at_33_of_50():
    assert [k for k in range(51) if D.stable_state(50, k, (-1, 2, 4, 0))] == [33]
    runs = {D.br_run(50, S.start_count(50, 50, rng), (-1, 2, 4, 0), rng).final for rng in [SplitMix64(5)] for _ in range(50)}
    assert runs == {33}


def test_h7_logit_mean_share_approaches_two_thirds_from_below_n50():
    means = [L.mean_share(L.stationary(50, (-1, 2, 4, 0), b)) for b in (0.5, 1.0, 2.0, 5.0)]
    assert means == pytest.approx([0.545, 0.571, 0.599, 0.629], abs=6e-4)
    assert all(m < 2 / 3 for m in means) and means == sorted(means)


# ---------------------------------------------------------------------------------------------------------------------------- Texte der Presets und der App


def test_default_tab_texts_carry_the_measured_numbers():
    gv = V.analyse_game((6, 4, 0, 4))
    text = V.verdict_game(gv)[1]
    assert "2/3 (0,667)" in text and "0,333" in text
    ex = X.e1_start_without_agreement()
    assert V.pct(ex[1]["all_s"], 2) == "1,06 %"


def test_e3_worst_deviation_is_a_small_positive_number():
    """Die Abweichung ist Rundungsrauschen (größer als 0, kleiner als 1e-12): ein Auswertungsfehler, der 0 meldet, fiele auf."""
    worst = max(r["worst"] for r in X.e3_logit_vs_gibbs())
    assert 0 < worst < 1e-12


def test_threshold_fixed_point_uses_the_initial_share():
    rows = {(r["mu"], r["sigma"], r["s0"]): r for r in X.e5_thresholds()}
    assert rows[(50, 20, 10)]["fixed_point"] == pytest.approx(0.1, abs=1e-9)


def test_two_role_br_share_counts_every_run():
    r = X.two_role_br_share(2, 2, 1, 5)
    assert r["at00"] + r["at11"] == 1 and r["runs"] == 1
    r = X.two_role_br_share(2, 2, 7, 5)
    assert r["at00"] + r["at11"] == 7


def test_two_role_shares_reports_runs_that_do_not_arrive(monkeypatch):
    monkeypatch.setattr(C, "TWO_ROLE_T", 0.3)
    s = X.two_role_shares(2, 2, grid=10)
    assert s["unreached"] > 0.5 and 0 <= s["share_00"] <= 1
