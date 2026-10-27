"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Population, Gleichstand, Grenzen inklusive, Extremwerte der Regler."""

from fractions import Fraction as F

import numpy as np
import pytest

import kos_constants as C
import kos_dynamics as D
import kos_evaluation as V
import kos_games as G
import kos_logit as L
import kos_scenario as S
import kos_threshold as T
from kos_rng import SplitMix64


def test_two_players_best_response_by_hand():
    """n = 2, Hirsch (5,2,1,2): gain(m) = 3m - (1 - m) = 4m - 1. Bei k = 1 will ein H-Spieler wechseln (gain(1) = 3 > 0) und ein S-Spieler auch (gain(0) = -1 < 0):
    aufwärts 1/2, abwärts 1/2, also h(1) = 1/2."""
    g = (5, 2, 1, 2)
    assert D.wants_switch(2, 1, g) == (True, True)
    assert D.br_absorption_probabilities(2, g) == [0, F(1, 2), 1]
    run = D.br_run(2, 1, g, SplitMix64(4))
    assert run.final in (0, 2) and run.converged


def test_two_players_logit_and_gibbs_agree():
    assert np.abs(L.stationary(2, (6, 4, 0, 4), 1.0) - L.gibbs(2, (6, 4, 0, 4), 1.0)).max() < 1e-14


def test_all_payoffs_zero_is_degenerate_and_has_no_p_star():
    assert G.classify(0, 0, 0, 0) == G.ENTARTET and G.p_star(0, 0, 0, 0) is None
    assert D.br_run(10, 5, (0, 0, 0, 0), SplitMix64(1)).steps == 0           # niemand will wechseln
    assert V.verdict_game(V.analyse_game((0, 0, 0, 0)))[0] == "warning"


def test_extreme_payoffs_still_classified():
    assert G.classify(10, 10, -10, -10) == G.KOORDINATION and G.p_star(10, 10, -10, -10) == F(1, 2)
    assert G.classify(-10, -10, 10, 10) == G.ANTI and G.p_star(-10, -10, 10, 10) == F(1, 2)
    assert G.p_star(10, -9, -10, 9) == F(1, 2)                                  # a - d = 1, b - c = 1: die kleinsten Unterschiede


def test_start_at_the_ends():
    """Anfangsanteil 0 % und 100 %: jeder startet gleich, die Kette ist schon festgehängt (Koordination)."""
    g = (6, 4, 0, 4)
    rng = SplitMix64(1)
    assert S.start_count(20, 0, rng) == 0 and S.start_count(20, 100, rng) == 20
    from kos_experiments import start_without_agreement
    assert start_without_agreement(20, 0, g) == 0 and start_without_agreement(20, 100, g) == 1


def test_start_probability_between_the_ends_is_a_probability():
    from kos_experiments import start_without_agreement
    for start in (1, 50, 99):
        assert 0 < start_without_agreement(30, start, (5, 2, 1, 2)) < 1


def test_logit_at_the_beta_ends():
    for beta in (C.BETA_MIN, C.BETA_MAX):
        for n in (C.N_MIN, C.N_MAX):
            pi = L.stationary(n, (6, 4, 0, 4), beta)
            assert np.isfinite(pi).all() and abs(pi.sum() - 1) < 1e-12


def test_slider_maximum_runs_complete():
    """Worst Case: n = 200, Läufe maximal; es darf nichts hängen oder platzen (Zeit wird nicht geprüft, nur ein Ende)."""
    pv = V.analyse_population((6, 4, 2, 4), C.N_MAX, 50, "Beste Antwort", 2.0, C.RUNS_MAX, 3)
    assert pv.sim_runs == C.RUNS_MAX and len(pv.finals) == C.RUNS_MAX and all(0 <= f <= C.N_MAX for f in pv.finals)
    assert 0 <= pv.exact_all_s <= 1


def test_absorption_at_n200_is_a_probability_vector():
    h = D.br_absorption_probabilities(C.N_MAX, (6, 4, 2, 4))
    assert h[0] == 0 and h[C.N_MAX] == 1 and all(0 <= v <= 1 for v in h)
    assert all(x <= y for x, y in zip(h, h[1:]))                               # monoton im Start


def test_threshold_extremes_complete():
    for mu, sigma, s0 in ((C.MU_MIN, C.SIGMA_MIN, C.S0_MIN), (C.MU_MAX, C.SIGMA_MAX, C.S0_MAX), (C.MU_MIN, C.SIGMA_MAX, C.S0_MAX), (C.MU_MAX, C.SIGMA_MIN, C.S0_MIN)):
        tv = V.analyse_thresholds(mu, sigma, s0, 10, 5)
        assert len(tv.finals) == 10 and (tv.finals >= s0 / 100 - 1e-9).all() and (tv.finals <= 1).all()


def test_threshold_all_start_when_s0_is_30_and_mu_low():
    tv = V.analyse_thresholds(10, 2, 30, 5, 1)
    assert (tv.finals >= 0.3).all()


def test_role_values_at_the_ends():
    for r1, r2 in ((C.ROLE_MIN, C.ROLE_MIN), (C.ROLE_MAX, C.ROLE_MAX), (C.ROLE_MIN, C.ROLE_MAX)):
        rv = V.analyse_roles(r1, r2, 10, 2)
        assert 0 <= rv.shares["share_00"] <= 1 and rv.shares["unreached"] < 0.05


def test_role_one_against_one_has_mixed_point_one_half():
    A, B = S.role_matrices(1, 1)
    assert D.two_role_mixed_point(A, B) == (F(1, 2), F(1, 2))


def test_count_final_with_no_threshold_below_zero_start():
    """Niemand hat Schwelle 0: ohne Anfangsanteil macht niemand mit; mit n0 = 1 und Schwelle 1 alle."""
    assert T.count_final([1, 1, 1]) == 0
    assert T.count_final([1, 1, 1], n0=1) == 3


def test_anti_view_for_games_without_p_star():
    av = V.analyse_anti((3, 1, 2, 0), 20, 1.0)                                  # dominant: kein gemischtes Gleichgewicht
    assert av.welfare_mixed is None and av.row["p_star"] is None and av.row["near"] == 0.0
    assert V.verdict_anti(av)[0] == "info"


@pytest.mark.parametrize("game", [(0, 0, 0, 0), (10, 10, -10, -10), (-10, -10, 10, 10), (3, 1, 2, 0), (1, 4, 0, 3), (3, 2, 1, 3)])
def test_every_view_completes_for_every_class(game):
    gv = V.analyse_game(game)
    assert V.verdict_game(gv)[0] in ("info", "success", "warning")
    for rule in C.RULES:
        pv = V.analyse_population(game, 10, 50, rule, 1.0, 10, 1)
        assert V.verdict_population(pv, game)[0] in ("info", "warning")
    assert V.verdict_logit(V.analyse_logit(game, 10, 1.0), game)[0] in ("info", "warning")
    assert V.verdict_anti(V.analyse_anti(game, 10, 1.0))[0] == "info"
