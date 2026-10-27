"""kos_logit und kos_threshold gegen Handrechnung."""

import math

import numpy as np
import pytest

import kos_logit as L
import kos_threshold as T
from kos_rng import SplitMix64

# ---------------------------------------------------------------------------------------------------------------------------- Logit


def test_n2_hirsch_ratio_by_hand():
    """Hirsch (5,2,1,2), n = 2: Delta_0 = (0·3 - 1·1)/1 = -1, Delta_1 = (1·3 - 0·1)/1 = 3. up(0) = 2/2·sigma(-beta), down(1) = 1/2·sigma(beta).
    pi(1)/pi(0) = up(0)/down(1) = 2·sigma(-beta)/sigma(beta) = 2·exp(-beta). Gibbs: Phi = (1, 0, 3), C(2,k) = (1, 2, 1): 2·exp(beta·(0 - 1)) = 2·exp(-beta)."""
    g = (5, 2, 1, 2)
    assert [L.advantage(2, k, g) for k in (0, 1)] == [-1.0, 3.0]
    assert [L.potential_of_state(2, k, g) for k in (0, 1, 2)] == [1.0, 0.0, 3.0]
    for beta in (0.5, 1.0, 3.0):
        pi = L.stationary(2, g, beta)
        assert pi[1] / pi[0] == pytest.approx(2 * math.exp(-beta), rel=1e-12)
        pg = L.gibbs(2, g, beta)
        assert pg[1] / pg[0] == pytest.approx(2 * math.exp(-beta), rel=1e-12)
        assert pi[2] / pi[1] == pytest.approx(0.5 * math.exp(3 * beta), rel=1e-12)      # up(1) = 1/2·sigma(3·beta), down(2) = 1·sigma(-3·beta)


def test_stationary_is_a_distribution_and_matches_gibbs(games):
    for g in games.values():
        for n in (2, 5, 30):
            for beta in (0.1, 2.0, 10.0):
                pi = L.stationary(n, g, beta)
                assert pi.sum() == pytest.approx(1.0, abs=1e-12) and (pi >= 0).all()
                assert np.abs(pi - L.gibbs(n, g, beta)).max() < 1e-12


def test_stationary_does_not_overflow_at_the_slider_maximum(games):
    pi = L.stationary(200, games["buendeln"], 10.0)
    assert np.isfinite(pi).all() and pi.sum() == pytest.approx(1.0, abs=1e-12)


def test_mass_ranges_by_hand():
    pi = np.array([0.1, 0.2, 0.3, 0.4])           # n = 3: Anteile 0, 1/3, 2/3, 1
    assert L.mass_in_range(pi, 0.0, 0.1) == pytest.approx(0.1)
    assert L.mass_in_range(pi, 0.9, 1.0) == pytest.approx(0.4)
    assert L.mass_in_range(pi, 0.3, 0.7) == pytest.approx(0.5)       # 1/3 und 2/3
    assert L.mean_share(pi) == pytest.approx((0.2 + 0.6 + 1.2) / 3)


def test_mass_range_boundary_is_inclusive():
    pi = np.full(11, 1 / 11)                       # n = 10: Anteile 0, 0,1, ..., 1
    assert L.mass_in_range(pi, 0.1, 0.1) == pytest.approx(1 / 11)
    assert L.mass_in_range(pi, 0.9, 1.0) == pytest.approx(2 / 11)


def test_logit_peak_is_the_risk_dominant_equilibrium(games):
    """Bündeln (6,4,0,4): Direkt risikodominant, bei beta = 2 liegt die Masse bei 'fast alle direkt', nicht bei 'alle bündeln'."""
    pi = L.stationary(50, games["buendeln"], 2.0)
    assert L.mass_in_range(pi, 0.0, 0.1) > 0.99 and L.mass_in_range(pi, 0.9, 1.0) < 1e-6
    pi = L.stationary(50, games["hirsch"], 2.0)
    assert L.mass_in_range(pi, 0.9, 1.0) > 0.99


def test_logit_run_is_deterministic_and_moves_one_step_at_most(games):
    k1, p1 = L.logit_run(20, 10, games["hirsch"], 2.0, SplitMix64(5), 200, record=True)
    k2, p2 = L.logit_run(20, 10, games["hirsch"], 2.0, SplitMix64(5), 200, record=True)
    assert (k1, p1) == (k2, p2) and len(p1) == 201 and p1[0] == 10
    assert all(abs(x - y) <= 1 for x, y in zip(p1, p1[1:]))
    assert all(0 <= x <= 20 for x in p1)


def test_logit_run_with_huge_beta_is_a_best_response_run(games):
    """beta = 10 mit Auszahlungsunterschied >= 1: fast nie ein Fehlerschritt; aus k = 18 von 20 (Hirsch) bleibt alles bei 'alle bündeln'."""
    k, _ = L.logit_run(20, 18, games["hirsch"], 10.0, SplitMix64(2), 2000)
    assert k >= 19


# ---------------------------------------------------------------------------------------------------------------------- Schwellenmodell


def test_granovetter_by_hand():
    """(0,1,2): die Person mit Schwelle 0 macht mit, dann die mit 1, dann die mit 2: 3. (0,2,2): nur die erste, dann keine mit Schwelle <= 1: 1."""
    assert T.count_final([0, 1, 2]) == 3
    assert T.count_final([0, 2, 2]) == 1


def test_granovetter_classic_example():
    assert T.count_final(list(range(100))) == 100
    changed = list(range(100))
    changed[1] = 2
    assert T.count_final(changed) == 1


def test_count_final_with_initial_movers():
    assert T.count_final([5, 5, 5, 5, 5, 5], n0=0) == 0
    assert T.count_final([5, 5, 5, 5, 5, 5], n0=5) == 6                      # 5 machen mit, Schwelle 5 erreicht: alle 6
    assert T.count_final([5, 5, 5, 5, 5, 5], n0=4) == 4                      # 4 < 5: stehen bleiben (die 4 Anfangs-Mitmacher bleiben)


def test_count_final_threshold_is_inclusive():
    assert T.count_final([1, 1], n0=1) == 2                                  # Schwelle genau erreicht
    assert T.count_final([1.5, 1.5], n0=1) == 1                              # 1 < 1,5


def test_agents_final_threshold_of_seven_persons_is_not_lost_to_rounding():
    """100 Personen: 10 mit Schwelle 0,07 (= 7 Personen), 90 mit 0,99; s0 = 7 % (7 Personen). 0,07·100 ist in Gleitkomma 7,000000000000001: ohne Runden blieben es 7 Mitmachende,
    mit Runden greift die Schwelle bei genau 7 und alle 10 machen mit (Endanteil 0,1)."""
    assert 0.07 * 100 > 7
    theta = [0.07] * 10 + [0.99] * 90
    assert T.agents_final(theta, 7) == pytest.approx(0.10)
    assert T.agents_final(theta, 6) == pytest.approx(0.06)          # 6 Mitmachende < 7: es bleibt bei den 6 vom Anfang


def test_agents_final_s0_floor():
    theta = [0.5] * 10
    assert T.agents_final(theta, 50) == 1.0           # 5 von 10 machen mit, Schwelle 5 Personen erreicht
    assert T.agents_final(theta, 49) == pytest.approx(0.4)    # 49 % von 10 = 4 (abgerundet), 4 < 5


def test_fixed_point_by_hand():
    """mu = 0,5, sigma = 0,3, s0 = 0: untere Lösung von x = Phi((x - 0,5)/0,3), gegen den eingesetzten Wert."""
    x = T.fixed_point_final(0.5, 0.3)
    assert x == pytest.approx(T.normal_cdf(x, 0.5, 0.3), abs=1e-10) and 0.05 < x < 0.12
    assert T.fixed_point_final(0.5, 0.3, 0.2) >= 0.2


def test_normal_cdf_reference_values():
    assert T.normal_cdf(0.0, 0.0, 1.0) == 0.5
    assert T.normal_cdf(1.0, 0.0, 1.0) == pytest.approx(0.8413447460685429, rel=1e-12)       # Phi(1)
    assert T.normal_cdf(0.7, 0.5, 0.2) == pytest.approx(0.8413447460685429, rel=1e-12)


def test_stable_fixed_point_counts_hand_cases():
    assert T.count_stable_fixed(0.5, 1.0) == 1          # sehr flach: ein Schnittpunkt
    assert T.count_stable_fixed(0.5, 0.1) == 2          # steiler als die Diagonale in der Mitte: unten und oben stabil
    assert T.count_stable_fixed(0.5, 0.4) == 1          # Steigung in der Mitte 1/(0,4·sqrt(2 pi)) = 0,997 < 1
    assert T.count_stable_fixed(0.5, 0.39) == 2         # 1/(0,39·sqrt(2 pi)) = 1,023 > 1


def test_standard_normals_are_the_inverse_cdf_of_the_splitmix_stream():
    from statistics import NormalDist
    z = T.standard_normals(7, 5000)
    assert z.shape == (5000,) and abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05
    assert np.array_equal(z, T.standard_normals(7, 5000))
    rng = SplitMix64(7)
    assert z[0] == NormalDist().inv_cdf(rng.uniform() + 2.0 ** -54)
    assert z[1] == NormalDist().inv_cdf(rng.uniform() + 2.0 ** -54)


def test_draw_thresholds_are_clipped_at_zero():
    th = T.draw_thresholds(0.1, 0.5, np.array([-3.0, 0.0, 1.0]))
    assert th.tolist() == pytest.approx([0.0, 0.1, 0.6])


def test_mass_range_with_a_thousand_players_uses_exact_scaling():
    """n = 1000, gleichverteilt über 1001 Zustände: Anteile k/1000. Mindestens 90 % heißt k = 900 bis 1000 (101 Zustände), höchstens 10 % heißt k = 0 bis 100."""
    pi = np.full(1001, 1 / 1001)
    assert L.mass_in_range(pi, 0.9, 1.0) == pytest.approx(101 / 1001) and L.mass_in_range(pi, 0.0, 0.1) == pytest.approx(101 / 1001)


def test_mass_range_upper_bound_is_inclusive_but_not_larger_with_a_thousand_players():
    pi = np.full(1001, 1 / 1001)
    assert L.mass_in_range(pi, 0.0, 0.9) == pytest.approx(901 / 1001)           # k = 0 bis 900, nicht 901
