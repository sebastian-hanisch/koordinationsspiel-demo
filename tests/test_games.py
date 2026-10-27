"""kos_games gegen Handrechnung: Klassen, p*, Dominanz, Potential, Gleichgewichte, Wohlfahrt (Erwartungswerte aus conftest.py)."""

from fractions import Fraction as F

import pytest

import kos_games as G


def test_classes_of_the_hand_games(games):
    assert G.classify(*games["hirsch"]) == G.KOORDINATION
    assert G.classify(*games["buendeln"]) == G.KOORDINATION
    assert G.classify(*games["feigling"]) == G.ANTI
    assert G.classify(*games["dominant"]) == G.DOMINANT
    assert G.classify(*games["entartet"]) == G.ENTARTET


def test_classes_at_the_boundaries():
    assert G.classify(2, 5, 1, 2) == G.ENTARTET          # a = d
    assert G.classify(3, 2, 2, 1) == G.ENTARTET          # b = c
    assert G.classify(3, 5, 1, 2) == G.KOORDINATION      # a - d = 1, b - c = 4 (kleinste Unterschiede)
    assert G.classify(1, 1, 2, 0) == G.DOMINANT          # a - d = 1 > 0, b - c = -1 < 0
    assert G.classify(0, 0, 1, 1) == G.ANTI              # a - d = -1, b - c = -1


def test_p_star_by_hand(games):
    assert G.p_star(*games["hirsch"]) == F(1, 4)         # 1 / (3 + 1)
    assert G.p_star(*games["buendeln"]) == F(2, 3)       # 4 / (2 + 4)
    assert G.p_star(*games["gleichauf"]) == F(1, 2)      # 2 / (2 + 2)
    assert G.p_star(*games["feigling"]) == F(2, 3)       # (2 - 4) / (-1 - 0 + 2 - 4) = -2 / -3
    assert G.p_star(*games["dominant"]) is None and G.p_star(*games["entartet"]) is None


def test_slope(games):
    assert G.slope(*games["hirsch"]) == 4                 # 3 + 1
    assert G.slope(*games["feigling"]) == -3              # -1 + -2


def test_dominance_verdicts(games):
    assert G.risk_dominant(*games["hirsch"]) == "S" and G.payoff_dominant(*games["hirsch"]) == "S"
    assert G.risk_dominant(*games["buendeln"]) == "H" and G.payoff_dominant(*games["buendeln"]) == "S"     # das Hook-Beispiel: auszahlungsdominant, aber risikodominiert
    assert G.risk_dominant(*games["gleichauf"]) is None and G.payoff_dominant(*games["gleichauf"]) == "S"
    assert G.risk_dominant(*games["feigling"]) is None                                                     # nur für Koordination definiert
    assert G.payoff_dominant(2, 2, 0, 0) is None                                                           # a = b


def test_basins(games):
    assert G.basin_of_S(*games["hirsch"]) == F(3, 4)
    assert G.payoff_dominant_basin(*games["buendeln"]) == F(1, 3)      # S ist auszahlungsdominant, Einzugsbereich 1 - 2/3
    assert G.payoff_dominant_basin(2, 2, 0, 0) is None                 # kein auszahlungsdominantes Gleichgewicht
    assert G.basin_of_S(*games["feigling"]) is None


def test_potential_by_hand(games):
    assert G.potential(*games["buendeln"]) == {("S", "S"): 2, ("H", "H"): 4, ("S", "H"): 0, ("H", "S"): 0}      # a - d = 2, b - c = 4
    assert G.potential(*games["hirsch"])[("S", "S")] == 3


def test_potential_difference_equals_gain_of_every_deviation(games):
    """Potentialeigenschaft: einseitige Abweichung ändert Auszahlung und Potential um dasselbe (beide Spieler, alle Profile)."""
    for g in games.values():
        a, b, c, d = g
        u = {("S", "S"): a, ("S", "H"): c, ("H", "S"): d, ("H", "H"): b}      # Auszahlung der Zeile
        phi = G.potential(*g)
        for other in "SH":
            assert u[("H", other)] - u[("S", other)] == phi[("H", other)] - phi[("S", other)], (g, other)      # Zeile weicht ab
            assert u[("H", other)] - u[("S", other)] == phi[(other, "H")] - phi[(other, "S")], (g, other)      # Spalte weicht ab (Spiegelung)


def test_equilibria_of_hirsch_and_dominant(games):
    rows = G.equilibria(*games["hirsch"])
    assert sorted((r["p"], r["q"]) for r in rows) == [(0, 0), (F(1, 4), F(1, 4)), (1, 1)]
    assert [r["art"] for r in rows].count("gemischt") == 1
    only = G.equilibria(*games["dominant"])
    assert [(r["p"], r["q"]) for r in only] == [(1, 1)]


def test_mixed_equilibrium_payoff_of_hirsch(games):
    """Hirschjagd (5,2,1,2), x = 1/4: u_S = 1/4·5 + 3/4·1 = 2, u_H = 1/4·2 + 3/4·2 = 2 (gleich, wie im gemischten Gleichgewicht verlangt)."""
    mixed = next(r for r in G.equilibria(*games["hirsch"]) if r["art"] == "gemischt")
    assert mixed["pay_row"] == 2 and mixed["pay_col"] == 2
    assert G.mixed_welfare(*games["hirsch"]) == 2


def test_anti_equilibria_are_asymmetric_pure_and_one_symmetric_mixed(games):
    rows = G.equilibria(*games["feigling"])
    assert sorted((r["p"], r["q"]) for r in rows) == [(0, 1), (F(2, 3), F(2, 3)), (1, 0)]
    assert [r["symmetrisch"] for r in rows if r["art"] == "rein"] == [False, False]


def test_feigling_welfare_by_hand(games):
    """V = 4, C = 6: im gemischten Gleichgewicht x = 2/3: Falke 2/3·(-1) + 1/3·4 = 2/3, Taube 2/3·0 + 1/3·2 = 2/3; Absprache 'abwechselnd': (4 + 0)/2 = 2."""
    assert G.mixed_welfare(*games["feigling"]) == F(2, 3)
    assert G.best_agreed_welfare(*games["feigling"]) == 2
    assert G.best_agreed_welfare(*games["buendeln"]) == 6


def test_degenerate_games_are_flagged_and_keep_their_pure_equilibria(games):
    a, b, c, d = games["entartet"]
    A, B = G.payoff_matrices(a, b, c, d)
    assert G.is_degenerate(A, B)
    pure = G.enumerate_ne(A, B)
    assert all(p in (0, 1) and q in (0, 1) for p, q in pure) and pure       # nur reine Profile, mindestens eines
    assert G.ne_count(*games["entartet"]) == len(pure)


def test_enumerate_ne_for_bos(bos):
    A, B, (p, q) = bos
    assert sorted(G.enumerate_ne(A, B)) == sorted([(F(1), F(1)), (F(0), F(0)), (p, q)])


@pytest.mark.parametrize("game", [(5, 2, 1, 2), (6, 4, 0, 4), (-1, 2, 4, 0), (3, 1, 2, 0)])
def test_equilibrium_count_by_class(game):
    expected = 1 if G.classify(*game) == G.DOMINANT else 3
    assert G.ne_count(*game) == expected


def test_equilibrium_names_are_assigned_to_the_right_profiles(games):
    names = {(r["p"], r["q"]): r["name"] for r in G.equilibria(*games["hirsch"])}
    assert names[(F(1), F(1))] == "beide bündeln" and names[(F(0), F(0))] == "beide direkt" and names[(F(1, 4), F(1, 4))].startswith("gemischt: jeder bündelt mit Wahrscheinlichkeit 1/4")
    anti = {(r["p"], r["q"]): r["name"] for r in G.equilibria(*games["feigling"])}
    assert anti[(F(1), F(0))] == "Zeile bündelt, Spalte fährt direkt" and anti[(F(0), F(1))] == "Zeile fährt direkt, Spalte bündelt"


def test_degenerate_game_keeps_exactly_its_weak_pure_equilibria():
    """(3, 2, 1, 3): a = d. (S,S): 3 >= 3 für beide, also ein Gleichgewicht (schwach); (H,H): 2 > 1; die gemischten Profile fehlen bewusst."""
    assert sorted(G.enumerate_ne(*G.payoff_matrices(3, 2, 1, 3))) == [(F(0), F(0)), (F(1), F(1))]
