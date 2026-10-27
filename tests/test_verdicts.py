"""Verdikte und Formate an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): Schwellen, Zweige und jede Zahl im Text. Die Verdikte sind reine Funktionen (Art, Text)."""

from fractions import Fraction as F

import numpy as np
import pytest

import kos_evaluation as V
import kos_games as G

# ---------------------------------------------------------------------------------------------------------------------------- Formate


def test_de_and_frac_and_num():
    assert V.de(0.6666666, 3) == "0,667" and V.de(2.0, 1) == "2,0"
    assert V.frac(F(2, 3)) == "2/3" and V.frac(F(4, 2)) == "2"
    assert V.num(F(2, 3)) == "2/3 (0,667)" and V.num(F(6)) == "6" and V.num(2) == "2" and V.num(2.5) == "2,500"


def test_pct_rounds_and_guards_the_ends():
    assert V.pct(0.5) == "50,0 %" and V.pct(0.0) == "0,0 %" and V.pct(1.0) == "100,0 %"
    assert V.pct(0.33449, 1) == "33,4 %" and V.pct(0.0106, 2) == "1,06 %"
    assert V.pct(0.99999) == "mehr als 99,9 %"                 # nicht auf 100,0 % gerundet
    assert V.pct(0.9994) == "99,9 %"
    assert V.pct(0.00001) == "weniger als 0,1 %"
    assert V.pct(0.0006) == "0,1 %"
    assert V.pct(0.99999, 0) == "mehr als 99 %" and V.pct(0.99999, 2) == "mehr als 99,99 %"


# ---------------------------------------------------------------------------------------------------------------------------- Tab 1


@pytest.mark.parametrize("game,kind,needle", [
    ((5, 2, 1, 2), "success", "auszahlungs- und risikodominant"),
    ((6, 4, 0, 4), "warning", "**Bündeln** ist auszahlungsdominant, aber **Direkt fahren** ist risikodominant"),
    ((6, 4, 2, 4), "info", "keines der Gleichgewichte ist risikodominant"),
    ((4, 4, 0, 0), "info", "**Gleichauf:**"),
    ((4, 4, 1, 2), "info", "Beide Gleichgewichte zahlen gleich viel (a = b); **Direkt fahren** ist risikodominant"),
    ((-1, 2, 4, 0), "info", "**Anti-Koordination:**"),
    ((3, 1, 2, 0), "info", "dominiert **Bündeln**"),
    ((1, 4, 0, 3), "info", "dominiert **Direkt fahren**"),
    ((3, 2, 1, 3), "warning", "Entartetes Spiel"),
])
def test_verdict_game_branches(game, kind, needle):
    k, text = V.verdict_game(V.analyse_game(game))
    assert k == kind and needle in text


def test_verdict_game_numbers_for_buendeln():
    text = V.verdict_game(V.analyse_game((6, 4, 0, 4)))[1]
    assert "kleineren Einzugsbereich (0,333). p* = 2/3 (0,667). Bündeln lohnt, sobald mehr als dieser Anteil mitmacht; sein Einzugsbereich ist 1 - p* = 0,333." in text


def test_verdict_game_numbers_for_hirsch():
    assert "p* = 1/4 (0,250)" in V.verdict_game(V.analyse_game((5, 2, 1, 2)))[1] and "1 - p* = 0,750" in V.verdict_game(V.analyse_game((5, 2, 1, 2)))[1]


def test_verdict_game_anti_names_slope_and_p_star():
    text = V.verdict_game(V.analyse_game((-1, 2, 4, 0)))[1]
    assert "p* = 2/3 (0,667)" in text and "s = -3" in text


# ---------------------------------------------------------------------------------------------------------------------------- Tab 2


def _pop(rule="Beste Antwort", klass=G.KOORDINATION, exact=0.25, sim=0.3, runs=20, finals=(), boundary=None, start_final=None, n=10, start=50):
    return V.PopulationView(n, start, rule, klass, exact, sim, runs, finals=finals, boundary=boundary, start_final=start_final)


def test_verdict_population_best_response_misses_payoff_dominant():
    kind, text = V.verdict_population(_pop(exact=0.25), (6, 4, 0, 4))
    assert kind == "warning" and "25,0 %" in text and "in den 20 simulierten Läufen 30,0 %" in text and "verfehlt" in text and "n = 10" in text and "mit 50 %" in text


def test_verdict_population_best_response_threshold_at_one_half():
    assert V.verdict_population(_pop(exact=0.5), (6, 4, 0, 4))[0] == "info"           # genau 1/2 ist keine Mehrheit der Verfehlung
    assert V.verdict_population(_pop(exact=0.4999), (6, 4, 0, 4))[0] == "warning"
    assert V.verdict_population(_pop(exact=0.25), (4, 6, 0, 2))[0] == "info"          # Bündeln nicht auszahlungsdominant: kein Verfehlungssatz


def test_verdict_population_best_response_anti():
    kind, text = V.verdict_population(_pop(klass=G.ANTI, finals=(3, 5)), (-1, 2, 4, 0))
    assert kind == "info" and "40,0 %" in text and "10 Betriebe" in text


def test_verdict_population_replicator():
    k, text = V.verdict_population(_pop("Replikator", boundary=0.6667, start_final=0.2), (6, 4, 0, 4))
    assert "0,667" in text and "„alle direkt“" in text and "p* = 0,667" in text
    assert "„alle bündeln“" in V.verdict_population(_pop("Replikator", boundary=0.25, start_final=0.9), (5, 2, 1, 2))[1]
    anti = V.verdict_population(_pop("Replikator", klass=G.ANTI, start_final=0.5), (-1, 2, 4, 0))[1]
    assert "50,0 %" in anti and "kein Koordinationsspiel" in anti


def test_verdict_population_logit_names_range():
    text = V.verdict_population(_pop("Logit", runs=2, finals=(2, 8)), (6, 4, 0, 4))[1]
    assert "2 Beispielläufe" in text and "20,0 % bis 80,0 %" in text and "300 Schritten" in text


# ---------------------------------------------------------------------------------------------------------------------------- Tab 3


def _logit(low=0.9, high=0.01, mean=0.1, peak=3, dev=1.234e-14):
    return V.LogitView(50, 2.0, np.zeros(51), np.zeros(51), dev, low, high, mean, peak, None)


def test_verdict_logit_risk_dominant_branch_names_the_mass():
    kind, text = V.verdict_logit(_logit(), (6, 4, 0, 4))
    assert kind == "warning" and "(Direkt fahren)" in text and "90,0 % der Masse bei höchstens 10 %" in text and "nicht das gewählte" in text and "1,2e-14" in text and "β = 2,0" in text


def test_verdict_logit_when_both_dominances_agree():
    kind, text = V.verdict_logit(_logit(low=0.0, high=0.8), (5, 2, 1, 2))
    assert kind == "info" and "(Bündeln)" in text and "80,0 % der Masse bei mindestens 90 %" in text and "nicht das gewählte" not in text


def test_verdict_logit_tie_anti_and_dominant():
    assert "Gleichauf (p* = 1/2)" in V.verdict_logit(_logit(), (6, 4, 2, 4))[1]
    anti = V.verdict_logit(_logit(mean=0.66, peak=33), (-1, 2, 4, 0))[1]
    assert "66,0 %" in anti and "33 von 50" in anti
    assert "dominanten Rand" in V.verdict_logit(_logit(mean=0.99), (3, 1, 2, 0))[1]


# ---------------------------------------------------------------------------------------------------------------------------- Tab 4


def _roles(r1, r2, share=0.5824):
    shares = {"p_star": 2 / 3, "q_star": 0.25, "share_00": share, "unreached": 0.0, "direct_00": 0.25, "direct_11": 1 / 6}
    return V.RolesView((r1, r2), shares, 100, 0.83, ())


def test_verdict_roles_numbers_and_branches():
    kind, text = V.verdict_roles(_roles(3, 2))
    assert kind == "info" and "(0,667; 0,250)" in text and "58,2 %" in text and "83,0 %" in text and "2500" in text and "0,0 % kommen nirgends an" in text
    assert "3 gegen 2" in text and "sie führen nur 25,0 % aller Anfangspunkte direkt zu „Option 0“" in text
    assert "nahe der Hälfte" in V.verdict_roles(_roles(2, 2))[1] and "stärkere" not in V.verdict_roles(_roles(2, 2))[1]


# ---------------------------------------------------------------------------------------------------------------------------- Tab 5


def _thr(tipped=0.4, stable=2, mean=0.3):
    return V.ThresholdView(30, 15, 0, 100, np.array([0.0, 0.5, 1.0]), mean, 0.044, stable, tipped, {})


def test_verdict_thresholds_branches():
    kind, text = V.verdict_thresholds(_thr())
    assert kind == "warning" and "hängt hier an der Ziehung" in text and "40,0 %" in text and "2 stabile Fixpunkte" in text and "von 0,0 % bis 100,0 %" in text and "Mittel 30 %" in text
    assert V.verdict_thresholds(_thr(tipped=0.0))[0] == "info" and V.verdict_thresholds(_thr(tipped=1.0))[0] == "info"
    assert V.verdict_thresholds(_thr(tipped=0.01))[0] == "warning"
    assert "1 stabilen Fixpunkt" in V.verdict_thresholds(_thr(stable=1))[1]


# ---------------------------------------------------------------------------------------------------------------------------- Tab 6


def test_verdict_anti_each_class():
    base = {"slope": 6, "p_star": 2 / 3, "near": 0.0, "edge": 1.0, "game": (6, 4, 0, 4)}
    coord = V.AntiView({**base, "class": G.KOORDINATION}, F(4), F(6), 50, (), 0.5)
    assert "instabil" in V.verdict_anti(coord)[1] and "100,0 % an einem Rand" in V.verdict_anti(coord)[1] and "s = 6 > 0" in V.verdict_anti(coord)[1]
    anti = V.AntiView({**base, "slope": -3, "near": 1.0, "edge": 0.0, "class": G.ANTI}, F(2, 3), F(2), 50, (33,), 0.6)
    text = V.verdict_anti(anti)[1]
    assert "**stabil**" in text and "100,0 % nahe p*" in text and "2/3 (0,667)" in text and "bei bester Absprache 2" in text and "s = -3 < 0" in text
    dom = V.AntiView({**base, "class": G.DOMINANT, "p_star": None}, None, F(6), 50, (), 0.5)
    assert "ohne zweites Gleichgewicht" in V.verdict_anti(dom)[1]


# ---------------------------------------------------------------------------------------------------------------------------- Experimente


def test_verdict_e1_random_numbers():
    rows = [{"mode": "allgemein", "draws": 100000, "seeds": 4, "games": 25033.5, "risk_dominated": 0.3344, "risk_dominated_std": 0.0026, "mean_basin": 0.59, "mean_basin_std": 0.0017},
            {"mode": "sicherer Hase", "draws": 100000, "seeds": 4, "games": 16678.0, "risk_dominated": 0.4995, "risk_dominated_std": 0.0019, "mean_basin": 0.5005, "mean_basin_std": 0.0007}]
    kind, text = V.verdict_e1_random(rows)
    assert kind == "warning" and "33,4 %" in text and "0,3 %" in text and "0,590" in text and "sind es 50,0 %." in text and "25.034 Spiele je Strom" in text and "4 Ströme" in text
    assert "mit anderer Verteilung sind es andere Zahlen" in text


def test_verdict_e1_start_names_the_second_game():
    rows = [{"game": (5, 2, 1, 2)}, {"game": (6, 4, 0, 4), "p_star": 2 / 3, "all_s": 0.0106, "n": 50, "start": 50}]
    text = V.verdict_e1_start(rows)[1]
    assert "(6, 4, 0, 4)" in text and "p* = 0,667" in text and "1,06 %" in text and "n = 50" in text and "mit 50 %" in text
    assert "Der Anfangsanteil 50 % liegt unter p* = 0,667" in text and "1 - p* = 0,333" in text


def test_verdict_e2_counts_games_within_the_tolerance():
    rows = [{"game": (5, 2, 1, 2), "p_star": 0.25, "replicator": 0.2500, "best_response": 0.25, "n": 100},
            {"game": (6, 4, 0, 4), "p_star": 2 / 3, "replicator": 0.6667, "best_response": 0.67, "n": 100},
            {"game": (1, 1, 0, 0), "p_star": 0.5, "replicator": 0.52, "best_response": 0.5, "n": 100}]
    text = V.verdict_e2(rows)[1]
    assert "In 2 von 3 Spielen" in text and "in 3 von 3" in text and "0,02000" in text and "0,003" in text
    close = [{"game": g, "p_star": 0.5, "replicator": 0.5, "best_response": 0.51, "n": 100} for g in ((1, 1, 0, 0),)]
    assert "in 1 von 1" in V.verdict_e2(close)[1]                     # Abweichung genau 0,01 gilt als innerhalb


def test_verdict_e3_reports_worst_cell():
    rows = [{"game": (5, 2, 1, 2), "worst": 2.5e-14, "cells": 30}, {"game": (6, 4, 0, 4), "worst": 3.5e-14, "cells": 30}]
    text = V.verdict_e3(rows)[1]
    assert "2 Spiele und 60 Zellen" in text and "3,5e-14" in text


def test_verdict_e4_lists_every_row():
    rows = [{"roles": (2, 2), "share_00": 0.49, "br_share_00": 0.5, "br_runs": 800}, {"roles": (3, 2), "share_00": 0.5824, "br_share_00": 0.8175, "br_runs": 800}]
    text = V.verdict_e4(rows)[1]
    assert "Rollen 2 gegen 2: Replikator 49,0 %, beste Antwort 50,0 %" in text and "Rollen 3 gegen 2: Replikator 58,2 %, beste Antwort 81,8 %" in text and "800 Läufe" in text and "2500 Anfangspunkte" in text


def _e5rows():
    rows = []
    for sigma, mean, fp, inner in ((10, 0.0014, 0.0014, 0.0012), (15, 0.0924, 0.0438, 0.1754), (20, 0.9997, 0.9998, 0.0004)):
        rows.append({"mu": 30, "s0": 0, "sigma": sigma, "mean": mean, "fixed_point": fp, "inner_std": inner, "draws": 100})
    return rows


def test_verdict_e5_names_three_sigmas():
    text = V.verdict_e5(_e5rows())[1]
    assert text.startswith("Bei mittlerer Schwelle 30 %") and "Bei dieser mittleren Schwelle öffnet mehr Streuung" in text and "Streuung 10 % bei 0,1 %" in text and "15 % bei 9,2 %" in text and "(Fixpunkt 4,4 %" in text and "17,5 %" in text and "20 % bei mehr als 99,9 %" in text and "100 Ziehungen" in text


def test_verdict_e6_lists_both_classes_and_welfare():
    rows = [{"game": (5, 2, 1, 2), "class": G.KOORDINATION, "near": 0.0, "edge": 1.0}, {"game": (-1, 2, 4, 0), "class": G.ANTI, "near": 1.0, "edge": 0.0},
            {"game": (0, 3, 4, 2), "class": G.ANTI, "near": 1.0, "edge": 0.0}]
    text = V.verdict_e6(rows)[1]
    assert "Spiele (5, 2, 1, 2): von 1000 Anfangsanteilen enden 100 % an einem Rand, 0 % nahe p*" in text
    assert "Spiele (-1, 2, 4, 0), (0, 3, 4, 2): 100 %, 100 % enden nahe p*, 0 %, 0 % an einem Rand" in text
    assert "2/3 (0,667) je Spieler, eine abwechselnde Absprache 2" in text and "8/3 (2,667) gegen 3" in text


# ---------------------------------------------------------------------------------------------------------------------------- Tabelle


def test_stability_labels():
    assert V.stability_label((5, 2, 1, 2), F(1), F(1)) == "stabil"            # a > d
    assert V.stability_label((5, 2, 1, 2), F(0), F(0)) == "stabil"            # c < b
    assert V.stability_label((5, 2, 1, 2), F(1, 4), F(1, 4)) == "instabil"    # s > 0
    assert V.stability_label((-1, 2, 4, 0), F(2, 3), F(2, 3)) == "stabil"     # s < 0
    assert V.stability_label((-1, 2, 4, 0), F(1), F(1)) == "instabil"         # a < d: Bündeln-Rand nicht stabil
    assert V.stability_label((3, 1, 1, 3), F(0), F(0)) == "unbestimmt"        # c = b
    assert V.stability_label((-1, 2, 4, 0), F(1), F(0)).startswith("– (asymmetrisch")


def test_equilibrium_table_rows_for_hirsch():
    rows = V.equilibrium_table(V.analyse_game((5, 2, 1, 2)))
    assert [r["Art"] for r in rows].count("rein") == 2 and len(rows) == 3
    mixed = next(r for r in rows if r["Art"] == "gemischt")
    assert mixed["Auszahlung Zeile / Spalte"] == "2 / 2" and mixed["Stabilität (Replikator, eine Population)"] == "instabil"
