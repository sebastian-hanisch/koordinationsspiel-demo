"""Abbildungen: jede Figur lässt sich für jede Spielklasse bauen, alle Achsen sind gesperrt, Beschriftungen stehen auf Deutsch (Dezimalkomma)."""

import numpy as np
import plotly.graph_objects as go
import pytest

import kos_constants as C
import kos_evaluation as V
import kos_experiments as X
import kos_games as G
import kos_visualization as Z

GAMES = [(6, 4, 0, 4), (5, 2, 1, 2), (-1, 2, 4, 0), (3, 1, 2, 0), (3, 2, 1, 3), (0, 0, 0, 0)]


def _locked(fig):
    assert isinstance(fig, go.Figure)
    for axis in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        assert axis.fixedrange is True


@pytest.mark.parametrize("game", GAMES)
def test_game_figures_for_every_class(game):
    gv = V.analyse_game(game)
    _locked(Z.build_matrix(game))
    _locked(Z.build_payoff_lines(game, gv.p_star))
    _locked(Z.build_flow(game))
    _locked(Z.build_potential(game, 10))
    _locked(Z.build_logit(V.analyse_logit(game, 10, 1.0)))


def test_matrix_marks_exactly_the_pure_equilibria():
    fig = Z.build_matrix((6, 4, 0, 4))
    thick = [s for s in fig.layout.shapes if s.line.width == 4]
    assert len(thick) == 2                                                      # beide bündeln, beide direkt
    assert len(Z.build_matrix((-1, 2, 4, 0)).layout.shapes) == 4
    assert len([s for s in Z.build_matrix((-1, 2, 4, 0)).layout.shapes if s.line.width == 4]) == 2       # die zwei asymmetrischen
    assert len([s for s in Z.build_matrix((3, 1, 2, 0)).layout.shapes if s.line.width == 4]) == 1        # dominant: eines


def test_payoff_lines_cross_at_p_star():
    fig = Z.build_payoff_lines((6, 4, 0, 4), G.p_star(6, 4, 0, 4))
    s_line, h_line = fig.data[0], fig.data[1]
    i = int(np.argmin(np.abs(np.array(s_line.y) - np.array(h_line.y))))
    assert s_line.x[i] == pytest.approx(2 / 3, abs=0.006)
    assert fig.data[2].x[0] == pytest.approx(2 / 3)


def test_population_figures():
    gv = V.analyse_game((6, 4, 0, 4))
    _locked(Z.build_basin_bar(gv.p_star, 50))
    for rule in C.RULES:
        pv = V.analyse_population((6, 4, 0, 4), 20, 50, rule, 1.0, 12, 3)
        if rule == "Replikator":
            _locked(Z.build_fan(pv.fan, gv.p_star))
        else:
            _locked(Z.build_paths(pv.paths, 20, gv.p_star))
        if rule == "Beste Antwort":
            _locked(Z.build_finals(pv.finals, 20))


def test_basin_bar_widths_sum_to_one():
    fig = Z.build_basin_bar(G.p_star(5, 2, 1, 2), 30)
    assert fig.data[0].x[0] + fig.data[1].x[0] == pytest.approx(1.0) and fig.data[2].x[0] == 0.3


def test_roles_and_threshold_figures():
    _locked(Z.build_phase(V.analyse_roles(3, 2, 10, 1)))
    tv = V.analyse_thresholds(30, 15, 0, 10, 1)
    _locked(Z.build_threshold_cdf(30, 15, 0))
    fig = Z.build_threshold_finals(tv.finals)
    _locked(fig)
    assert sum(fig.data[0].y) == 10 and "," in fig.data[0].x[0] and "." not in fig.data[0].x[0]


def test_threshold_cdf_curve_starts_at_the_initial_share():
    fig = Z.build_threshold_cdf(50, 5, 20)
    assert fig.data[0].y[0] == pytest.approx(0.2)                            # bei Stand 0 machen mindestens die 20 % mit


def test_experiment_figures():
    _locked(Z.build_random_games(X.e1_random_games(draws=2000)))
    _locked(Z.build_sigma_sweep(X.e5_thresholds(sigmas=(10, 20), cells=((30, 0),), draws=3, seeds=(1,)), 30, 0))
