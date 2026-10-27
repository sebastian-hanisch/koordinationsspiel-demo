"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen, gerechnete Aussagen der Hilfetexte."""

from fractions import Fraction as F

import kos_constants as C
import kos_evaluation as V
import kos_games as G
import kos_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert V.Settings(**C.PRESETS[first]) == V.Settings()
    assert first.startswith("Bündeln am Hub")


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_two_presets_are_identical():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_help_text_statements_about_p_star_and_dominance():
    """Die Hilfetexte nennen p* und Dominanzen; hier gegen die Rechnung."""
    g = {name: (p["a"], p["b"], p["c"], p["d"]) for name, p in C.PRESETS.items()}
    assert G.p_star(*g["Bündeln am Hub (6, 4, 0, 4)"]) == F(2, 3) and G.risk_dominant(*g["Bündeln am Hub (6, 4, 0, 4)"]) == "H" and G.payoff_dominant(*g["Bündeln am Hub (6, 4, 0, 4)"]) == "S"
    assert G.p_star(*g["Hirschjagd klassisch (5, 2, 1, 2)"]) == F(1, 4) and G.risk_dominant(5, 2, 1, 2) == "S" and G.payoff_dominant(5, 2, 1, 2) == "S"
    assert G.p_star(*g["Gleichauf (6, 4, 2, 4)"]) == F(1, 2) and G.risk_dominant(6, 4, 2, 4) is None and C.PRESETS["Gleichauf (6, 4, 2, 4)"]["a"] > C.PRESETS["Gleichauf (6, 4, 2, 4)"]["b"]
    assert G.p_star(*g["Reine Koordination (3, 1, 0, 0)"]) == F(1, 4)
    assert G.classify(*g["Feiglingsspiel (V = 4, C = 6)"]) == G.ANTI and G.p_star(-1, 2, 4, 0) == F(4, 6)       # V/C
    # Feiglingsspiel aus V und C: a = (V - C)/2, b = V/2, c = V, d = 0
    V_, C_ = 4, 6
    assert (g["Feiglingsspiel (V = 4, C = 6)"]) == ((V_ - C_) // 2, V_ // 2, V_, 0)


def test_granovetter_preset_changes_only_the_spread():
    base, preset = C.PRESETS[next(iter(C.PRESETS))], C.PRESETS["Schwellen streuen (Granovetter)"]
    assert {k for k in base if base[k] != preset[k]} == {"sigma"} and (base["sigma"], preset["sigma"]) == (15, 20) and preset["mu"] == 30 and preset["s0"] == 0


def test_role_preset_is_symmetric_and_differs_from_default_in_the_roles_only():
    base, preset = C.PRESETS[next(iter(C.PRESETS))], C.PRESETS["Kampf der Geschlechter (zwei Rollen)"]
    assert {k for k in base if base[k] != preset[k]} == {"role1"} and preset["role1"] == preset["role2"] == 2


def test_permalink_params_are_unique():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_slider_ranges_contain_the_feigling_payoffs():
    assert C.PAY_MIN <= -1 and C.PAY_MAX >= 4
