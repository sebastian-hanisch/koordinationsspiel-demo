"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import kos_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None      # bei Auswahlreglern: die erlaubten Werte


B = C.BASE
SETTING_SPECS = {
    "a_slider": SettingSpec("a", int, B["a"], C.PAY_MIN, C.PAY_MAX),
    "b_slider": SettingSpec("b", int, B["b"], C.PAY_MIN, C.PAY_MAX),
    "c_slider": SettingSpec("c", int, B["c"], C.PAY_MIN, C.PAY_MAX),
    "d_slider": SettingSpec("d", int, B["d"], C.PAY_MIN, C.PAY_MAX),
    "n_slider": SettingSpec("n", int, B["n"], C.N_MIN, C.N_MAX),
    "start_slider": SettingSpec("start", int, B["start"], C.START_MIN, C.START_MAX),
    "rule_select": SettingSpec("rule", str, B["rule"], choices=C.RULES),
    "beta_slider": SettingSpec("beta", float, B["beta"], C.BETA_MIN, C.BETA_MAX),
    "runs_slider": SettingSpec("runs", int, B["runs"], C.RUNS_MIN, C.RUNS_MAX),
    "role1_slider": SettingSpec("r1", int, B["role1"], C.ROLE_MIN, C.ROLE_MAX),
    "role2_slider": SettingSpec("r2", int, B["role2"], C.ROLE_MIN, C.ROLE_MAX),
    "mu_slider": SettingSpec("mu", int, B["mu"], C.MU_MIN, C.MU_MAX),
    "sigma_slider": SettingSpec("sigma", int, B["sigma"], C.SIGMA_MIN, C.SIGMA_MAX),
    "s0_slider": SettingSpec("s0", int, B["s0"], C.S0_MIN, C.S0_MAX),
    "seed_input": SettingSpec("seed", int, B["seed"], 0, C.SEED_MAX),
}
PRESET_KEYS = {
    "a": "a_slider", "b": "b_slider", "c": "c_slider", "d": "d_slider", "n": "n_slider", "start": "start_slider", "rule": "rule_select", "beta": "beta_slider",
    "runs": "runs_slider", "role1": "role1_slider", "role2": "role2_slider", "mu": "mu_slider", "sigma": "sigma_slider", "s0": "s0_slider", "seed": "seed_input",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf erlaubte Werte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.choices is not None:
                if value not in spec.choices:
                    continue
            else:
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
