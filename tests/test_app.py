"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_both_play_buttons_work_individually_and_show_their_tables():
    """Jeder ▶️-Knopf einzeln: erst nach dem Klick erscheint die Tabelle samt Verdikt (die Smoke-Schleife oben lässt sie aus)."""
    cases = (("e2_button", "e2_done", "Grenze Replikator"), ("e4_button", "e4_done", "Replikator: Anteil „beide Option 0“"))
    for button, done, column in cases:
        at = _fresh_app()
        assert not any(column in df.value.columns.tolist() for df in at.dataframe), "vor dem Klick darf die Tabelle nicht stehen"
        at.button(key=button).click().run(timeout=TIMEOUT)
        assert not at.exception, (button, [str(e) for e in at.exception])
        assert at.session_state[done] is True
        assert any(column in df.value.columns.tolist() for df in at.dataframe), button


def test_play_button_result_survives_the_next_rerun():
    at = _fresh_app()
    at.button(key="e2_button").click().run(timeout=TIMEOUT)
    at.slider(key="n_slider").set_value(60).run(timeout=TIMEOUT)
    assert not at.exception and any("Grenze Replikator" in df.value.columns.tolist() for df in at.dataframe)


def test_rule_selectbox_every_option_renders():
    for rule in ("Beste Antwort", "Replikator", "Logit"):
        at = _fresh_app()
        at.selectbox(key="rule_select").set_value(rule).run(timeout=TIMEOUT)
        assert not at.exception, (rule, [str(e) for e in at.exception])
        assert any(rule in m.value for m in at.markdown), rule


def test_default_state_shows_the_six_tabs_and_the_hook_verdict():
    at = _fresh_app()
    assert [t.label for t in at.tabs] == ["1 Zwei Betriebe", "2 Viele Betriebe", "3 Logit und Potential", "4 Zwei Rollen", "5 Heterogene Schwellen", "6 Wenn Gleichwahl bestraft wird"]
    warnings = " ".join(w.value for w in at.warning)
    assert "**Bündeln** ist auszahlungsdominant, aber **Direkt fahren** ist risikodominant" in warnings


def test_permalink_sets_the_game_and_is_limited_to_the_bounds():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["a"] = "99"
    at.query_params["rule"] = "Logit"
    at.query_params["n"] = "0"
    at.run()
    assert not at.exception
    assert at.slider(key="a_slider").value == 10 and at.selectbox(key="rule_select").value == "Logit" and at.slider(key="n_slider").value == 2


def test_permalink_ignores_an_unknown_rule_and_garbage():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["rule"] = "Zufall"
    at.query_params["b"] = "viel"
    at.run()
    assert not at.exception
    assert at.selectbox(key="rule_select").value == "Beste Antwort" and at.slider(key="b_slider").value == 4


def test_pdf_button_creates_a_pdf_and_offers_the_download():
    at = _fresh_app()
    at.button(key="pdf_button").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.session_state["pdf_bytes"][:5] == b"%PDF-"


def test_every_preset_shows_a_tab_one_verdict_of_the_expected_kind():
    expected = {"Bündeln am Hub (6, 4, 0, 4)": "warning", "Hirschjagd klassisch (5, 2, 1, 2)": "success", "Gleichauf (6, 4, 2, 4)": "info", "Reine Koordination (3, 1, 0, 0)": "success",
                "Feiglingsspiel (V = 4, C = 6)": "info"}
    for name, kind in expected.items():
        at = _fresh_app()
        at.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, name
        texts = {"warning": [w.value for w in at.warning], "success": [s.value for s in at.success], "info": [i.value for i in at.info]}[kind]
        assert any("dominant" in t or "Anti-Koordination" in t or "Gleichauf" in t or "keines" in t for t in texts), (name, kind)


def test_all_sliders_at_their_minima_and_all_at_their_maxima_do_not_raise():
    """Eckfall-Matrix (DEMO-PLAYBOOK §1d): alle Regler zugleich am Minimum, dann alle am Maximum (n = 200, Läufe 300, β = 10 ...), alle drei Lernregeln."""
    for edge in ("min", "max"):
        for rule in ("Beste Antwort", "Replikator", "Logit"):
            at = _fresh_app()
            for slider in at.slider:
                value = slider.min if edge == "min" else slider.max
                slider.set_value(int(value) if isinstance(slider.value, int) else value)
            at.selectbox(key="rule_select").set_value(rule)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (edge, rule, [str(e) for e in at.exception])
