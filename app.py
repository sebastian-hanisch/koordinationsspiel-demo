"""Koordinationsspiele: wenn Gleichwahl belohnt wird - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Erstes Stück der Reihe Koordinationsspiele. Zwei Betriebe oder viele wählen zwischen Bündeln am Hub und direkt fahren; die Demo zeigt die Gleichgewichte, die Einzugsbereiche,
den Start ohne Absprache, Logit und Potential, zwei Rollen, verteilte Schwellen und den Gegenfall (Anti-Koordination).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kos_constants as C
import kos_evaluation as V
import kos_experiments as X
import kos_games as G
import kos_visualization as Z
from kos_layout import dunkles_design, kopf_css
from kos_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Koordinationsspiele – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _game(game):
    return V.analyse_game(game)


@st.cache_data(show_spinner=False)
def _population(game, n, start, rule, beta, runs, seed):
    return V.analyse_population(game, n, start, rule, beta, runs, seed)


@st.cache_data(show_spinner=False)
def _logit(game, n, beta):
    return V.analyse_logit(game, n, beta)


@st.cache_data(show_spinner=False)
def _roles(role1, role2, runs, seed):
    return V.analyse_roles(role1, role2, runs, seed)


@st.cache_data(show_spinner=False)
def _thresholds(mu, sigma, s0, draws, seed):
    return V.analyse_thresholds(mu, sigma, s0, draws, seed)


@st.cache_data(show_spinner=False)
def _anti(game, n, beta):
    return V.analyse_anti(game, n, beta)


@st.cache_data(show_spinner=False)
def _e1():
    return X.e1_random_games(), X.e1_start_without_agreement()


@st.cache_data(show_spinner=False)
def _e2():
    return X.e2_basins()


@st.cache_data(show_spinner=False)
def _e3():
    return X.e3_logit_vs_gibbs()


@st.cache_data(show_spinner=False)
def _e3_masses(n, game):
    return X.e3_masses(n, game)


@st.cache_data(show_spinner=False)
def _e4():
    return X.e4_two_roles()


@st.cache_data(show_spinner=False)
def _e5():
    return X.e5_thresholds()


@st.cache_data(show_spinner=False)
def _e6():
    return X.e6_anti()


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kos_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Koordinationsspiele: wenn Gleichwahl belohnt wird")
st.markdown(
    """
**Erstes Stück der Reihe Koordinationsspiele, ihre Wurzel.** Betriebe entscheiden unabhängig voneinander, ob sie **am Hub bündeln** oder **direkt fahren**. Bündeln lohnt sich nur, wenn genug andere mitmachen;
wer allein bündelt, verliert Zeit. Daraus folgen mehrere Gleichgewichte, und die Fragen sind: **Welche gibt es, in welches läuft man, und ist das bessere auch das gewählte?**
Im Überlastungsspiel der nash-demo meiden die Lkw dasselbe Tor; hier wird Gleichwahl belohnt, und der Gegenfall steht im letzten Tab.

**Eigennutz und die Lernregeln (beste Antwort, Replikator, Logit) sind Modellannahmen**, keine Aussagen über Menschen oder Terminals. Die Auszahlungen (Zeitersparnis je Sendung in Minuten) sind gewählt, nicht gemessen.

Weiter unten: sechs Tabs mit je einem **🔬 Experiment**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert ein Koordinationsspiel", expanded=True):
    st.markdown(
        """
1. **Die Auszahlungsmatrix.** Jeder Betrieb wählt **S = bündeln** oder **H = direkt fahren**. Die Auszahlung für mich ist **a**, wenn beide bündeln, **b**, wenn beide direkt fahren, **c**, wenn ich bündele und der andere direkt fährt, und **d**, wenn ich direkt fahre und der andere bündelt.
2. **Beste Antworten.** Im Koordinationsspiel ist Bündeln besser als Direktfahren, sobald der Anteil der Bündelnden unter den anderen über **p\\* = (b − c) / ((a − d) + (b − c))** liegt. Die Steigung s = (a − d) + (b − c) entscheidet, ob das ein Koordinationsspiel ist.
3. **Vier Klassen.** *Koordination* (a > d und b > c): zwei stabile Gleichgewichte (beide bündeln, beide direkt) und ein instabiles gemischtes. *Anti-Koordination* (a < d und b < c): Gleichwahl wird bestraft. *Dominant*: eine Strategie ist immer besser, es gibt nichts zu koordinieren. *Entartet*: ein Gleichstand (a = d oder b = c).

| Begriff | Bedeutung | Wo gezeigt |
|---|---|---|
| **Auszahlungsdominanz** | Das Gleichgewicht, in dem beide am meisten bekommen (a gegen b). | Tab 1 |
| **Risikodominanz** | Das Gleichgewicht mit dem größeren Einzugsbereich: Weicht der andere ab, verliere ich dort weniger (a − d gegen b − c). | Tab 1, Tab 2 |
| **Einzugsbereich** | Die Anfangsanteile, von denen aus eine Lernregel bei diesem Gleichgewicht endet. Für „alle bündeln“ ist das der Bereich oberhalb von p\\*, seine Größe 1 − p\\*. | Tab 2 |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🎮 Das Spiel")
    st.caption("Zeitersparnis je Sendung in Minuten; negativ heißt Zeit verloren.")
    a = st.slider("a: beide bündeln", *bounds("a_slider"), key="a_slider", help="Meine Auszahlung, wenn beide bündeln.")
    b = st.slider("b: beide direkt", *bounds("b_slider"), key="b_slider", help="Meine Auszahlung, wenn beide direkt fahren.")
    c = st.slider("c: ich bündele, der andere fährt direkt", *bounds("c_slider"), key="c_slider", help="Wer allein bündelt, bekommt oft wenig oder verliert Zeit.")
    d = st.slider("d: ich fahre direkt, der andere bündelt", *bounds("d_slider"), key="d_slider", help="Meine Auszahlung, wenn ich direkt fahre und der andere bündelt.")
    st.header("👥 Population und Lernregel")
    n = st.slider("Betriebe n", *bounds("n_slider"), key="n_slider", help="Wie viele Betriebe sich vollständig durchmischt begegnen (jeder trifft jeden gleich oft).")
    start = st.slider("Anfangsanteil der Bündelnden (%)", *bounds("start_slider"), key="start_slider",
                      help="Beim Start ohne Absprache entscheidet jeder Betrieb unabhängig mit dieser Wahrscheinlichkeit für Bündeln.")
    rule = st.selectbox("Lernregel", C.RULES, key="rule_select", help="Beste Antwort: ein zufällig gezogener Betrieb wechselt, wenn das strikt besser ist. Replikator: Anteile wachsen mit dem Auszahlungsvorsprung. Logit: gute Wahl ist wahrscheinlicher, nie sicher.")
    beta = st.slider("Schärfe β (Logit)", *bounds("beta_slider"), key="beta_slider", step=0.1, help="Je größer β, desto sicherer wählt ein Betrieb die bessere Strategie. Wirkt in Tab 3 und bei der Lernregel Logit.")
    runs = st.slider("Läufe", *bounds("runs_slider"), key="runs_slider", help="Wiederholungen der Zufallsläufe (beste Antwort in Tab 2, Zwei Rollen in Tab 4, Ziehungen der Schwellen in Tab 5).")
    st.header("🎭 Zwei Rollen (Tab 4)")
    role1 = st.slider("Wert der Wunschoption von Rolle 1", *bounds("role1_slider"), key="role1_slider", help="Rolle 1 zieht Option 0 vor (Wert gegen 1 für Option 1), Einigung auf die andere zahlt 1, Nichteinigung 0.")
    role2 = st.slider("Wert der Wunschoption von Rolle 2", *bounds("role2_slider"), key="role2_slider", help="Rolle 2 zieht Option 1 vor (Wert gegen 1 für Option 0).")
    st.header("📊 Schwellen (Tab 5)")
    mu = st.slider("Mittlere Schwelle (%)", *bounds("mu_slider"), key="mu_slider", help="Wie viele andere schon mitmachen müssen, damit eine durchschnittliche Person mitmacht.")
    sigma = st.slider("Streuung der Schwellen (%)", *bounds("sigma_slider"), key="sigma_slider", help="Wie verschieden die Schwellen der Personen sind.")
    s0 = st.slider("Anteil, der von sich aus mitmacht (%)", *bounds("s0_slider"), key="s0_slider", help="Sie machen mit, auch wenn niemand sonst mitmacht.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"a_slider": int(a), "b_slider": int(b), "c_slider": int(c), "d_slider": int(d), "n_slider": int(n), "start_slider": int(start), "rule_select": rule, "beta_slider": float(beta),
                   "runs_slider": int(runs), "role1_slider": int(role1), "role2_slider": int(role2), "mu_slider": int(mu), "sigma_slider": int(sigma), "s0_slider": int(s0), "seed_input": int(seed)})

settings = V.Settings(int(a), int(b), int(c), int(d), int(n), int(start), rule, float(beta), int(runs), int(seed), int(role1), int(role2), int(mu), int(sigma), int(s0))
game = settings.game
gv = _game(game)
verdicts = []   # (Tab, Verdikt) für den PDF-Export

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1 Zwei Betriebe", "2 Viele Betriebe", "3 Logit und Potential", "4 Zwei Rollen", "5 Heterogene Schwellen", "6 Wenn Gleichwahl bestraft wird"])

# --- Tab 1 ---------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.markdown("## 🎯 Zwei Betriebe: Matrix, beste Antworten, Gleichgewichte")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(Z.build_matrix(game), width="stretch", key="matrix_chart")
        st.caption("Feld: (meine Auszahlung | die des anderen). Fett: beste Antwort auf die Wahl des anderen. Grüner Rahmen: reines Gleichgewicht (beide wählen eine beste Antwort).")
    with col2:
        st.plotly_chart(Z.build_payoff_lines(game, gv.p_star), width="stretch", key="payoff_lines_chart")
        st.caption("Wo sich die Linien kreuzen, ist Bündeln genauso gut wie Direktfahren: p*. Links davon ist Direkt besser, rechts davon Bündeln (Koordinationsspiel).")
    m1, m2, m3, m4 = st.columns([1.7, 1.4, 1, 1])
    m1.metric("Klasse", gv.klass)
    m2.metric("p*", "–" if gv.p_star is None else f"{V.frac(gv.p_star)} ({V.de(float(gv.p_star), 3)})")
    m3.metric("Steigung s", gv.slope)
    m4.metric("Gleichgewichte", len(gv.equilibria))
    st.markdown("#### ❓ Kernfrage: Ist das auszahlungsdominante Gleichgewicht auch das risikodominante?")
    v1 = V.verdict_game(gv)
    show(v1)
    verdicts.append(v1)
    st.dataframe(V.equilibrium_table(gv), hide_index=True, width="stretch")
    st.caption("Reine Gleichgewichte: niemand will einseitig wechseln. Gemischt: Jeder bündelt mit der angegebenen Wahrscheinlichkeit, und beide Strategien zahlen gleich viel. Bei Gleichstand (entartet) kann die Liste unvollständig sein.")

# --- Tab 2 ---------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.markdown("## 👥 Viele Betriebe: Einzugsbereich, Start ohne Absprache")
    pv = _population(game, settings.n, settings.start, settings.rule, settings.beta, settings.runs, settings.seed)
    if gv.klass == G.KOORDINATION:
        st.plotly_chart(Z.build_basin_bar(gv.p_star, settings.start), width="stretch", key="basin_bar")
        st.caption("Der Einzugsbereich: Von Anfangsanteilen links von p* laufen Replikator und beste Antwort zu „alle direkt“, von rechts zu „alle bündeln“ (bei der besten Antwort bleibt an der Grenze ein Wechselzustand, von dem aus beide Richtungen möglich sind). Die Raute ist der gewählte Anfangsanteil.")
    st.markdown(f"**Lernregel: {settings.rule}**")
    if settings.rule == "Beste Antwort":
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(Z.build_paths(pv.paths, settings.n, gv.p_star, "Schritt (gezogene Betriebe)"), width="stretch", key="br_paths")
            st.caption("Die ersten bis zu acht Läufe. Der Start ist zufällig (jeder Betrieb unabhängig), jeder Schritt zieht einen Betrieb; er wechselt nur bei strikter Verbesserung.")
        with c2:
            st.plotly_chart(Z.build_finals(pv.finals, settings.n), width="stretch", key="br_finals")
            st.caption(f"Endzustände von {pv.sim_runs} Läufen; mittlere Zahl der Schritte bis zum Stillstand: {V.de(pv.mean_steps, 0)}.")
    elif settings.rule == "Replikator":
        st.plotly_chart(Z.build_fan(pv.fan, gv.p_star if gv.klass == G.KOORDINATION else None), width="stretch", key="fan_chart")
        st.caption("Jede Linie ist ein Anfangsanteil; die Dynamik ist deterministisch (Runge-Kutta). Grün endet bei „alle bündeln“, orange bei „alle direkt“.")
    else:
        st.plotly_chart(Z.build_paths(pv.paths, settings.n, gv.p_star if gv.klass == G.KOORDINATION else None, "Schritt (gezogene Betriebe)"), width="stretch", key="logit_paths")
        st.caption(f"{C.LOGIT_PATHS} Beispielläufe mit β = {V.de(settings.beta, 1)}: Ein gezogener Betrieb wählt Bündeln mit der logistischen Wahrscheinlichkeit seines Auszahlungsvorsprungs.")
    v2 = V.verdict_population(pv, game)
    show(v2)
    verdicts.append(v2)
    if settings.rule == "Beste Antwort" and gv.klass == G.KOORDINATION:
        st.caption(f"Exakte Wahrscheinlichkeit für „alle bündeln“ aus der Kette über die Zahl der Bündelnden (Brüche): {V.pct(pv.exact_all_s, 2)}; die simulierten Läufe weichen im Zufall davon ab.")

    st.markdown("---")
    st.subheader("🔬 Experiment E1: Wie oft ist das bessere Gleichgewicht nicht das gewählte?")
    st.caption(f"Zufallsspiele: a, b, c, d unabhängig gleichverteilt auf [0, 10), {len(C.SWEEP_SEEDS)} feste Ströme zu je {C.RANDOM_GAME_DRAWS:,} Ziehungen".replace(",", ".")
               + ", nur Koordinationsspiele mit a ≠ b. Dazu der Start ohne Absprache im Bündeln-Spiel.")
    e1_random, e1_start = _e1()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(Z.build_random_games(e1_random), width="stretch", key="e1_chart")
    with c2:
        st.dataframe([{"Ziehung": r["mode"], "Spiele je Strom": f"{r['games']:.0f}", "ausz.-dominant, aber risikodominiert": V.pct(r["risk_dominated"]), "Streuung über Ströme": V.pct(r["risk_dominated_std"]),
                       "mittlerer Einzugsbereich": V.de(r["mean_basin"], 3)} for r in e1_random], hide_index=True, width="stretch")
    show(V.verdict_e1_random(e1_random))
    st.dataframe([{"Spiel (a, b, c, d)": str(r["game"]), "p*": V.de(r["p_star"], 3), "auszahlungsdominant": V.NAMES.get(r["payoff_dominant"], "–"), "risikodominant": V.NAMES.get(r["risk_dominant"], "–"),
                   f"alle bündeln (n = {r['n']}, Start {r['start']} %)": V.pct(r["all_s"], 2)} for r in e1_start], hide_index=True, width="stretch")
    show(V.verdict_e1_start(e1_start))

    st.subheader("🔬 Experiment E2: Liegt die Grenze des Einzugsbereichs bei p*?")
    st.caption("Vier Koordinationsspiele: Grenze unter dem Replikator (gerechnet, Bisektion über Anfangsanteile) und unter der besten Antwort bei n = 100 (exakt aus der Kette). Rechnet etwa eine Sekunde.")
    if st.button("▶️ Einzugsbereiche rechnen", key="e2_button"):
        st.session_state["e2_done"] = True
    if st.session_state.get("e2_done"):
        with st.spinner("Rechne die Grenzen …"):
            e2 = _e2()
        st.dataframe([{"Spiel (a, b, c, d)": str(r["game"]), "p*": V.de(r["p_star"], 4), "Grenze Replikator": V.de(r["replicator"], 4), f"Grenze beste Antwort (n = {r['n']})": V.de(r["best_response"], 3)} for r in e2],
                     hide_index=True, width="stretch")
        show(V.verdict_e2(e2))

# --- Tab 3 ---------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.markdown("## 📈 Logit-Lernen maximiert das Potential")
    lv = _logit(game, settings.n, settings.beta)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(Z.build_logit(lv), width="stretch", key="logit_chart")
        st.caption(f"Stationäre Verteilung bei n = {settings.n} und β = {V.de(settings.beta, 1)}: Balken aus den Übergangswahrscheinlichkeiten der Kette, Kreuze aus dem Gibbs-Maß des Potentials.")
    with c2:
        st.plotly_chart(Z.build_potential(game, settings.n), width="stretch", key="potential_chart")
        st.caption("Potential Φ(k) für k Bündelnde unter n. Ein Koordinationsspiel hat zwei Gipfel an den Rändern; der höhere ist das risikodominante Gleichgewicht.")
    v3 = V.verdict_logit(lv, game)
    show(v3)
    verdicts.append(v3)
    mt1, mt2, mt3 = st.columns(3)
    mt1.metric("Masse bei höchstens 10 % Bündelnden", V.pct(lv.low))
    mt2.metric("Masse bei mindestens 90 % Bündelnden", V.pct(lv.high))
    mt3.metric("Erwarteter Anteil der Bündelnden", V.pct(lv.mean))

    st.markdown("---")
    st.subheader("🔬 Experiment E3: Trifft das Gibbs-Maß die Kette?")
    ns = ", ".join(str(v) for v in C.LOGIT_GRID_N)
    betas = "; ".join(V.de(v, 1) for v in C.LOGIT_GRID_BETA)
    st.caption(f"Vier Spiele × n ∈ {{{ns}}} × β ∈ {{{betas}}}: größte Abweichung zwischen der stationären Verteilung der Kette und dem Gibbs-Maß.")
    e3 = _e3()
    st.dataframe([{"Spiel (a, b, c, d)": str(r["game"]), "größte Abweichung": f"{r['worst']:.1e}".replace(".", ","), "Zellen": r["cells"]} for r in e3], hide_index=True, width="stretch")
    show(V.verdict_e3(e3))
    st.dataframe([{"β": V.de(r["beta"], 1), "Masse bei höchstens 10 % Bündelnden": V.pct(r["low"], 2), "Masse bei mindestens 90 % Bündelnden": V.pct(r["high"], 2), "erwarteter Anteil": V.pct(r["mean"], 1)}
                  for r in _e3_masses(settings.n, game)], hide_index=True, width="stretch")
    st.caption(f"Das gewählte Spiel {game} bei n = {settings.n}: Je größer β, desto schärfer liegt die Masse am Gipfel des Potentials.")

# --- Tab 4 ---------------------------------------------------------------------------------------------------------------------------------
with tab4:
    st.markdown("## 🎭 Zwei Rollen: Kampf der Geschlechter")
    st.markdown("Zwei verschiedene Rollen (zum Beispiel Spediteur und Terminal) wollen sich auf **Option 0 oder 1** einigen, haben aber verschiedene Vorlieben: Rolle 1 will Option 0, Rolle 2 will Option 1. "
                "Es gibt zwei reine Gleichgewichte und ein gemischtes; wohin die Dynamik läuft, hängt von der Stärke der Vorlieben ab.")
    rv = _roles(settings.role1, settings.role2, settings.runs, settings.seed)
    st.plotly_chart(Z.build_phase(rv), width="stretch", key="phase_chart")
    st.caption("Zehn Anfangspunkte: Verlauf der Anteile beider Rollen (Replikator). Grün endet bei „beide Option 0“, orange bei „beide Option 1“. Das Kreuz ist das gemischte Gleichgewicht.")
    v4 = V.verdict_roles(rv)
    show(v4)
    verdicts.append(v4)

    st.markdown("---")
    st.subheader("🔬 Experiment E4: Gewinnt die stärkere Vorliebe?")
    st.caption("Drei Spiele (Rollenwerte 2 gegen 2, 3 gegen 2, 4 gegen 4). Replikator von 2 500 Anfangspunkten, beste Antwort mit 30 + 30 Spielern und 200 Läufen je Strom. Rechnet einige Sekunden.")
    if st.button("▶️ Zwei-Rollen-Experiment rechnen", key="e4_button"):
        st.session_state["e4_done"] = True
    if st.session_state.get("e4_done"):
        with st.spinner("Rechne die Einzugsbereiche …"):
            e4 = _e4()
        st.dataframe([{"Rollenwerte": f"{r['roles'][0]} gegen {r['roles'][1]}", "(p*; q*)": f"({V.de(r['p_star'], 3)}; {V.de(r['q_star'], 3)})", "Replikator: Anteil „beide Option 0“": V.pct(r["share_00"]),
                       "beste Antwort: Anteil „beide Option 0“": V.pct(r["br_share_00"]), "Streuung über Ströme": V.pct(r["br_std"]), "nicht angekommen (Replikator)": V.pct(r["unreached"])} for r in e4],
                     hide_index=True, width="stretch")
        show(V.verdict_e4(e4))

# --- Tab 5 ---------------------------------------------------------------------------------------------------------------------------------
with tab5:
    st.markdown("## 📊 Heterogene Schwellen (Granovetter)")
    st.markdown("Jede Person macht mit, wenn mindestens ein gewisser Anteil der anderen schon mitmacht (ihre **Schwelle**). Die Schwellen sind verschieden; ob die Masse kippt, hängt an ihrer Verteilung, nicht nur an ihrem Mittel.")
    tv = _thresholds(settings.mu, settings.sigma, settings.s0, settings.runs, settings.seed)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(Z.build_threshold_cdf(settings.mu, settings.sigma, settings.s0), width="stretch", key="cdf_chart")
        st.caption("Der Anteil, dessen Schwelle bei Stand x erreicht ist. Wo die Kurve die Diagonale schneidet, bleibt der Stand stehen; der rote Punkt ist der Endstand ab dem Anfangsanteil.")
    with c2:
        st.plotly_chart(Z.build_threshold_finals(tv.finals), width="stretch", key="threshold_finals_chart")
        st.caption(f"Endanteile von {tv.draws} Ziehungen mit je {C.N_AGENTS} Personen (Normalverteilung der Schwellen, bei 0 gekappt).")
    v5 = V.verdict_thresholds(tv)
    show(v5)
    verdicts.append(v5)
    cl = tv.classic
    st.info(f"**Granovetters Beispiel mit 100 Personen:** Schwellen 0, 1, 2, …, 99 (je eine Person): Am Ende machen {cl['full']} mit. Wird die Person mit Schwelle 1 durch eine mit Schwelle 2 ersetzt "
            f"(Mittel der Schwellen {V.de(cl['mean_full'], 1)} gegen {V.de(cl['mean_changed'], 2)}), endet die Kette bei {cl['changed']}.")

    st.markdown("---")
    st.subheader("🔬 Experiment E5: Öffnet mehr Streuung das Kippen?")
    st.caption(f"Normalverteilte Schwellen, {C.N_AGENTS} Personen, {len(C.SWEEP_SEEDS)} Ströme zu je {X.E5_DRAWS} Ziehungen; Mittel der Endanteile gegen den Fixpunkt der Dichte.")
    e5 = _e5()
    st.plotly_chart(Z.build_sigma_sweep(e5, 30, 0), width="stretch", key="e5_chart")
    st.dataframe([{"mittlere Schwelle": f"{r['mu']} %", "Anfangsanteil": f"{r['s0']} %", "Streuung": f"{r['sigma']} %", "Endanteil (Mittel der Ziehungen)": V.pct(r["mean"]),
                   "Streuung über Ströme": V.pct(r["seed_std"]), "Streuung innerhalb eines Stroms": V.pct(r["inner_std"]), "Fixpunkt der Dichte": V.pct(r["fixed_point"])} for r in e5], hide_index=True, width="stretch")
    show(V.verdict_e5(e5))

# --- Tab 6 ---------------------------------------------------------------------------------------------------------------------------------
with tab6:
    st.markdown("## ⚔️ Wenn Gleichwahl bestraft wird")
    st.markdown("Im Feiglingsspiel (Vordrängeln gegen Abwarten am Tor) lohnt es, das Gegenteil des anderen zu wählen (a < d und b < c). Die Steigung **s = (a − d) + (b − c)** trennt beide Welten: "
                "Bei **s > 0** ist das gemischte Gleichgewicht instabil, bei **s < 0** stabil. Der Schnellstart „Feiglingsspiel“ lädt V = 4, C = 6.")
    av = _anti(game, settings.n, settings.beta)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(Z.build_flow(game), width="stretch", key="flow_chart")
        st.caption("Änderungsrate des Anteils (Replikator). Nullstellen sind Ruhepunkte; fällt die Kurve dort von oben nach unten, ist der Ruhepunkt stabil.")
    with c2:
        k_text = " und ".join(str(k) for k in av.stable_states) if av.stable_states else "keiner"
        st.metric(f"Stabile Zustände der besten Antwort (Zahl der Bündelnden unter {settings.n})", k_text)
        st.metric("Mittlere Auszahlung im gemischten Gleichgewicht", "–" if av.welfare_mixed is None else V.num(av.welfare_mixed))
        st.metric("Mittlere Auszahlung bei bester Absprache", V.num(av.welfare_agreed))
        st.metric(f"Erwarteter Anteil der Bündelnden (Logit, β = {V.de(settings.beta, 1)})", V.pct(av.logit_mean))
    v6 = V.verdict_anti(av)
    show(v6)
    verdicts.append(v6)

    st.markdown("---")
    st.subheader("🔬 Experiment E6: Wird das gemischte Gleichgewicht stabil?")
    st.caption(f"Vier Spiele, Replikator von {C.ANTI_GRID} gleichmäßig verteilten Anfangsanteilen: Anteil der Läufe, die nahe p* (Abstand unter 0,01) enden, gegen den Anteil an einem Rand.")
    e6 = _e6()
    st.dataframe([{"Spiel (a, b, c, d)": str(r["game"]), "Klasse": r["class"], "Steigung s": r["slope"], "p*": V.de(r["p_star"], 3), "enden nahe p*": V.pct(r["near"], 0), "enden an einem Rand": V.pct(r["edge"], 0)} for r in e6],
                 hide_index=True, width="stretch")
    show(V.verdict_e6(e6))

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **2×2 und vollständige Durchmischung:** jeder Betrieb trifft jeden anderen gleich oft | In Netzen und bei Nachbarschaften entscheidet, wer wen sieht; die Schwelle p* gilt dann nicht mehr als Einzugsbereich der ganzen Population. | komplexe-ansteckung-demo (gleiche Schwellen auf Netzen) |
| **Auszahlungen gewählt, nicht gemessen** | Alle Zahlen hängen an der Wahl von a, b, c, d. Der Anteil „ausz.-dominant, aber risikodominiert“ in E1 gehört zur gezogenen Verteilung der Auszahlungen und ist kein Naturgesetz. | – |
| **Lernregeln sind Modellannahmen** (beste Antwort, Replikator, Logit) | Menschen und Terminals lernen anders; die Demo zeigt, was *diese* Regeln tun. Im 2×2-Spiel haben Replikator und beste Antwort dieselbe Grenze p* (E2: Abweichung unter 0,01 bei n = 100), auf anderen Wegen. | – |
| **Kein dauerndes Rauschen, keine Wechselzeit** | Logit zeigt nur die Identität „stationäre Verteilung = Gibbs-Maß des Potentials“. Wie oft und wie schnell die Konvention kippt, ist nicht Gegenstand dieser Demo. | – |
| **Kein gemeinsames Signal, keine Absprache vorab** | Ein Vermittler oder gemeinsamer Fokalpunkt kann die Wahl koordinieren; hier startet jeder unabhängig. | korreliert-demo (Vermittler) |
| **Verteilte Schwellen vollständig durchmischt** | Auf Netzen kippt die Masse anders; das Granovetter-Modell hier kennt keine Nachbarschaft. | komplexe-ansteckung-demo |
"""
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist ein symmetrisches 2×2-Spiel mit den Auszahlungen $a, b, c, d$; gesucht sind seine Gleichgewichte, ihre Einzugsbereiche und die Verteilung, die Logit-Lernen annimmt.

| Symbol | Bedeutung |
|---|---|
| $a, b$ | Auszahlung, wenn beide bündeln bzw. beide direkt fahren |
| $c, d$ | Auszahlung, wenn ich bündele und der andere direkt fährt bzw. umgekehrt |
| $x$ | Anteil der Bündelnden |
| $p^*$ | Anteil, bei dem Bündeln und Direktfahren gleich viel zahlen |
| $n,\ k$ | Zahl der Betriebe, Zahl der Bündelnden |
| $\beta$ | Schärfe des Logit-Lernens |
| $\Phi$ | Potential |

**Auszahlung gegen einen Anteil $x$ der Bündelnden unter den anderen.**

$$u_S(x) = x\,a + (1-x)\,c, \qquad u_H(x) = x\,d + (1-x)\,b$$

**Schwelle des gemischten Gleichgewichts** (Koordination: $a > d$ und $b > c$):

$$p^* = \frac{b-c}{(a-d) + (b-c)}$$

**Replikatordynamik.**

$$\dot x = x\,(1-x)\,\bigl(u_S(x) - u_H(x)\bigr)$$

**Risikodominanz.** Bündeln ist risikodominant, wenn

$$a - d > b - c \iff p^* < \tfrac12$$

**Potential des Spiels.**

$$\Phi(S,S) = a-d, \qquad \Phi(H,H) = b-c, \qquad \Phi(S,H) = \Phi(H,S) = 0$$

**Potential bei $k$ Bündelnden unter $n$** (alle Paare gezählt):

$$\Phi(k) = \frac{(a-d)\binom{k}{2} + (b-c)\binom{n-k}{2}}{n-1}$$

**Logit-Wahl** (ein gezogener Betrieb mit Vorteil $\Delta$ der Strategie S):

$$\Pr(S) = \frac{1}{1 + e^{-\beta\Delta}}$$

**Stationäre Verteilung der Logit-Kette** (Gibbs-Maß):

$$\pi(k) \propto \binom{n}{k}\, e^{\beta\,\Phi(k)}$$

**Verteilte Schwellen** (Anteil, dessen Schwelle bei Stand $x$ erreicht ist, und Endstand ab dem Anfangsanteil $s_0$):

$$F(x) = \Phi_N\!\left(\frac{x-\mu}{\sigma}\right), \qquad x_{t+1} = \max\bigl(s_0,\ F(x_t)\bigr)$$

**Literatur.** Die Quellen sind in der README mit Fundstelle genannt (am Abstract geprüft, nicht im Volltext gelesen).
        """
    )

st.markdown("---")
st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from kos_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(settings, gv, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="koordinationsspiele.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
