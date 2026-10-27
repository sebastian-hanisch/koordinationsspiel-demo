"""Auswertungen für die App: Einstellungen, Ansichten je Tab und die Verdikte als reine Funktionen (Art, Text). Die Zahlen im Text stammen aus der Rechnung, nicht aus Konstanten."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import kos_constants as C
import kos_dynamics as D
import kos_experiments as X
import kos_games as G
import kos_logit as L
import kos_scenario as S
import kos_threshold as T
from kos_rng import SplitMix64

NAMES = {"S": "Bündeln", "H": "Direkt fahren"}


# ---------------------------------------------------------------------------------------------------------------------------- Formate


def de(x: float, digits: int = 3) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    """Anteil als Prozentzahl mit Komma und Leerzeichen vor dem Prozentzeichen. Ein Anteil knapp unter 1 (knapp über 0) wird nicht auf 100,0 % (0,0 %) gerundet,
    sondern als „mehr als 99,9 %“ („weniger als 0,1 %“) gezeigt: gerundete Randwerte sähen aus wie exakte."""
    s = f"{100 * x:.{digits}f}"
    if x < 1 and float(s) >= 100:
        s = "mehr als " + f"{100 - 10 ** -digits:.{digits}f}"
    elif x > 0 and float(s) == 0:
        s = "weniger als " + f"{10 ** -digits:.{digits}f}"
    return s.replace(".", ",") + " %"


def frac(x: Fraction) -> str:
    """Bruch als 2/3, ganze Zahlen ohne Nenner."""
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def num(x: Fraction | int | float) -> str:
    """Auszahlung für die Anzeige: Bruch mit Dezimalwert, ganze Zahl ohne Nachkommastellen."""
    if isinstance(x, Fraction):
        return frac(x) if x.denominator == 1 else f"{frac(x)} ({de(float(x), 3)})"
    return de(float(x), 3) if not float(x).is_integer() else str(int(x))


# ---------------------------------------------------------------------------------------------------------------------------- Einstellungen


@dataclass(frozen=True)
class Settings:
    a: int = C.BASE["a"]
    b: int = C.BASE["b"]
    c: int = C.BASE["c"]
    d: int = C.BASE["d"]
    n: int = C.BASE["n"]
    start: int = C.BASE["start"]
    rule: str = C.BASE["rule"]
    beta: float = C.BASE["beta"]
    runs: int = C.BASE["runs"]
    seed: int = C.BASE["seed"]
    role1: int = C.BASE["role1"]
    role2: int = C.BASE["role2"]
    mu: int = C.BASE["mu"]
    sigma: int = C.BASE["sigma"]
    s0: int = C.BASE["s0"]

    @property
    def game(self) -> tuple[int, int, int, int]:
        return (self.a, self.b, self.c, self.d)


# ---------------------------------------------------------------------------------------------------------------------------- Tab 1: zwei Betriebe


@dataclass(frozen=True)
class GameView:
    game: tuple
    klass: str
    p_star: Fraction | None
    slope: int
    equilibria: tuple
    risk: str | None
    payoff: str | None
    potential: dict
    basin_s: Fraction | None
    degenerate: bool


def analyse_game(game) -> GameView:
    a, b, c, d = game
    A, B = G.payoff_matrices(*game)
    return GameView(game, G.classify(*game), G.p_star(*game), G.slope(*game), tuple(G.equilibria(*game)), G.risk_dominant(*game), G.payoff_dominant(*game),
                    G.potential(*game), G.basin_of_S(*game), G.is_degenerate(A, B))


def verdict_game(v: GameView) -> tuple[str, str]:
    """Hauptverdikt von Tab 1 nach Klasse; im Koordinationsspiel vergleicht es Auszahlungs- und Risikodominanz."""
    a, b, c, d = v.game
    if v.klass == G.ENTARTET:
        return "warning", ("Entartetes Spiel: Bei a = d oder b = c ist eine Antwort gleich gut wie die andere, dann kann es unendlich viele Gleichgewichte geben. "
                           "Die Tabelle nennt nur die reinen; p* und die Dominanzurteile gibt es nicht. Eine Auszahlung ändern, um ein Koordinationsspiel zu bekommen.")
    if v.klass == G.DOMINANT:
        better = "Bündeln" if a > d else "Direkt fahren"
        return "info", (f"Hier dominiert **{better}**: Es gibt genau ein Gleichgewicht, und es gibt nichts zu koordinieren. "
                        "Ein Koordinationsspiel braucht a > d und b > c (beide Seiten wollen dasselbe wie der andere).")
    ps = v.p_star
    if v.klass == G.ANTI:
        return "info", (f"**Anti-Koordination:** Gleichwahl wird bestraft (a < d und b < c). Die beiden reinen Gleichgewichte sind asymmetrisch (einer bündelt, der andere fährt direkt); "
                        f"das gemischte bei p* = {frac(ps)} ({de(float(ps), 3)}) ist hier stabil (Steigung s = {v.slope}).")
    payoff, risk = v.payoff, v.risk
    base = f"p* = {frac(ps)} ({de(float(ps), 3)}). Bündeln lohnt, sobald mehr als dieser Anteil mitmacht; sein Einzugsbereich ist 1 - p* = {de(float(1 - ps), 3)}."
    if payoff is None and risk is None:
        return "info", f"**Gleichauf:** beide Gleichgewichte zahlen gleich viel (a = b) und sind gleich riskant. {base}"
    if payoff is None:
        return "info", f"Beide Gleichgewichte zahlen gleich viel (a = b); **{NAMES[risk]}** ist risikodominant. {base}"
    if risk is None:
        return "info", f"**{NAMES[payoff]}** ist auszahlungsdominant; keines der Gleichgewichte ist risikodominant (p* = 1/2, gleich große Einzugsbereiche). {base}"
    if payoff == risk:
        return "success", f"**{NAMES[payoff]}** ist auszahlungs- und risikodominant: Das bessere Gleichgewicht ist auch das, in das die Lernregeln laufen. {base}"
    return "warning", (f"**{NAMES[payoff]}** ist auszahlungsdominant, aber **{NAMES[risk]}** ist risikodominant: Das bessere Gleichgewicht hat den kleineren Einzugsbereich "
                       f"({de(float(1 - ps if payoff == 'S' else ps), 3)}). {base}")


# ---------------------------------------------------------------------------------------------------------------------------- Tab 2: viele Betriebe


@dataclass(frozen=True)
class PopulationView:
    n: int
    start: int
    rule: str
    klass: str
    exact_all_s: float               # Wahrscheinlichkeit 'alle bündeln' bei unabhängigem Start (beste Antwort), exakt
    sim_all_s: float | None          # Anteil der simulierten Läufe mit 'alle bündeln'
    sim_runs: int
    finals: tuple = ()               # Endzahlen der simulierten Läufe (beste Antwort)
    mean_steps: float = 0.0
    paths: tuple = ()                # Pfade (k nach jedem Schritt) einiger Läufe
    fan: tuple = ()                  # Replikator: (Anfangsanteil, Pfad) je Startwert
    boundary: float | None = None    # Replikator-Grenze des Einzugsbereichs
    start_final: float | None = None


FAN_STARTS = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


def analyse_population(game, n: int, start: int, rule: str, beta: float, runs: int, seed: int) -> PopulationView:
    klass = G.classify(*game)
    exact = float(X.start_without_agreement(n, start, game))
    rng = SplitMix64(seed)
    if rule == "Beste Antwort":
        finals, steps, paths = [], [], []
        for r in range(runs):
            k0 = S.start_count(n, start, rng)
            run = D.br_run(n, k0, game, rng, C.BR_MAX_STEPS, record=r < 8)
            finals.append(run.final)
            steps.append(run.steps)
            if r < 8:
                paths.append(run.path)
        return PopulationView(n, start, rule, klass, exact, sum(f == n for f in finals) / runs, runs, tuple(finals), float(np.mean(steps)), tuple(paths))
    if rule == "Replikator":
        starts = tuple(sorted(set(FAN_STARTS + (start / 100,))))
        t_end, steps = 30.0, 600
        path = D.replicator_path(np.array(starts), game, t_end, steps)
        fan = tuple((x0, path[:, i]) for i, x0 in enumerate(starts))
        boundary = D.basin_boundary(game) if klass == G.KOORDINATION else None
        final = float(D.replicator_final(start / 100, game, t_end, steps))
        return PopulationView(n, start, rule, klass, exact, None, 0, fan=fan, boundary=boundary, start_final=final)
    paths, finals = [], []
    for _ in range(C.LOGIT_PATHS):
        k0 = S.start_count(n, start, rng)
        k, path = L.logit_run(n, k0, game, beta, rng, C.LOGIT_PATH_STEPS_PER_PLAYER * n, record=True)
        paths.append(path)
        finals.append(k)
    return PopulationView(n, start, rule, klass, exact, None, C.LOGIT_PATHS, tuple(finals), 0.0, tuple(paths))


def verdict_population(v: PopulationView, game) -> tuple[str, str]:
    """Verdikt von Tab 2 je Lernregel. Beste Antwort: exakte Wahrscheinlichkeit gegen den Anteil der simulierten Läufe."""
    a, b, c, d = game
    if v.rule == "Beste Antwort":
        sim = f"in den {v.sim_runs} simulierten Läufen {pct(v.sim_all_s)}"
        if v.klass == G.ANTI:
            share = np.mean(v.finals) / v.n
            return "info", (f"Anti-Koordination: Die Läufe enden nicht bei einem Rand, sondern in einem Zustand, in dem keiner mehr wechseln will; im Mittel bündeln dann {pct(share)} der {v.n} Betriebe "
                            f"(Start: jeder bündelt unabhängig mit {v.start} %).")
        text = (f"Start ohne Absprache: Bündelt jeder Betrieb zu Beginn unabhängig mit {v.start} %, endet die beste Antwort bei n = {v.n} mit Wahrscheinlichkeit {pct(v.exact_all_s)} bei "
                f"„alle bündeln“ (exakt gerechnet), {sim}.")
        if v.klass == G.KOORDINATION and G.payoff_dominant(*game) == "S" and v.exact_all_s < 0.5:
            return "warning", text + " Das auszahlungsdominante Gleichgewicht wird also in der Mehrheit der Läufe verfehlt."
        return "info", text
    if v.rule == "Replikator":
        if v.klass == G.KOORDINATION:
            side = "alle bündeln" if v.start_final > 0.5 else "alle direkt"
            return "info", (f"Replikator: Die Grenze des Einzugsbereichs liegt bei {de(v.boundary, 3)} (gerechnet; p* = {de(float(G.p_star(*game)), 3)}). "
                            f"Der Anfangsanteil {v.start} % endet bei „{side}“.")
        return "info", f"Replikator: Der Anfangsanteil {v.start} % läuft zu einem Anteil von {pct(v.start_final)} Bündelnden (keine Grenze zwischen zwei Rändern: das Spiel ist kein Koordinationsspiel)."
    ends = np.array(v.finals) / v.n
    return "info", (f"Logit: {v.sim_runs} Beispielläufe mit {C.LOGIT_PATH_STEPS_PER_PLAYER * v.n} Schritten enden bei Anteilen von {pct(float(ends.min()))} bis {pct(float(ends.max()))} Bündelnden. "
                    "Wohin die Kette langfristig gehört, zeigt der Tab „Logit und Potential“ exakt.")


# ---------------------------------------------------------------------------------------------------------------------------- Tab 3: Logit und Potential


@dataclass(frozen=True)
class LogitView:
    n: int
    beta: float
    pi: np.ndarray
    gibbs: np.ndarray
    deviation: float
    low: float
    high: float
    mean: float
    peak_state: int
    potential_peak: str | None       # 'S' oder 'H': das Gleichgewicht mit dem größeren Potential


def analyse_logit(game, n: int, beta: float) -> LogitView:
    pi = L.stationary(n, game, beta)
    gb = L.gibbs(n, game, beta)
    a, b, c, d = game
    peak = "S" if a - d > b - c else ("H" if b - c > a - d else None)
    return LogitView(n, beta, pi, gb, float(np.abs(pi - gb).max()), L.mass_in_range(pi, 0.0, 0.1), L.mass_in_range(pi, 0.9, 1.0), L.mean_share(pi), int(np.argmax(pi)), peak)


def verdict_logit(v: LogitView, game) -> tuple[str, str]:
    klass = G.classify(*game)
    dev = f"{v.deviation:.1e}".replace(".", ",")
    base = f"Kette und Gibbs-Maß stimmen bis auf {dev} überein."
    if klass == G.KOORDINATION:
        risk = G.risk_dominant(*game)
        if risk is None:
            return "info", (f"Gleichauf (p* = 1/2): Beide Ränder haben dasselbe Potential, die Masse teilt sich: höchstens 10 % Bündelnde {pct(v.low)}, mindestens 90 % Bündelnde {pct(v.high)}. {base}")
        where = "höchstens 10 %" if risk == "H" else "mindestens 90 %"
        mass = v.low if risk == "H" else v.high
        payoff = G.payoff_dominant(*game)
        extra = " Das auszahlungsdominante Gleichgewicht ist hier nicht das gewählte." if payoff is not None and payoff != risk else ""
        return ("warning" if extra else "info"), (f"Das Potential ist beim risikodominanten Gleichgewicht ({NAMES[risk]}) am höchsten; bei β = {de(v.beta, 1)} und n = {v.n} liegen {pct(mass)} der Masse bei {where} "
                                                    f"Bündelnden.{extra} {base}")
    if klass == G.ANTI:
        return "info", (f"Anti-Koordination: Das Potential hat einen Gipfel im Inneren; erwarteter Anteil der Bündelnden {pct(v.mean)}, wahrscheinlichster Zustand {v.peak_state} von {v.n}. {base}")
    return "info", f"Im Spiel ohne zweites Gleichgewicht sammelt sich die Masse am dominanten Rand; erwarteter Anteil der Bündelnden {pct(v.mean)}. {base}"


# ---------------------------------------------------------------------------------------------------------------------------- Tab 4: zwei Rollen


@dataclass(frozen=True)
class RolesView:
    roles: tuple
    shares: dict
    br_runs: int
    br_share_00: float
    paths: tuple         # Beispielpfade (Form (T, 2)) von Anfangspunkten


ROLE_STARTS = ((0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8), (0.5, 0.9), (0.9, 0.5), (0.5, 0.1), (0.1, 0.5), (0.4, 0.6), (0.6, 0.4))


def analyse_roles(role1: int, role2: int, runs: int, seed: int) -> RolesView:
    A, B = S.role_matrices(role1, role2)
    path = D.two_role_replicator(np.array(ROLE_STARTS), A, B, 40.0, 400, record=True)
    br = X.two_role_br_share(role1, role2, runs, seed)
    return RolesView((role1, role2), X.two_role_shares(role1, role2), runs, br["share_00"], tuple(path[:, i, :] for i in range(len(ROLE_STARTS))))


def verdict_roles(v: RolesView) -> tuple[str, str]:
    s = v.shares
    r1, r2 = v.roles
    text = (f"Mischpunkt (p*, q*) = ({de(s['p_star'], 3)}; {de(s['q_star'], 3)}). Aus {C.TWO_ROLE_GRID ** 2} gleichmäßig verteilten Anfangspunkten endet die Replikatordynamik in {pct(s['share_00'])} bei „beide Option 0“ "
            f"(Wunsch der Rolle 1); {pct(s['unreached'])} kommen nirgends an. Mit 30 + 30 Spielern und bester Antwort (Start 1/2 je Spieler, {v.br_runs} Läufe) sind es {pct(v.br_share_00)}.")
    if r1 == r2:
        return "info", text + " Gleich starke Rollen: Der Wert liegt nahe der Hälfte; die Wunschoptionen sind austauschbar."
    return "info", text + f" Die stärkere Wunschoption (Rolle 1: {r1} gegen {r2}) gewinnt mehr Anfangspunkte; die Zahl folgt nicht aus den vier Quadranten um den Mischpunkt allein (sie führen nur {pct(s['direct_00'])} aller Anfangspunkte direkt zu „Option 0“)."


# ---------------------------------------------------------------------------------------------------------------------------- Tab 5: Schwellen


@dataclass(frozen=True)
class ThresholdView:
    mu: int
    sigma: int
    s0: int
    draws: int
    finals: np.ndarray
    mean: float
    fixed_point: float
    stable: int
    tipped: float          # Anteil der Ziehungen mit Endanteil >= 1/2
    classic: dict


def analyse_thresholds(mu: int, sigma: int, s0: int, draws: int, seed: int) -> ThresholdView:
    cell = X.threshold_cell(mu, sigma, s0, draws, seed)
    return ThresholdView(mu, sigma, s0, draws, cell["finals"], cell["mean"], cell["fixed_point"], T.count_stable_fixed(mu / 100, sigma / 100), float(np.mean(cell["finals"] >= 0.5)),
                         X.classic_example())


def verdict_thresholds(v: ThresholdView) -> tuple[str, str]:
    spread = f"{pct(float(v.finals.min()))} bis {pct(float(v.finals.max()))}"
    text = (f"Aus {v.draws} Ziehungen mit je {C.N_AGENTS} Personen (Schwellen im Mittel {v.mu} %, Streuung {v.sigma} %, {v.s0} % machen von sich aus mit) enden die Läufe im Mittel bei {pct(v.mean)}, "
            f"von {spread}; der Fixpunkt der Dichte-Fassung liegt bei {pct(v.fixed_point)}. In {pct(v.tipped)} der Ziehungen ist mindestens die Hälfte mitgekippt. "
            f"Die Funktion F hat {v.stable} stabile{'n' if v.stable == 1 else ''} Fixpunkt{'' if v.stable == 1 else 'e'} (Gitter 20 001).")
    if v.tipped > 0 and v.tipped < 1:
        return "warning", text + " Der Ausgang hängt hier an der Ziehung: gleiche Verteilung, verschiedene Ergebnisse."
    return "info", text


# ---------------------------------------------------------------------------------------------------------------------------- Tab 6: Anti-Koordination


@dataclass(frozen=True)
class AntiView:
    row: dict
    welfare_mixed: Fraction | None
    welfare_agreed: Fraction
    n: int
    stable_states: tuple
    logit_mean: float


def analyse_anti(game, n: int, beta: float) -> AntiView:
    stable = tuple(k for k in range(n + 1) if D.stable_state(n, k, game))
    return AntiView(X.anti_row(game), G.mixed_welfare(*game), G.best_agreed_welfare(*game), n, stable, L.mean_share(L.stationary(n, game, beta)))


def verdict_anti(v: AntiView) -> tuple[str, str]:
    r = v.row
    s = r["slope"]
    if r["class"] == G.KOORDINATION:
        text = (f"Steigung s = {s} > 0: Das gemischte Gleichgewicht p* = {de(r['p_star'], 3)} ist **instabil**. Von {C.ANTI_GRID} gleichverteilten Anfangsanteilen enden "
                f"{pct(r['edge'])} an einem Rand und {pct(r['near'])} nahe p*.")
        return "info", text
    if r["class"] == G.ANTI:
        mixed, agreed = v.welfare_mixed, v.welfare_agreed
        text = (f"Steigung s = {s} < 0: Das gemischte Gleichgewicht p* = {de(r['p_star'], 3)} ist **stabil**. Von {C.ANTI_GRID} gleichverteilten Anfangsanteilen enden {pct(r['near'])} nahe p* "
                f"und {pct(r['edge'])} an einem Rand. Mittlere Auszahlung je Spieler im gemischten Gleichgewicht {num(mixed)}, bei bester Absprache {num(agreed)}.")
        return "info", text
    return "info", f"Klasse „{r['class']}“: ohne zweites Gleichgewicht gibt es kein gemischtes, dessen Stabilität man prüfen könnte (p* ist nicht definiert)."


# ---------------------------------------------------------------------------------------------------------------------------- Experimente


def _row(rows, **match):
    (hit,) = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return hit


def verdict_e1_random(rows) -> tuple[str, str]:
    """E1a: Anteil der Koordinationsspiele, in denen das auszahlungsdominante Gleichgewicht risikodominiert ist, für beide Ziehungsarten. Die Verteilung der Auszahlungen steht im Satz."""
    g, h = _row(rows, mode=S.MODE_GENERAL), _row(rows, mode=S.MODE_SAFE_H)
    return "warning", (f"Bei unabhängig gleichverteilten Auszahlungen auf [0, 10) (nur Spiele mit a > d, b > c, a ≠ b; im Mittel {f"{g['games']:,.0f}".replace(",", ".")} Spiele je Strom, {g['seeds']} Ströme) "
                       f"ist das auszahlungsdominante Gleichgewicht in {pct(g['risk_dominated'])} der Spiele risikodominiert (Streuung über die Ströme {pct(g['risk_dominated_std'])}), sein mittlerer Einzugsbereich ist {de(g['mean_basin'], 3)}. "
                       f"Zahlt Direkt-Fahren unabhängig vom anderen (d = b, „sicherer Hase“), sind es {pct(h['risk_dominated'])}. "
                       "Die Anteile gehören zu dieser Ziehung der Auszahlungen; mit anderer Verteilung sind es andere Zahlen.")


def verdict_e1_start(rows) -> tuple[str, str]:
    """E1b: Start ohne Absprache, exakte Wahrscheinlichkeit für 'alle bündeln'."""
    r = rows[1]
    return "warning", (f"Im Bündeln-Spiel {r['game']} (Bündeln auszahlungsdominant, p* = {de(r['p_star'], 3)}) endet die beste Antwort bei n = {r['n']} und unabhängigem Start mit {r['start']} % nur mit Wahrscheinlichkeit {pct(r['all_s'], 2)} "
                       f"bei „alle bündeln“. Der Anfangsanteil {r['start']} % liegt unter p* = {de(r['p_star'], 3)}, deshalb läuft fast jeder Lauf nach „alle direkt“. "
                       f"Der Einzugsbereich von „alle bündeln“ bei gleichverteiltem Anfangsanteil wäre 1 - p* = {de(1 - r['p_star'], 3)}: ein anderes Maß als dieser feste Start.")


def verdict_e2(rows) -> tuple[str, str]:
    """E2: Grenze des Einzugsbereichs unter Replikator und bester Antwort gegen p*."""
    tol = 0.01
    rep_ok = sum(abs(r["replicator"] - r["p_star"]) < tol for r in rows)
    br_ok = sum(abs(r["best_response"] - r["p_star"]) < tol + 1e-9 for r in rows)
    worst_rep = max(abs(r["replicator"] - r["p_star"]) for r in rows)
    worst_br = max(abs(r["best_response"] - r["p_star"]) for r in rows)
    return "success", (f"In {rep_ok} von {len(rows)} Spielen liegt die gerechnete Grenze des Replikators innerhalb von {de(tol, 2)} von p* (größte Abweichung {de(worst_rep, 5)}); "
                       f"die exakte Grenze der besten Antwort bei n = {rows[0]['n']} in {br_ok} von {len(rows)} (größte Abweichung {de(worst_br, 3)}). "
                       "Im Ein-Populations-Fall eines 2×2-Spiels ist das kein Befund über die Dynamik, sondern die Probe, dass beide Regeln dieselbe Schwelle haben.")


def verdict_e3(rows) -> tuple[str, str]:
    worst = max(r["worst"] for r in rows)
    cells = sum(r["cells"] for r in rows)
    return "success", (f"Über {len(rows)} Spiele und {cells} Zellen (Spiel × n × β) weicht die stationäre Verteilung der Kette um höchstens {f'{worst:.1e}'.replace('.', ',')} vom Gibbs-Maß ab. "
                       "Das ist die Identität „Logit-Lernen maximiert das Potential“ in der Form, die Stück 2 braucht.")


def verdict_e4(rows) -> tuple[str, str]:
    parts = [f"Rollen {r['roles'][0]} gegen {r['roles'][1]}: Replikator {pct(r['share_00'])}, beste Antwort {pct(r['br_share_00'])}" for r in rows]
    return "info", ("Anteil der Anfangspunkte bzw. Läufe, die bei „beide Option 0“ enden. " + "; ".join(parts) + ". "
                    f"Replikator: {C.TWO_ROLE_GRID ** 2} Anfangspunkte auf einem Gitter; beste Antwort: {rows[0]['br_runs']} Läufe mit 30 + 30 Spielern. Die Startverteilungen beider Verfahren sind verschieden, die Zahlen sind nur je Verfahren zu lesen.")


def verdict_e5(rows) -> tuple[str, str]:
    lo, hi = _row(rows, mu=30, s0=0, sigma=10), _row(rows, mu=30, s0=0, sigma=20)
    mid = _row(rows, mu=30, s0=0, sigma=15)
    return "warning", (f"Bei mittlerer Schwelle 30 % und ohne Anfangsanteil endet das Mittel der Läufe bei Streuung 10 % bei {pct(lo['mean'])}, bei 15 % bei {pct(mid['mean'])} (Fixpunkt {pct(mid['fixed_point'])}, "
                       f"Streuung innerhalb eines Stroms {pct(mid['inner_std'])}) und bei 20 % bei {pct(hi['mean'])}: Bei dieser mittleren Schwelle öffnet mehr Streuung der Schwellen das Kippen, und nahe der Grenze entscheidet die Ziehung. "
                       f"({lo['draws']} Ziehungen je Zelle, {C.N_AGENTS} Personen).")


def verdict_e6(rows) -> tuple[str, str]:
    coord = [r for r in rows if r["class"] == G.KOORDINATION]
    anti = [r for r in rows if r["class"] == G.ANTI]
    fe = _row(rows, game=(-1, 2, 4, 0))
    names = lambda rs: ", ".join(str(r["game"]) for r in rs)       # noqa: E731
    return "info", (f"Koordination (s > 0), Spiele {names(coord)}: von {C.ANTI_GRID} Anfangsanteilen enden {', '.join(pct(r['edge'], 0) for r in coord)} an einem Rand, {', '.join(pct(r['near'], 0) for r in coord)} nahe p*. "
                    f"Anti-Koordination (s < 0), Spiele {names(anti)}: {', '.join(pct(r['near'], 0) for r in anti)} enden nahe p*, {', '.join(pct(r['edge'], 0) for r in anti)} an einem Rand. "
                    + anti_welfare_text(fe["game"]))


def anti_welfare_text(game) -> str:
    wf = X.anti_welfare(game)
    return (f"Im Feiglingsspiel {game} zahlt das gemischte Gleichgewicht {num(wf['mixed'])} je Spieler, eine abwechselnde Absprache {num(wf['agreed'])}; "
            f"im Spiel (0, 3, 4, 2) sind es {num(X.anti_welfare((0, 3, 4, 2))['mixed'])} gegen {num(X.anti_welfare((0, 3, 4, 2))['agreed'])}: Der Gewinn durch Absprache hängt von den Auszahlungen ab.")


# ---------------------------------------------------------------------------------------------------------------------------- Tabelle der Gleichgewichte


def stability_label(game, p: Fraction, q: Fraction) -> str:
    """Stabilität eines Gleichgewichts unter der Replikatordynamik einer Population (nur für symmetrische Profile p = q): Vorzeichen der Ableitung der Änderungsrate am Ruhepunkt.
    Rand 0: u_S - u_H = c - b, Rand 1: -(a - d), innen: x(1 - x)·s. Asymmetrische Profile gibt es in einer Population nicht."""
    a, b, c, d = game
    if p != q:
        return "– (asymmetrisch, nicht Ruhepunkt einer Population)"
    if p == 0:
        slope_at = c - b
    elif p == 1:
        slope_at = -(a - d)
    else:
        slope_at = G.slope(*game)
    return "stabil" if slope_at < 0 else ("instabil" if slope_at > 0 else "unbestimmt")


def equilibrium_table(v: GameView) -> list[dict]:
    rows = []
    for e in v.equilibria:
        rows.append({"Gleichgewicht": e["name"], "Art": e["art"], "Auszahlung Zeile / Spalte": f"{num(e['pay_row'])} / {num(e['pay_col'])}",
                     "Stabilität (Replikator, eine Population)": stability_label(v.game, e["p"], e["q"])})
    return rows
