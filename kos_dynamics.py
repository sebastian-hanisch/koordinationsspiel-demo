"""Lernregeln im vollständig durchmischten Spiel: Replikatordynamik, sequentielle beste Antwort, Zwei-Rollen-Fassung.

`game` ist das Tupel (a, b, c, d) ganzer Zahlen (Bedeutung in kos_games.py). x = Anteil der S-Spieler (bündeln), k = ihre Anzahl unter N.
Die beste Antwort rechnet ganzzahlig über Zählungen (nie über Anteile in Gleitkomma: 3·(10/30) - 1 ist nicht 0, sondern knapp daneben)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

# ---------------------------------------------------------------------------------------------------------------- Replikator


def replicator_rhs(x, game):
    """dx/dt = x (1 - x) (u_S(x) - u_H(x)) mit u_S = x·a + (1 - x)·c und u_H = x·d + (1 - x)·b; arbeitet auf Zahlen und Feldern."""
    a, b, c, d = game
    return x * (1 - x) * ((x * a + (1 - x) * c) - (x * d + (1 - x) * b))


def _rk4_step(x, game, h):
    k1 = replicator_rhs(x, game)
    k2 = replicator_rhs(x + h / 2 * k1, game)
    k3 = replicator_rhs(x + h / 2 * k2, game)
    k4 = replicator_rhs(x + h * k3, game)
    return np.clip(x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, 1.0)


def replicator_path(x0, game, t_end: float = 60.0, steps: int = 6000):
    """Verlauf des Anteils (Runge-Kutta 4. Ordnung, Schrittweite t_end/steps). x0: Zahl oder Feld; Ergebnis hat vorn die Zeitachse: Form (steps + 1,) + x0.shape."""
    h = t_end / steps
    x = np.asarray(x0, dtype=float)
    out = np.empty((steps + 1,) + x.shape)
    out[0] = x
    for i in range(steps):
        x = _rk4_step(x, game, h)
        out[i + 1] = x
    return out


def replicator_final(x0, game, t_end: float = 60.0, steps: int = 6000):
    """Endanteil nach t_end (ohne den ganzen Verlauf zu halten)."""
    h = t_end / steps
    x = np.asarray(x0, dtype=float)
    for _ in range(steps):
        x = _rk4_step(x, game, h)
    return x


def basin_boundary(game, t_end: float = 60.0, steps: int = 1500, rounds: int = 3, points: int = 33) -> float:
    """Grenze des Einzugsbereichs von 'alle S' unter der Replikatordynamik (Koordinationsspiel): der kleinste Anfangsanteil, von dem der Lauf bei S endet.
    Gesucht wird schrittweise verfeinert: je Runde ein Gitter von `points` Anfangsanteilen, alle zugleich gerechnet (vektorisiert); die Grenze liegt zwischen dem letzten
    Anteil, der bei H endet, und dem ersten, der bei S endet. Ergebnis auf (points - 1)^-rounds genau (hier 3·10^-5). Endet schon 0.001 bei S: 0.0; endet auch 0.999 bei H: 1.0."""
    lo, hi = 0.0, 1.0
    probe = replicator_final(np.array([0.001, 0.999]), game, t_end, steps)
    if probe[0] > 0.5:
        return 0.0
    if probe[1] <= 0.5:
        return 1.0
    for _ in range(rounds):
        grid = np.linspace(lo, hi, points)
        ends_s = replicator_final(grid, game, t_end, steps) > 0.5
        first = int(np.argmax(ends_s))
        if first == 0:
            return float(grid[0])
        lo, hi = float(grid[first - 1]), float(grid[first])
    return hi


# ---------------------------------------------------------------------------------------------------------------- beste Antwort


def switch_gain(m: int, n: int, game) -> int:
    """(Auszahlung S) - (Auszahlung H), mal (N - 1), wenn m der N - 1 anderen S spielen. Ganzzahl; > 0 heißt: S ist strikt besser."""
    a, b, c, d = game
    return m * (a - d) - (n - 1 - m) * (b - c)


def wants_switch(n: int, k: int, game) -> tuple[bool, bool]:
    """(ein H-Spieler will zu S wechseln, ein S-Spieler will zu H wechseln) bei k S-Spielern unter n. Nur strikte Verbesserungen zählen (bei Gleichstand bleibt jeder)."""
    h_to_s = k < n and switch_gain(k, n, game) > 0
    s_to_h = k > 0 and switch_gain(k - 1, n, game) < 0
    return h_to_s, s_to_h


def stable_state(n: int, k: int, game) -> bool:
    """Niemand will wechseln: ein Gleichgewicht des endlichen Spiels mit n Spielern."""
    return wants_switch(n, k, game) == (False, False)


@dataclass(frozen=True)
class BRRun:
    final: int              # Zahl der S-Spieler am Ende
    steps: int              # gezogene Spieler (Schritte)
    converged: bool         # niemand wollte mehr wechseln
    path: tuple = ()        # k nach jedem Schritt (nur bei record=True), path[0] = Start


def br_run(n: int, k0: int, game, rng, max_steps: int = 20000, record: bool = False) -> BRRun:
    """Sequentielle beste Antwort: ein zufällig gezogener Spieler wechselt, wenn der Wechsel gegen die n - 1 anderen strikt besser ist. Zustand ist nur die Zahl k der S-Spieler
    (alle Spieler sind gleichwertig). `rng` ist ein SplitMix64; je Schritt wird genau eine Zufallszahl gezogen (Spieler Nummer int(u·n), die ersten k spielen S)."""
    k = k0
    path = [k] if record else []
    steps = 0
    while steps < max_steps:
        h_to_s, s_to_h = wants_switch(n, k, game)
        if not h_to_s and not s_to_h:
            return BRRun(k, steps, True, tuple(path))
        i = int(rng.uniform() * n)
        steps += 1
        if i < k:
            if s_to_h:
                k -= 1
        elif h_to_s:
            k += 1
        if record:
            path.append(k)
    return BRRun(k, steps, stable_state(n, k, game), tuple(path))


def br_chain_probs(n: int, game) -> tuple[list[int], list[int]]:
    """Übergangs-Indikatoren der Kette auf k = 0..n: up[k] = 1 und down[k] = 1, wenn ein H- bzw. S-Spieler dort wechseln will (die Wahrscheinlichkeit ist (n - k)/n bzw. k/n)."""
    up, down = [], []
    for k in range(n + 1):
        h_to_s, s_to_h = wants_switch(n, k, game)
        up.append(int(h_to_s))
        down.append(int(s_to_h))
    return up, down


def br_absorption_probabilities(n: int, game) -> list[Fraction]:
    """Exakte Wahrscheinlichkeit, aus jedem Start k0 bei k = n (alle S) anzukommen, bevor die Kette woanders hängen bleibt (Wechselwahrscheinlichkeiten (n - k)/n und k/n).
    Löst die Geburts-Sterbe-Kette mit Brüchen (Spielerruin-Formel über Abschnitte zwischen stabilen Zuständen). Zustände ohne Wechselwunsch halten (Wert 0, außer n)."""
    up, down = br_chain_probs(n, game)
    pu = [Fraction(n - k, n) * up[k] for k in range(n + 1)]
    pd = [Fraction(k, n) * down[k] for k in range(n + 1)]
    stuck = [k for k in range(n + 1) if pu[k] == 0 and pd[k] == 0]
    if not stuck:
        return [Fraction(0)] * (n + 1)          # kein Haltezustand (kommt bei diesen Spielen nicht vor)
    result = [Fraction(0)] * (n + 1)
    value = {k: Fraction(1 if k == n else 0) for k in stuck}
    for k in stuck:
        result[k] = value[k]
    # unterhalb des ersten und oberhalb des letzten Haltezustands läuft die Kette sicher in ihn (der Rand ist reflektierend)
    for k in range(stuck[0]):
        result[k] = value[stuck[0]]
    for k in range(stuck[-1] + 1, n + 1):
        result[k] = value[stuck[-1]]
    # zwischen zwei Haltezuständen: lineares System (pu + pd)·h(k) = pu·h(k+1) + pd·h(k-1), tridiagonal, mit Brüchen nach Thomas gelöst
    for lo, hi in zip(stuck, stuck[1:]):
        inner = list(range(lo + 1, hi))
        if not inner:
            continue
        sub = [-pd[k] for k in inner]                       # Koeffizient von h(k-1)
        diag = [pu[k] + pd[k] for k in inner]
        sup = [-pu[k] for k in inner]                       # Koeffizient von h(k+1)
        rhs = [Fraction(0)] * len(inner)
        rhs[0] += pd[inner[0]] * value[lo]
        rhs[-1] += pu[inner[-1]] * value[hi]
        for i in range(1, len(inner)):                      # Vorwärtselimination
            w = sub[i] / diag[i - 1]
            diag[i] -= w * sup[i - 1]
            rhs[i] -= w * rhs[i - 1]
        sol = [Fraction(0)] * len(inner)
        sol[-1] = rhs[-1] / diag[-1]
        for i in range(len(inner) - 2, -1, -1):             # Rückwärtseinsetzen
            sol[i] = (rhs[i] - sup[i] * sol[i + 1]) / diag[i]
        for k, v in zip(inner, sol):
            result[k] = v
    return result


# ---------------------------------------------------------------------------------------------------------------- zwei Rollen


def two_role_rhs(z, A, B):
    """Replikatordynamik für zwei Rollen (Kampf der Geschlechter): z[..., 0] = Anteil der Zeilenrolle mit Option 0, z[..., 1] = der Spaltenrolle mit Option 0.
    Jede Rolle antwortet auf den Anteil der anderen Rolle (nicht auf den eigenen)."""
    p, q = z[..., 0], z[..., 1]
    ur = q * (A[0][0] - A[1][0]) + (1 - q) * (A[0][1] - A[1][1])
    uc = p * (B[0][0] - B[0][1]) + (1 - p) * (B[1][0] - B[1][1])
    return np.stack([p * (1 - p) * ur, q * (1 - q) * uc], axis=-1)


def two_role_replicator(z0, A, B, t_end: float = 60.0, steps: int = 3000, record: bool = False):
    """Runge-Kutta 4 für die Zwei-Rollen-Dynamik; z0 hat die Form (..., 2). Mit record=True der ganze Verlauf (steps + 1, ..., 2), sonst nur der Endpunkt."""
    h = t_end / steps
    z = np.asarray(z0, dtype=float)
    out = [z] if record else None
    for _ in range(steps):
        k1 = two_role_rhs(z, A, B)
        k2 = two_role_rhs(z + h / 2 * k1, A, B)
        k3 = two_role_rhs(z + h / 2 * k2, A, B)
        k4 = two_role_rhs(z + h * k3, A, B)
        z = np.clip(z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, 1.0)
        if record:
            out.append(z)
    return np.stack(out) if record else z


def two_role_goal(z, margin: float = 0.1):
    """Ziel eines Endpunkts: 0 = beide Rollen bei Option 0, 1 = beide bei Option 1, -1 = keines von beiden (noch unterwegs)."""
    z = np.asarray(z)
    at0 = (z[..., 0] > 1 - margin) & (z[..., 1] > 1 - margin)
    at1 = (z[..., 0] < margin) & (z[..., 1] < margin)
    return np.where(at0, 0, np.where(at1, 1, -1))


def two_role_mixed_point(A, B) -> tuple[Fraction, Fraction]:
    """(p*, q*) des gemischten Gleichgewichts: p* = Anteil der Zeilenrolle mit Option 0, bei dem die Spalte indifferent ist; q* analog (exakt)."""
    dq = (A[0][0] - A[1][0]) - (A[0][1] - A[1][1])
    dp = (B[0][0] - B[0][1]) - (B[1][0] - B[1][1])
    return Fraction(B[1][1] - B[1][0], dp), Fraction(A[1][1] - A[0][1], dq)


def two_role_br_run(n_row: int, n_col: int, kr0: int, kc0: int, A, B, rng, max_steps: int = 20000) -> tuple[int, int, bool]:
    """Sequentielle beste Antwort mit zwei Rollen (n_row Zeilen- und n_col Spaltenspieler; Zeilenspieler antworten auf die Spalten und umgekehrt). Zählt ganzzahlig.
    Gibt (Zahl der Zeilenspieler mit Option 0, Zahl der Spaltenspieler mit Option 0, Lauf endete in einem Gleichgewicht der endlichen Population)."""
    kr, kc = kr0, kc0
    for _ in range(max_steps):
        # Gewinn von Option 0 gegen Option 1, wenn kc Spalten Option 0 spielen (Zeile) bzw. kr Zeilen Option 0 (Spalte); mal der Zahl der Gegner
        gain_row = kc * (A[0][0] - A[1][0]) + (n_col - kc) * (A[0][1] - A[1][1])
        gain_col = kr * (B[0][0] - B[0][1]) + (n_row - kr) * (B[1][0] - B[1][1])
        row_up = kr < n_row and gain_row > 0
        row_down = kr > 0 and gain_row < 0
        col_up = kc < n_col and gain_col > 0
        col_down = kc > 0 and gain_col < 0
        if not (row_up or row_down or col_up or col_down):
            return kr, kc, True
        i = int(rng.uniform() * (n_row + n_col))
        if i < n_row:
            if i < kr:
                kr -= 1 if row_down else 0
            else:
                kr += 1 if row_up else 0
        else:
            j = i - n_row
            if j < kc:
                kc -= 1 if col_down else 0
            else:
                kc += 1 if col_up else 0
    return kr, kc, False
