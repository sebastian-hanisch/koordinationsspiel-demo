"""Das symmetrische 2x2-Spiel: Klassen, Gleichgewichte, Dominanz, Potential. Alles exakt in Brüchen (fractions), nichts gerundet.

Strategie S = bündeln (Hirsch), H = direkt fahren (Hase). Auszahlungen für mich: a = beide S, b = beide H, c = ich S und der andere H, d = ich H und der andere S.
Zeilenmatrix A = [[a, c], [d, b]] (Zeile: meine Wahl S, H; Spalte: die des anderen); der andere Spieler hat die Spiegelmatrix B[i][j] = A[j][i].
Die Gleichgewichte kommen aus der Aufzählung `enumerate_ne` (Indifferenzbedingungen, keine Formel für p*); `p_star` ist die geschlossene Formel,
mit der die Aufzählung in tests/test_oracle_koordinationsspiel.py verglichen wird."""

from __future__ import annotations

from fractions import Fraction

KOORDINATION = "Koordination"
ANTI = "Anti-Koordination"
DOMINANT = "dominant"
ENTARTET = "entartet"
CLASSES = (KOORDINATION, ANTI, DOMINANT, ENTARTET)


def classify(a: int, b: int, c: int, d: int) -> str:
    """Klasse des symmetrischen Spiels aus den Vorzeichen von a - d (Anreiz zu S, wenn der andere S spielt) und b - c (Anreiz zu H, wenn der andere H spielt).
    Beide positiv: Koordination. Beide negativ: Anti-Koordination. Verschiedene Vorzeichen: eine Strategie dominiert. Eine Differenz 0: entartet."""
    x, y = a - d, b - c
    if x == 0 or y == 0:
        return ENTARTET
    if x > 0 and y > 0:
        return KOORDINATION
    if x < 0 and y < 0:
        return ANTI
    return DOMINANT


def slope(a: int, b: int, c: int, d: int) -> int:
    """Steigung s = (a - d) + (b - c): Wie schnell wächst der Vorteil von S gegenüber H mit dem Anteil der S-Spieler (u_S - u_H = s·x - (b - c))."""
    return (a - d) + (b - c)


def p_star(a: int, b: int, c: int, d: int) -> Fraction | None:
    """Anteil der S-Spieler, bei dem S und H gleich viel zahlen: p* = (b - c) / ((a - d) + (b - c)). Nur für Koordination und Anti-Koordination sinnvoll (liegt dann in (0, 1))."""
    if classify(a, b, c, d) not in (KOORDINATION, ANTI):
        return None
    return Fraction(b - c, slope(a, b, c, d))


def payoff_dominant(a: int, b: int, c: int, d: int) -> str | None:
    """'S', wenn beide bündeln mehr zahlt als beide direkt (a > b), 'H' bei b > a, None bei Gleichstand."""
    return "S" if a > b else ("H" if b > a else None)


def risk_dominant(a: int, b: int, c: int, d: int) -> str | None:
    """Risikodominanz nach Harsanyi und Selten für das 2x2-Koordinationsspiel: das Gleichgewicht mit dem größeren Produkt der Abweichungsverluste.
    Beide Spieler tragen bei Abweichung vom Gleichgewicht (S,S) den Verlust a - d, bei (H,H) den Verlust b - c; es gewinnt das größere. Gleichstand: None.
    Gleichbedeutend: das risikodominante Gleichgewicht hat den größeren Einzugsbereich (p* < 1/2 heißt S risikodominant)."""
    if classify(a, b, c, d) != KOORDINATION:
        return None
    return "S" if a - d > b - c else ("H" if b - c > a - d else None)


def potential(a: int, b: int, c: int, d: int) -> dict[tuple[str, str], int]:
    """Potential des symmetrischen Spiels: Phi(S,S) = a - d, Phi(H,H) = b - c, gemischte Profile 0 (Monderer und Shapley: der Gewinn jeder einseitigen Abweichung ist die Potentialdifferenz)."""
    return {("S", "S"): a - d, ("H", "H"): b - c, ("S", "H"): 0, ("H", "S"): 0}


def payoff_matrices(a: int, b: int, c: int, d: int) -> tuple[list[list[int]], list[list[int]]]:
    """(A, B) des Zweipersonenspiels: Zeile spielt S/H, Spalte spielt S/H; Index 0 = S, 1 = H."""
    A = [[a, c], [d, b]]
    B = [[a, d], [c, b]]
    return A, B


def is_degenerate(A, B) -> bool:
    """Entartet (Gleichstand in einer besten Antwort): dann kann es unendlich viele Gleichgewichte geben und die Aufzählung ist unvollständig."""
    return any(A[0][j] == A[1][j] for j in (0, 1)) or any(B[i][0] == B[i][1] for i in (0, 1))


def enumerate_ne(A, B) -> list[tuple[Fraction, Fraction]]:
    """Alle Gleichgewichte eines 2x2-Bimatrix-Spiels als (p, q): p = Wahrscheinlichkeit, dass die Zeile Strategie 0 spielt, q = die der Spalte. Exakt in Brüchen.

    Reine Profile: keiner hat eine strikt bessere Antwort. Gemischtes Profil mit Träger 2x2: jede Seite mischt so, dass die andere indifferent ist (beide Nenner nicht 0, beide
    Wahrscheinlichkeiten echt zwischen 0 und 1). Bei Gleichstand (`is_degenerate`) kann es unendlich viele geben; dann nur die reinen Profile (Liste unvollständig)."""
    found: list[tuple[Fraction, Fraction]] = []
    for i in (0, 1):
        for j in (0, 1):
            if A[i][j] >= A[1 - i][j] and B[i][j] >= B[i][1 - j]:
                found.append((Fraction(1 - i), Fraction(1 - j)))
    dq = (A[0][0] - A[1][0]) - (A[0][1] - A[1][1])   # Zeile indifferent: q·A00 + (1-q)·A01 = q·A10 + (1-q)·A11
    dp = (B[0][0] - B[0][1]) - (B[1][0] - B[1][1])   # Spalte indifferent in p
    if dq != 0 and dp != 0 and not is_degenerate(A, B):
        q = Fraction(A[1][1] - A[0][1], dq)
        p = Fraction(B[1][1] - B[1][0], dp)
        if 0 < p < 1 and 0 < q < 1:
            found.append((p, q))
    return found


def equilibria(a: int, b: int, c: int, d: int) -> list[dict]:
    """Gleichgewichte des symmetrischen Spiels für die Anzeige: Liste von Zeilen mit Name, Art (rein/gemischt), (p, q), Auszahlungen beider Spieler (Brüche) und Symmetrie."""
    A, B = payoff_matrices(a, b, c, d)
    rows = []
    for p, q in enumerate_ne(A, B):
        pay_row = p * q * A[0][0] + p * (1 - q) * A[0][1] + (1 - p) * q * A[1][0] + (1 - p) * (1 - q) * A[1][1]
        pay_col = p * q * B[0][0] + p * (1 - q) * B[0][1] + (1 - p) * q * B[1][0] + (1 - p) * (1 - q) * B[1][1]
        mixed = 0 < p < 1 or 0 < q < 1
        if mixed:
            name = f"gemischt: jeder bündelt mit Wahrscheinlichkeit {p}"
        else:
            name = "".join("S" if v == 1 else "H" for v in (p, q))
            name = {"SS": "beide bündeln", "HH": "beide direkt", "SH": "Zeile bündelt, Spalte fährt direkt", "HS": "Zeile fährt direkt, Spalte bündelt"}[name]
        rows.append({"p": p, "q": q, "art": "gemischt" if mixed else "rein", "name": name, "pay_row": pay_row, "pay_col": pay_col, "symmetrisch": p == q})
    return rows


def ne_count(a: int, b: int, c: int, d: int) -> int:
    """Zahl der Gleichgewichte (nicht entartet: 3 bei Koordination und Anti-Koordination, 1 bei dominant)."""
    return len(enumerate_ne(*payoff_matrices(a, b, c, d)))


def basin_of_S(a: int, b: int, c: int, d: int) -> Fraction | None:
    """Einzugsbereich von 'alle bündeln' bei gleichverteiltem Anfangsanteil: 1 - p* (nur Koordination)."""
    if classify(a, b, c, d) != KOORDINATION:
        return None
    return 1 - p_star(a, b, c, d)


def payoff_dominant_basin(a: int, b: int, c: int, d: int) -> Fraction | None:
    """Einzugsbereich des auszahlungsdominanten Gleichgewichts (nur Koordination, a != b)."""
    s = payoff_dominant(a, b, c, d)
    if classify(a, b, c, d) != KOORDINATION or s is None:
        return None
    ps = p_star(a, b, c, d)
    return 1 - ps if s == "S" else ps


def mixed_welfare(a: int, b: int, c: int, d: int) -> Fraction | None:
    """Mittlere Auszahlung je Spieler im symmetrischen gemischten Gleichgewicht x = p*: x·u_S(x) + (1 - x)·u_H(x) (beide Strategien zahlen dort gleich viel)."""
    x = p_star(a, b, c, d)
    if x is None:
        return None
    return x * (x * a + (1 - x) * c) + (1 - x) * (x * d + (1 - x) * b)


def best_agreed_welfare(a: int, b: int, c: int, d: int) -> Fraction:
    """Mittlere Auszahlung je Spieler bei bester Absprache über die Profile 'beide S', 'beide H' und 'abwechselnd S/H' (Mittel aus c und d)."""
    return max(Fraction(a), Fraction(b), Fraction(c + d, 2))
