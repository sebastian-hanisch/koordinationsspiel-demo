"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Spiel, Gleichgewichte und die Verdikte der sechs Tabs. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import kos_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "ε": "eps", "β": "beta", "σ": "sigma", "μ": "mu", "Φ": "Phi", "×": "x"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: V.Settings, game_view: V.GameView, verdicts: list[tuple[str, str]]) -> bytes:
    s = settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Koordinationsspiele: wenn Gleichwahl belohnt wird"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Spiel (a, b, c, d) = {s.game}, Klasse {game_view.klass}, n = {s.n}, Anfangsanteil {s.start} %, Lernregel {s.rule}, beta = {V.de(s.beta, 1)}, Läufe {s.runs}, "
                                f"Zufalls-Seed {s.seed}; Rollen {s.role1} und {s.role2}; Schwellen: Mittel {s.mu} %, Streuung {s.sigma} %, Anfangsanteil {s.s0} %."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Gleichgewichte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for row in game_view.equilibria:
        pdf.multi_cell(0, 5, _latin(f"- {row['name']} ({row['art']}): Auszahlung Zeile {V.num(row['pay_row'])}, Spalte {V.num(row['pay_col'])}"), new_x="LMARGIN", new_y="NEXT")
    if game_view.p_star is not None:
        pdf.multi_cell(0, 5, _latin(f"- p* = {V.frac(game_view.p_star)}, Steigung s = {game_view.slope}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Eigennutz und Lernregeln sind Modellannahmen, keine Aussagen über Menschen oder Terminals. Auszahlungen: Zeitersparnis je Sendung in Minuten, gewählt, nicht gemessen."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
