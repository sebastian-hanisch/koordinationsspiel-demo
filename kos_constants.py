"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenaufwand am Slider-Maximum (DEMO-PLAYBOOK §1c): Replikator O(Schritte) vektorisiert; beste Antwort O(Läufe · Schritte), Schritte bis zum Stillstand etwa n·ln(n);
Logit-Kette O(n) im Logarithmus; exakte Absorptionswahrscheinlichkeit O(n) Bruchrechnung; Schwellenmodell O(Ziehungen · n log n)."""

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
PAY_MIN, PAY_MAX = -10, 10                  # Zeitersparnis je Sendung in Minuten; negativ = Zeit verloren (Feiglingsspiel: a = -1)
N_MIN, N_MAX, DEFAULT_N = 2, 200, 50        # Betriebe in der Population; Aufwand beste Antwort etwa n·ln(n) Schritte je Lauf
START_MIN, START_MAX, DEFAULT_START = 0, 100, 50   # Anfangsanteil der Bündelnden in Prozent (jeder startet unabhängig mit dieser Wahrscheinlichkeit)
RULES = ("Beste Antwort", "Replikator", "Logit")
BETA_MIN, BETA_MAX, DEFAULT_BETA = 0.1, 10.0, 2.0
RUNS_MIN, RUNS_MAX, DEFAULT_RUNS = 10, 300, 100     # Läufe der Zufallsverfahren (beste Antwort, Schwellen-Ziehungen); O(Läufe · n·ln n)
SEED_MAX, DEFAULT_SEED = 999999, 35
ROLE_MIN, ROLE_MAX = 1, 8                   # Kampf der Geschlechter: Wert der Wunschoption der Rolle (die andere Option zahlt 1)
MU_MIN, MU_MAX = 10, 90                     # mittlere Schwelle in Prozent der Gesamtheit
SIGMA_MIN, SIGMA_MAX = 2, 50                # Streuung der Schwellen in Prozent
S0_MIN, S0_MAX = 0, 30                      # Anteil, der von sich aus mitmacht, in Prozent
N_AGENTS = 1000                             # Personen im Schwellenmodell (fest); O(Ziehungen · n log n) wegen Sortierung

# --- feste Annahmen -------------------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = (100001, 100002, 100003, 100004)      # feste Zufallsströme der Messreihen: nie ein einzelner Seed als Befund
RANDOM_GAME_DRAWS = 100000                          # Ziehungen (a, b, c, d) je Seed vor der Bedingung a > d, b > c
BR_MAX_STEPS = 20000                                # Schritte, nach denen ein Lauf der besten Antwort abbricht
REPLICATOR_T, REPLICATOR_STEPS = 60.0, 6000         # Zeitspanne und Schrittzahl des Runge-Kutta
TWO_ROLE_GRID = 50                                  # 50 x 50 Anfangspunkte der Zwei-Rollen-Dynamik
TWO_ROLE_T, TWO_ROLE_STEPS = 40.0, 2000
ANTI_GRID = 1000                                    # Anfangsanteile (i + 1/2)/1000 für den Stabilitätstest des gemischten Gleichgewichts
ANTI_T, ANTI_STEPS = 40.0, 2000
LOGIT_GRID_N = (4, 10, 30, 100, 200)
LOGIT_GRID_BETA = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
LOGIT_PATH_STEPS_PER_PLAYER = 30                    # Schritte je Spieler der Beispielläufe der Logit-Regel
LOGIT_PATHS = 6

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------
COLOR_S = "#2F6B65"          # bündeln
COLOR_H = "#C0762B"          # direkt
COLOR_LINE = "#4c78a8"
COLOR_GRAY = "rgba(150,150,150,0.55)"
COLOR_RED = "#d62728"

# --- Presets --------------------------------------------------------------------------------------------------------------------------------
BASE = {
    "a": 6, "b": 4, "c": 0, "d": 4,
    "n": DEFAULT_N, "start": DEFAULT_START, "rule": RULES[0], "beta": DEFAULT_BETA, "runs": DEFAULT_RUNS, "seed": DEFAULT_SEED,
    "role1": 3, "role2": 2,
    "mu": 30, "sigma": 15, "s0": 0,
}
PRESETS = {
    "Bündeln am Hub (6, 4, 0, 4)": dict(BASE),
    "Hirschjagd klassisch (5, 2, 1, 2)": {**BASE, "a": 5, "b": 2, "c": 1, "d": 2},
    "Gleichauf (6, 4, 2, 4)": {**BASE, "c": 2},
    "Reine Koordination (3, 1, 0, 0)": {**BASE, "a": 3, "b": 1, "c": 0, "d": 0},
    "Kampf der Geschlechter (zwei Rollen)": {**BASE, "role1": 2, "role2": 2},
    "Feiglingsspiel (V = 4, C = 6)": {**BASE, "a": -1, "b": 2, "c": 4, "d": 0},
    "Schwellen streuen (Granovetter)": {**BASE, "sigma": 20},
}
PRESET_HELP = {
    "Bündeln am Hub (6, 4, 0, 4)": "Bündeln zahlt am meisten, wenn beide mitmachen (6 gegen 4), aber p* = 2/3: Direkt fahren ist risikodominant. Das bessere Gleichgewicht muss nicht das gewählte sein (Tab 1, Tab 2).",
    "Hirschjagd klassisch (5, 2, 1, 2)": "Der Klassiker: p* = 1/4, Bündeln ist auszahlungs- und risikodominant.",
    "Gleichauf (6, 4, 2, 4)": "Beide Gleichgewichte gleich riskant: p* = 1/2, gleich große Einzugsbereiche; Bündeln zahlt beim Zusammentreffen mehr (6 gegen 4).",
    "Reine Koordination (3, 1, 0, 0)": "Gleichwahl zahlt, Verschiedenwahl nichts (ein Standard wie ein Datenformat): p* = 1/4. S steht hier für den Standard, der mehr bringt (3 gegen 1).",
    "Kampf der Geschlechter (zwei Rollen)": "Tab 4 mit gleich starken Rollen (Wert 2 gegen 1 für beide): zwei reine Gleichgewichte, ein gemischtes.",
    "Feiglingsspiel (V = 4, C = 6)": "Vordrängeln gegen Abwarten am Tor: Hier steht „Bündeln“ für Vordrängeln und „Direkt“ für Abwarten. Gleichwahl wird bestraft, das gemischte Gleichgewicht liegt bei V/C = 2/3 und ist stabil (Tab 6).",
    "Schwellen streuen (Granovetter)": "Tab 5 mit Streuung 20 % statt 15 %: mittlere Schwelle 30 %, kein Anfangsanteil. Mit mehr Streuung kippt die Masse.",
}
