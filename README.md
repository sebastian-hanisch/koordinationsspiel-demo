# Koordinationsspiele: wenn Gleichwahl belohnt wird – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-koordinationsspiel-demo.streamlit.app/)** -->

Erstes Stück der Reihe **Koordinationsspiele** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, ihre Wurzel. Vehikel: **Bündelung am Hub.** Jeder Betrieb fährt direkt (sicher) oder bündelt am Hub (lohnt nur, wenn genug andere mitmachen), die Auszahlung ist die Zeitersparnis je Sendung in Minuten. Die Demo zeigt die Gleichgewichte eines solchen Spiels, den **Einzugsbereich** jedes Gleichgewichts, den **Start ohne Absprache**, die Wirkung von **Logit-Lernen** (Gibbs-Maß des Potentials), **zwei Rollen** mit verschiedenen Vorlieben, **verteilte Schwellen** nach Granovetter und den Gegenfall, in dem Gleichwahl bestraft wird.

Abgrenzung: Im Überlastungsspiel der `nash-demo` meiden die Lkw dasselbe Tor; das Mini-Spiel dort ist formal ein Anti-Koordinationsspiel, hier wird Gleichwahl belohnt und der Gegenfall (Tab 6) mit der Stabilität des gemischten Gleichgewichts und der Wohlfahrt gegen Absprache betrachtet. `replikator-demo` rechnet Replikatordynamik für sechs Automaten im wiederholten Gefangenendilemma; hier ist sie der eindimensionale Fall eines 2×2-Koordinationsspiels. `komplexe-ansteckung-demo` misst gleiche Schwellen auf Netzen, hier stehen verteilte Schwellen in der vollständig durchmischten Population. **Eigennutz und die Lernregeln (beste Antwort, Replikator, Logit) sind Modellannahmen**, keine Aussagen über Menschen oder Terminals; die Auszahlungen sind gewählt, nicht gemessen.

## Kernfrage

*Wenn jeder dasselbe wählen will: welche Gleichgewichte gibt es, in welches läuft man, und ist das auszahlungsdominante auch das gewählte?* Risikodominanz, Hirschjagd und der Einzugsbereich 1 − p\* sind Lehrbuchstoff (Harsanyi und Selten 1988, Kandori, Mailath und Rob 1993). Eigenes dieser Demo sind (1) die **Zahlen über Zufallsspiele**, (2) der **exakte Start ohne Absprache** im Bündelungs-Vehikel (Kette über die Zahl der Bündelnden, in Brüchen), (3) der Beleg, dass das Gibbs-Maß die Logit-Kette trifft, (4) der Kampf der Geschlechter mit gemessenen Einzugsbereichen und (5) die Streuung im Granovetter-Modell.

## Befunde und Korrekturen gegenüber dem Plan

Die Hypothesen des Einzelplans (`koordinationsspiele-planung/STUECK-1-koordinationsspiel.md`) wurden im Bau mit dem **eigenen SplitMix64-Strom** neu gemessen (vier feste Ströme `SWEEP_SEEDS`); die Wegwerf-Vormessung des Plans (numpy-Zufall) liefert keine README-Zahl.

| | Hypothese | Ergebnis | Zahl (diese Demo; Test) |
|---|---|---|---|
| H1 | Jedes nicht entartete Koordinations- und Anti-Koordinationsspiel hat genau 3 Gleichgewichte, jedes dominante genau 1; das gemischte liegt bei p\* = (b − c)/((a − d) + (b − c)) | **Ja** | Alle 10 000 ganzzahligen Spiele mit Auszahlungen 0 bis 9: 2 025 Koordination, 2 025 Anti-Koordination, 4 050 dominant, 1 900 entartet; in den 8 100 nicht entarteten stimmen Zahl und Lage ohne Ausnahme (`test_h1_*`) |
| H2 | Die Grenze des Einzugsbereichs (Replikator und beste Antwort) ist p\* | **Ja** | Vier Spiele: Replikator 0,25 / 0,50 / 0,667 / 0,25 gegen p\* ebenso (Abweichung unter 10⁻⁴); beste Antwort bei n = 100 exakt 0,25 / 0,50 / 0,67 / 0,25 (`test_h2_*`). Im 2×2-Fall mit einer Population ist das ein Orakel-Abgleich, kein Befund über die Dynamik |
| H3 | Das auszahlungsdominante Gleichgewicht ist oft nicht das, in das man startet | **Ja, verteilungsabhängig** | Zufallsspiele (a, b, c, d) unabhängig gleichverteilt auf [0, 10), nur a > d, b > c, a ≠ b, 4 Ströme zu 100 000 Ziehungen (je Strom rund 25 000 Spiele): in **33,4 %** (Streuung über die Ströme 0,3 Pp) ist das auszahlungsdominante Gleichgewicht risikodominiert, sein mittlerer Einzugsbereich 0,590; mit „sicherem Hasen“ (d = b) **50,0 %** (0,500 Einzugsbereich). Gegenprobe mit vier anderen Strömen zu 400 000 Ziehungen: 33,2 % und 50,2 %. Dieselbe Frage mit allen ganzzahligen Spielen 0 bis 9 (1 500 Spiele): **28,0 %** (420 von 1 500). Bündeln-Spiel (6, 4, 0, 4), n = 50, unabhängiger Start mit 50 %: „alle bündeln“ mit Wahrscheinlichkeit **1,06 %** (exakt; Simulation mit 20 000 Läufen im Test auf 0,4 Pp genau); Hirschjagd (5, 2, 1, 2) 99,98 %; (6, 4, 3, 4) 98,94 %; bei p\* = 1/2 genau 50 % (`test_h3_*`) |
| H4 | Logit-Lernen maximiert das Koordinationspotential; dessen Gipfel ist das risikodominante Gleichgewicht | **Ja** | Stationäre Verteilung der Kette gegen Gibbs-Maß: größte Abweichung unter 10⁻¹² über 120 Zellen (4 Spiele × n ∈ {4, 10, 30, 100, 200} × β ∈ {0,1; 0,5; 1; 2; 5; 10}), die gemessene größte Abweichung ist Rundungsrauschen (größer als 0), und über 300 Zufallsspiele (Orakel). Bündeln-Spiel, n = 50: Masse bei höchstens 10 % Bündelnden **0,114 / 0,998 / mehr als 0,999** für β = 0,5 / 1 / 2, bei mindestens 90 % unter 10⁻⁹. In allen 1 500 ganzzahligen Spielen hat bei n = 50, β = 2 der risikodominante Rand die größere Masse, der auszahlungsdominante nur in 1 080 (72,0 %) (`test_h4_*`) |
| H5 | Im Kampf der Geschlechter hängt das Ziel von den Stärken beider Rollen ab, nicht nur von der Lage in den vier Quadranten | **Ja, richtungsweise; keine Formel** | Replikator, 2 500 Anfangspunkte (Gitter): Anteil „beide Option 0“ **0,49** (Rollenwerte 2 gegen 2), **0,58** (3 gegen 2), 0,49 (4 gegen 4); alle laufen ein. Die Quadranten um das gemischte Gleichgewicht erklären im symmetrischen Fall nur 4/9 = 44 % der Starts direkt. Beste Antwort mit 30 + 30 Spielern, Start 1/2: rund 0,5 / **0,8** / rund 0,5 (Band 0,76 bis 0,86 für 3 gegen 2). Die Startverteilungen beider Verfahren sind verschieden, die Zahlen nur je Verfahren zu lesen (`test_h5_*`) |
| H6 | Streuung der Schwellen öffnet das Kippen; nahe der Grenze entscheidet die Ziehung | **Ja** | Mittlere Schwelle 30 %, kein Anfangsanteil, 1 000 Personen, 100 Ziehungen je Zelle: Endanteil im Mittel 0,0 % (Streuung 5 %), 0,14 % (10 %), 9,2 % (15 %; Fixpunkt 4,4 %), **99,97 %** (20 %), 98,9 % (30 %), 87,4 % (50 %). Bei Streuung 15 % kippt in 4 bis 7 % der Ziehungen mindestens die Hälfte (drei Ströme zu 400 Ziehungen), die meisten enden unter 6 %; das Mittel (8 bis 11 %) liegt über dem Fixpunkt, weil die gekippten Läufe es heben (`test_h6_*`) |
| H7 | Das gemischte Gleichgewicht ist bei s > 0 instabil, bei s < 0 stabil; Absprache gewinnt im Feiglingsspiel ein Mehrfaches der Wohlfahrt | **Ja; der Gewinn hängt an den Auszahlungen** | Replikator von 1 000 Anfangsanteilen: bei (5, 2, 1, 2) und (6, 4, 0, 4) 0 % nahe p\*, 100 % am Rand; bei (−1, 2, 4, 0) und (0, 3, 4, 2) (s = −3) 100 % nahe p\*, 0 % am Rand. Feiglingsspiel V = 4, C = 6: mittlere Auszahlung im gemischten Gleichgewicht 2/3, bei abwechselnder Absprache 2 (ein Drittel); im Spiel (0, 3, 4, 2) 8/3 gegen 3 (8/9). Beste Antwort mit n = 50 bleibt bei 33 Bündelnden stehen; Logit-Mittel des Anteils 0,545 / 0,571 / 0,599 / 0,629 für β = 0,5 / 1 / 2 / 5, von unten gegen V/C = 2/3 (Ursache nicht untersucht) (`test_h7_*`) |

### Korrekturen gegenüber dem Plan

- **Zahl der stabilen Fixpunkte des Schwellenmodells: 1 oder 2, nie 3.** Die Vormessung nannte „drei nur in schmalen Bändern“ (μ = 0,3, σ = 0,05); das war eine **Doppelzählung** (ein Randpunkt nahe 0 und derselbe Vorzeichenwechsel). F(x) = Φ((x − μ)/σ) ist positiv und bleibt unter 1, bei drei Schnittpunkten mit der Diagonale sind zwei stabil. Auf dem Raster μ ∈ {0,2; 0,3; 0,5; 0,7} × σ ∈ {0,02 … 1} sind es in 16 von 32 Zellen zwei, sonst einer; ein zweites Verfahren (Iteration von 81 Startwerten) zählt gleich (`test_h6_stable_fixed_point_counts_are_one_or_two_never_three`).
- **Start ohne Absprache bei p\* = 1/2:** Die Vormessung hatte 0,484 (Simulation, 4 × 400 Läufe); exakt gerechnet ist es **1/2**, eine Simulation mit 8 000 Läufen liegt nahe 0,5. Die Grenze der besten Antwort ist jetzt ebenfalls exakt (Vormessung 0,508 für „gleichauf“, exakt 0,500). Die Kette über die Zahl der Bündelnden ist vollständig durchmischt und braucht keine Zufallsläufe.
- **Auszahlungsregler −10 bis 10 statt 0 bis 10:** Das Feiglingsspiel braucht a = (V − C)/2 = −1.
- **Gewinn durch Absprache:** Die Hypothese „ein Drittel der Wohlfahrt“ gilt für V = 4, C = 6 (2/3 von 2), nicht allgemein: im Spiel (0, 3, 4, 2) sind es 8/9.
- **Tab 6 hat keinen eigenen V/C-Regler:** Er wertet das gewählte Spiel aus (sonst wäre ein Regler im Koordinationsfall ohne Wirkung); der Schnellstart „Feiglingsspiel“ lädt V = 4, C = 6.
- **Zwei-Rollen-Gitter statt Zufallspunkte:** 50 × 50 feste Anfangspunkte, die Spaltenrolle um ein Viertel Zelle versetzt, damit im symmetrischen Spiel die Trennlinie p + q = 1 nicht 50-mal getroffen wird. Dadurch 0,49 statt 0,497 (Vormessung, 10 000 Zufallspunkte) im symmetrischen Fall; der Grenzwert ist 0,5.
- **Anti-Tab:** 1 000 feste Anfangsanteile statt zufälliger Ziehungen (kein Strom nötig).
- **Quellen** aus dem Plan sind nur dort aufgenommen, wo ein Abstract geprüft wurde (siehe unten).

## Modell

Symmetrisches 2×2-Spiel: Auszahlung für mich a (beide bündeln), b (beide direkt), c (ich bündele, der andere fährt direkt), d (ich fahre direkt, der andere bündelt). Gleichgewichte kommen aus einer Aufzählung (Indifferenzbedingungen, exakt in Brüchen), nicht aus der Formel; p\* = (b − c)/((a − d) + (b − c)) ist die Probe. **Klassen:** Koordination (a > d, b > c), Anti-Koordination (a < d, b < c), dominant (verschiedene Vorzeichen), entartet (a = d oder b = c). **Population:** n Betriebe, vollständig durchmischt, Zustand ist die Zahl k der Bündelnden. **Beste Antwort:** ein zufällig gezogener Betrieb wechselt nur bei strikter Verbesserung, ganzzahlig über Zählungen gerechnet; die Wahrscheinlichkeit, bei „alle bündeln“ zu enden, folgt exakt aus der Geburts-Sterbe-Kette. **Replikator:** dx/dt = x(1 − x)(u_S − u_H), Runge-Kutta 4. **Logit:** Wahl von S mit Wahrscheinlichkeit 1/(1 + e^(−βΔ)), Δ = mittlerer Vorteil gegen die n − 1 anderen; die stationäre Verteilung ist das Gibbs-Maß π(k) ∝ C(n, k)·e^(βΦ(k)) des Potentials. **Zwei Rollen:** A = [[r₁, 0], [0, 1]], B = [[1, 0], [0, r₂]]; jede Rolle antwortet auf den Anteil der anderen. **Schwellen:** Person i macht mit, wenn mindestens ihre Schwelle der Personen mitmacht; Normalverteilung, bei 0 gekappt, Vergleich über Personenzahlen. Formeln: 📐-Expander der App.

## Methodik

| Baustein | Datei |
|---|---|
| Klassen, p\*, Dominanz, Potential, Aufzählung der Gleichgewichte (Brüche) | `kos_games.py` |
| Replikator (Runge-Kutta), Grenze des Einzugsbereichs, beste Antwort (ganzzahlig), Absorptionswahrscheinlichkeit (Brüche), Zwei-Rollen-Dynamik | `kos_dynamics.py` |
| Logit-Kette im Logarithmus, Gibbs-Maß | `kos_logit.py` |
| Schwellenmodell, Fixpunkte | `kos_threshold.py` |
| Zufallsspiele, Startzahlen, Rollenmatrizen | `kos_scenario.py`, `kos_rng.py` (SplitMix64, wörtliche Kopie aus `ab-test-demo/abt_rng.py`) |
| Die sechs Experimente E1 bis E6 | `kos_experiments.py` |

Alle Zufallszahlen kommen aus SplitMix64 (gleiche Eingabe, gleiche Zahlen auf jeder Plattform). Laufzeit: das Experiment E2 (etwa eine Sekunde) und E4 (einige Sekunden) rechnen erst auf Knopfdruck, alles andere beim Laden.

## Ehrliche Grenzen

- **Die Häufigkeit in H3 ist eine Eigenschaft der gewählten Verteilung der Auszahlungen** (33,4 % bei unabhängig gleichverteilten, 50,0 % mit „sicherem Hasen“, 28,0 % bei den ganzzahligen Spielen 0 bis 9), kein Naturgesetz.
- 2×2 und vollständige Durchmischung; in Netzen und Nachbarschaften gilt p\* nicht als Einzugsbereich der Population.
- Die beste Antwort und der Replikator haben im 2×2-Fall dieselbe Grenze; das ist eine Probe, keine Entdeckung. Unterschiede zeigen sich nur in der Zwei-Rollen-Fassung, dort mit verschiedenen Startverteilungen.
- Kampf der Geschlechter: gemessen, keine Formel für den Anteil; das Ergebnis des symmetrischen Falls (0,49) ist per Symmetrie 0,5 plus Gitterfehler.
- Das Logit-Mittel im Feiglingsspiel nähert sich V/C von unten; die Ursache (Endlichkeit n, Wahl der Nachbarzahl n − 1) wurde nicht untersucht.
- Granovetter: Normalverteilung der Schwellen und 1 000 Personen sind Annahmen; das Mittel der Endanteile verdeckt die Zweigipfligkeit (die meisten Läufe enden nahe dem Fixpunkt, wenige kippen).
- Mausklicks auf die Diagramme sind nicht getestet (die Diagramme haben keine Auswahl).
- Nur an erfundenen Auszahlungen geprüft; keine Messung an Terminals oder Menschen.

## Quellen

Am Abstract oder an der bibliographischen Angabe geprüft (nicht im Volltext gelesen): Kandori, Mailath, Rob (1993), „Learning, Mutation, and Long Run Equilibria in Games“, *Econometrica* 61(1), 29–56 (in 2×2-Spielen mit zwei strikten Gleichgewichten wird bei großer Population das risikodominante gewählt). Monderer, Shapley (1996), „Potential Games“, *Games and Economic Behavior* 14, 124–143. Blume (1993), „The Statistical Mechanics of Strategic Interaction“, *Games and Economic Behavior* 5(3), 387–424 (Logit-ähnliche Wahl und Gleichgewichtsauswahl in 2×2-Koordinationsspielen). Granovetter (1978), „Threshold Models of Collective Behavior“, *American Journal of Sociology* 83(6), 1420–1443. Harsanyi, Selten (1988), *A General Theory of Equilibrium Selection in Games* (Risikodominanz und Auszahlungsdominanz; Verlag nicht geprüft). **Nicht geprüft und nicht verwendet:** Young (1993), Schelling (1960).

## Verifikation

- **Kern gegen Handrechnung** (`tests/conftest.py`, `test_games.py`, `test_dynamics.py`, `test_logit_threshold.py`): p\* = 1/4 und 2/3, Risikodominanz, Potential, Gleichgewichte des Kampfs der Geschlechter ((2/3, 1/3)), Feiglingsspiel (s = −3, Wohlfahrt 2/3 und 2), Absorptionswahrscheinlichkeiten für n = 3 von Hand, Logit mit n = 2 (π(1)/π(0) = 2·e^(−β)), Granovetter (0, 1, 2) → 3 und (0, 2, 2) → 1, Schwellenwert genau (n = 31, p\* = 1/3).
- **Orakel** (`tests/test_oracle_koordinationsspiel.py`): Support-Enumeration für allgemeine Bimatrix-Spiele mit Brüchen gegen Aufzählung und Formel (1 500 Zufallsspiele inklusive Gleichständen im Repo; alle 10 000 ganzzahligen Spiele in `test_claims.py`); Replikator gegen `scipy.integrate.solve_ivp`; Logit-Kette und Gibbs-Maß gegen die stationäre Verteilung der vollen Übergangsmatrix nach Grassmann-Taksar-Heyman; Absorptionswahrscheinlichkeit gegen die lineare Lösung der vollen Kette (numpy) und gegen eine Simulation mit einzelnen Spielern; Schwellenmodell mit Quantil-Schwellen gegen den Fixpunkt; Zahl stabiler Fixpunkte gegen die Iteration von 81 Startwerten; Zufallsspiele Spiel für Spiel in Brüchen. Zwei Referenzen waren zunächst ungenauer als der Prüfling (`numpy.linalg.eig` und ein lineares System mit ersetzter Zeile für die Kette; Fehler 5·10⁻⁷ bei n = 25) und wurden durch GTH ersetzt.
- **Claims** (`tests/test_claims.py`): jede Zahl dieser README.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, die beiden ▶️-Knöpfe einzeln, jeder Regler am Minimum und Maximum), Permalink, PDF, Presets, Layout, Footer.
- **Mutationsprüfung** (`tools/oracle_mutation.py`, ganze Suite ohne Oberflächen-Tests, ohne `kos_rng.py`): alle 565 Mutationsstellen in `kos_games`, `kos_dynamics`, `kos_logit`, `kos_threshold`, `kos_experiments` und `kos_evaluation` geprüft; 513 von der Suite gefangen, 50 als äquivalent festgehalten (`tests/mutation_state.json`: Wächter, die eine vorangehende Prüfung unerreichbar macht, Konstanten, die sich in der Normierung kürzen, Toleranzstellen, tote Zweige), 2 Zeitüberschreitungen (zählen nicht). Die Orakel-Dateien allein fangen rund die Hälfte, der Rest fällt durch Handrechnung, Claims, Verdikte und Randfälle. Der erste Lauf fand 27 Überlebende in 150 Stellen; Lücken waren die Zwei-Rollen-Routinen mit allgemeinen Matrizen (die Rollenmatrizen haben Nullen außerhalb der Diagonale), Grenzfälle der Zielerkennung, die Schrittzahl des Replikators und die Skalierung der Massenbereiche.

## Tests

278 Tests (`pytest tests`), rund 100 Sekunden lokal; die langsamsten sind die AppTest-Läufe über alle Regler. Zusammensetzung: Kern gegen Handrechnung 90 (`test_games.py` 20, `test_dynamics.py` 49, `test_logit_threshold.py` 21), Orakel 19, Claims 32, Verdikte und Formate 34, Randfälle 21, Szenario und SplitMix64-Kopie 12, Abbildungen und PDF 19, Oberfläche (AppTest, Layout, Presets, Footer, Laufzeit-Importe) 51.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: sechs Tabs mit je einem Experiment |
| `kos_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kos_games.py` | Klassen, p\*, Dominanz, Potential, Aufzählung der Gleichgewichte |
| `kos_dynamics.py` | Replikator, beste Antwort, Absorptionswahrscheinlichkeit, Zwei Rollen |
| `kos_logit.py` | Logit-Kette, Gibbs-Maß |
| `kos_threshold.py` | Schwellenmodell |
| `kos_scenario.py` | Zufallsspiele, Startzahlen, Rollenmatrizen |
| `kos_rng.py` | SplitMix64 (wörtliche Kopie aus `ab-test-demo`) |
| `kos_experiments.py` | Die Experimente E1 bis E6 |
| `kos_evaluation.py` | Einstellungen, Ansichten je Tab, Verdikte als reine Funktionen |
| `kos_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `kos_pdf_export.py` | PDF-Export (fpdf2) |
| `kos_layout.py` | Fester Kopf (Stylesheet) |
| `kos_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Orakel) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handrechnung, Orakel, Claims, Verdikte, Randfälle, Presets, Szenario und Kopie, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Dauerndes Rauschen und Wechselzeit der Konvention:** Die Demo zeigt nur die Identität „stationäre Verteilung = Gibbs-Maß“, nicht, wie oft oder wie schnell eine Konvention kippt.
- **Netze und Nachbarschaften** (siehe `komplexe-ansteckung-demo`), **gemeinsame Signale und Vermittler** (siehe `korreliert-demo`), **Fokalpunkte ohne Absprache**.
- **Menschenexperimente:** Eigennutz und Lernregeln sind Modellannahmen.
- Eine Formel für den Anteil der Ziele im Kampf der Geschlechter: gemessen, nicht behauptet.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Demo läuft mit Python 3.12 oder neuer (CI: 3.12). In der Streamlit Cloud ist die Python-Version in den App-Einstellungen gewählt. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
