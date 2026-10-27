"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import kos_constants as C
import kos_games as G
import kos_threshold as T


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70 if legend else 10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


# ---------------------------------------------------------------------------------------------------------------------------- Tab 1


def build_matrix(game) -> go.Figure:
    """Auszahlungsmatrix als 2 x 2 Felder mit (meine, andere) Auszahlung; fett: beste Antwort; grüner Rahmen: reines Gleichgewicht."""
    a, b, c, d = game
    A, B = G.payoff_matrices(*game)
    names = ("Bündeln", "Direkt")
    fig = go.Figure()
    ne = {(int(1 - p), int(1 - q)) for p, q in G.enumerate_ne(A, B) if p in (0, 1) and q in (0, 1)}
    for i in (0, 1):
        for j in (0, 1):
            x0, y0 = j, 1 - i
            row_best = A[i][j] >= A[1 - i][j]
            col_best = B[i][j] >= B[i][1 - j]
            r = f"<b>{A[i][j]}</b>" if row_best else f"{A[i][j]}"
            s = f"<b>{B[i][j]}</b>" if col_best else f"{B[i][j]}"
            is_ne = (i, j) in ne
            fig.add_shape(type="rect", x0=x0, x1=x0 + 1, y0=y0, y1=y0 + 1, line=dict(color=C.COLOR_S if is_ne else "rgba(128,128,128,0.6)", width=4 if is_ne else 1),
                          fillcolor="rgba(47,107,101,0.12)" if is_ne else "rgba(0,0,0,0)")
            fig.add_annotation(x=x0 + 0.5, y=y0 + 0.5, text=f"{r} | {s}", showarrow=False, font=dict(size=22))
    for i in (0, 1):
        fig.add_annotation(x=-0.02, y=1 - i + 0.5, text=f"Ich: {names[i]}", showarrow=False, xanchor="right")
        fig.add_annotation(x=i + 0.5, y=2.07, text=f"Anderer: {names[i]}", showarrow=False, yanchor="bottom")
    fig.update_xaxes(range=[-0.7, 2.05], visible=False)
    fig.update_yaxes(range=[-0.05, 2.45], visible=False)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return lock_axes(fig)


def build_payoff_lines(game, p_star) -> go.Figure:
    """Auszahlung von Bündeln und von Direkt in Abhängigkeit vom Anteil der Bündelnden unter den anderen; der Schnittpunkt ist p*."""
    a, b, c, d = game
    x = np.linspace(0, 1, 101)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=x * a + (1 - x) * c, mode="lines", name="Bündeln", line=dict(color=C.COLOR_S, width=3), hovertemplate="Anteil %{x:.2f}: %{y:.2f}<extra>Bündeln</extra>"))
    fig.add_trace(go.Scatter(x=x, y=x * d + (1 - x) * b, mode="lines", name="Direkt fahren", line=dict(color=C.COLOR_H, width=3), hovertemplate="Anteil %{x:.2f}: %{y:.2f}<extra>Direkt</extra>"))
    if p_star is not None:
        px = float(p_star)
        py = px * a + (1 - px) * c
        fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers", name="p*", marker=dict(size=12, color=C.COLOR_RED), hovertemplate="p* = %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Anteil der anderen, die bündeln", range=[0, 1])
    fig.update_yaxes(title_text="Auszahlung (Minuten je Sendung)")
    return _base(fig)


# ---------------------------------------------------------------------------------------------------------------------------- Tab 2


def build_basin_bar(p_star, start_percent: int) -> go.Figure:
    """Anfangsanteil 0 bis 1: links von p* läuft die Dynamik zu 'alle direkt', rechts zu 'alle bündeln'. Die Marke zeigt den gewählten Anfangsanteil."""
    p = float(p_star)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["Einzugsbereich"], x=[p], orientation="h", name="endet bei „alle direkt“", marker_color=C.COLOR_H, hovertemplate="0 bis %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(y=["Einzugsbereich"], x=[1 - p], orientation="h", name="endet bei „alle bündeln“", marker_color=C.COLOR_S, hovertemplate="Breite %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[start_percent / 100], y=["Einzugsbereich"], mode="markers", name="Anfangsanteil", marker=dict(symbol="diamond", size=16, color="white", line=dict(color="black", width=2)),
                             hovertemplate="Anfangsanteil %{x:.2f}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Anfangsanteil der Bündelnden", range=[0, 1])
    fig.update_yaxes(visible=False)
    _base(fig, height=190)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=40), legend=dict(orientation="h", y=1.05, yanchor="bottom"))
    return fig


def build_fan(fan, p_star=None) -> go.Figure:
    """Replikator: Verlauf des Anteils der Bündelnden über die Zeit für mehrere Anfangsanteile."""
    fig = go.Figure()
    for x0, path in fan:
        t = np.linspace(0, 30, len(path))
        fig.add_trace(go.Scatter(x=t, y=path, mode="lines", line=dict(width=2, color=C.COLOR_S if path[-1] > 0.5 else C.COLOR_H), showlegend=False, hovertemplate=f"Start {x0:.2f}: %{{y:.3f}}<extra></extra>"))
    if p_star is not None:
        fig.add_hline(y=float(p_star), line=dict(color=C.COLOR_RED, dash="dash"), annotation_text="p*", annotation_position="right")
    fig.update_xaxes(title_text="Zeit")
    fig.update_yaxes(title_text="Anteil der Bündelnden", range=[-0.02, 1.02])
    return _base(fig, legend=False)


def build_paths(paths, n: int, p_star=None, xtitle: str = "Schritt") -> go.Figure:
    """Verläufe einzelner Läufe (Zahl der Bündelnden k, geteilt durch n)."""
    fig = go.Figure()
    for i, path in enumerate(paths):
        ys = np.array(path) / n
        fig.add_trace(go.Scatter(x=np.arange(len(ys)), y=ys, mode="lines", line=dict(width=2, shape="hv"), name=f"Lauf {i + 1}", showlegend=False, hovertemplate=f"Lauf {i + 1}, Schritt %{{x}}: %{{y:.2f}}<extra></extra>"))
    if p_star is not None:
        fig.add_hline(y=float(p_star), line=dict(color=C.COLOR_RED, dash="dash"), annotation_text="p*", annotation_position="right")
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text="Anteil der Bündelnden", range=[-0.02, 1.02])
    return _base(fig, legend=False)


def build_finals(finals, n: int) -> go.Figure:
    """Endanteile der Läufe als Häufigkeit."""
    counts: dict = {}
    for f in finals:
        counts[f] = counts.get(f, 0) + 1
    xs = sorted(counts)
    fig = go.Figure(go.Bar(x=[f"{x}/{n}" for x in xs], y=[counts[x] for x in xs], marker_color=[C.COLOR_S if x == n else (C.COLOR_H if x == 0 else C.COLOR_LINE) for x in xs],
                           hovertemplate="%{x} Bündelnde: %{y} Läufe<extra></extra>"))
    fig.update_xaxes(title_text="Bündelnde am Ende", type="category")
    fig.update_yaxes(title_text="Läufe")
    return _base(fig, height=260, legend=False)


# ---------------------------------------------------------------------------------------------------------------------------- Tab 3


def build_logit(view) -> go.Figure:
    """Stationäre Verteilung der Logit-Kette (Balken) und Gibbs-Maß des Potentials (Punkte): beide liegen aufeinander."""
    n = view.n
    xs = np.arange(n + 1) / n
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=view.pi, name="Kette (Übergangswahrscheinlichkeiten)", marker_color=C.COLOR_LINE, width=0.6 / n if n > 1 else 0.3, hovertemplate="Anteil %{x:.2f}: %{y:.4f}<extra>Kette</extra>"))
    fig.add_trace(go.Scatter(x=xs, y=view.gibbs, mode="markers", name="Gibbs-Maß (Potential)", marker=dict(color=C.COLOR_RED, size=6, symbol="x"), hovertemplate="Anteil %{x:.2f}: %{y:.4f}<extra>Gibbs</extra>"))
    fig.update_xaxes(title_text="Anteil der Bündelnden", range=[-0.02, 1.02])
    fig.update_yaxes(title_text="Wahrscheinlichkeit")
    return _base(fig)


def build_potential(game, n: int) -> go.Figure:
    """Potential Phi(k) für k = 0..n: die Gipfel sind die Gleichgewichte (im Koordinationsspiel die Ränder)."""
    import kos_logit as L
    ks = np.arange(n + 1)
    phi = [L.potential_of_state(n, int(k), game) for k in ks]
    fig = go.Figure(go.Scatter(x=ks / n, y=phi, mode="lines", line=dict(color=C.COLOR_S, width=3), hovertemplate="Anteil %{x:.2f}: Φ = %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Anteil der Bündelnden", range=[-0.02, 1.02])
    fig.update_yaxes(title_text="Potential Φ")
    return _base(fig, height=260, legend=False)


# ---------------------------------------------------------------------------------------------------------------------------- Tab 4


def build_phase(view) -> go.Figure:
    """Zwei Rollen: Verläufe von zehn Anfangspunkten im Quadrat (Anteil der Rolle 1 mit Option 0, Anteil der Rolle 2 mit Option 0)."""
    fig = go.Figure()
    for i, path in enumerate(view.paths):
        end = path[-1]
        color = C.COLOR_S if (end[0] > 0.5 and end[1] > 0.5) else C.COLOR_H
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(width=2, color=color), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[path[0, 0]], y=[path[0, 1]], mode="markers", marker=dict(size=7, color=color), showlegend=False, hovertemplate="Start (%{x:.2f}; %{y:.2f})<extra></extra>"))
    s = view.shares
    fig.add_trace(go.Scatter(x=[s["p_star"]], y=[s["q_star"]], mode="markers", name="gemischtes Gleichgewicht", marker=dict(symbol="x", size=13, color=C.COLOR_RED), hovertemplate="(p*; q*) = (%{x:.3f}; %{y:.3f})<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="markers", name="reine Gleichgewichte", marker=dict(symbol="circle-open", size=14, color="black", line=dict(width=2)), hovertemplate="(%{x:.0f}; %{y:.0f})<extra></extra>"))
    fig.update_xaxes(title_text="Rolle 1: Anteil mit Option 0", range=[-0.03, 1.03])
    fig.update_yaxes(title_text="Rolle 2: Anteil mit Option 0", range=[-0.03, 1.03])
    return _base(fig, height=420)


# ---------------------------------------------------------------------------------------------------------------------------- Tab 5


def build_threshold_cdf(mu: int, sigma: int, s0: int) -> go.Figure:
    """F(x): Anteil der Personen, deren Schwelle höchstens x ist, gegen die Diagonale; Fixpunkte liegen auf den Schnittstellen. Gestartet wird bei s0."""
    x = np.linspace(0, 1, 201)
    f = np.array([T.normal_cdf(float(v), mu / 100, sigma / 100) for v in x])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=np.maximum(f, s0 / 100), mode="lines", line=dict(color=C.COLOR_S, width=3), name="Anteil, der bei Stand x mitmacht", hovertemplate="x = %{x:.2f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dash"), name="Diagonale (Stand bleibt)", hoverinfo="skip"))
    fp = T.fixed_point_final(mu / 100, sigma / 100, s0 / 100)
    fig.add_trace(go.Scatter(x=[fp], y=[fp], mode="markers", marker=dict(size=12, color=C.COLOR_RED), name="Endstand ab Anfangsanteil", hovertemplate="Endstand %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Anteil, der bereits mitmacht", range=[0, 1])
    fig.update_yaxes(title_text="Anteil, dessen Schwelle erreicht ist", range=[0, 1.02])
    return _base(fig)


def build_threshold_finals(finals) -> go.Figure:
    """Endanteile der Ziehungen (Häufigkeit in Zwanzigstel-Klassen)."""
    edges = np.linspace(0, 1, 21)
    counts, _ = np.histogram(finals, bins=edges)
    fig = go.Figure(go.Bar(x=[f"{edges[i]:.2f}–{edges[i + 1]:.2f}".replace(".", ",") for i in range(20)], y=counts, marker_color=C.COLOR_LINE, hovertemplate="%{x}: %{y} Ziehungen<extra></extra>"))
    fig.update_xaxes(title_text="Endanteil der Mitmachenden", type="category", tickangle=-60)
    fig.update_yaxes(title_text="Ziehungen")
    return _base(fig, height=300, legend=False)


# ---------------------------------------------------------------------------------------------------------------------------- Tab 6


def build_flow(game) -> go.Figure:
    """Änderungsrate dx/dt der Replikatordynamik gegen x: Nullstellen sind die Ruhepunkte; fällt die Kurve dort von + nach -, ist er stabil."""
    import kos_dynamics as D
    x = np.linspace(0, 1, 201)
    y = D.replicator_rhs(x, game)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C.COLOR_LINE, width=3), hovertemplate="Anteil %{x:.2f}: dx/dt = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="gray", dash="dash"))
    ps = G.p_star(*game)
    if ps is not None:
        fig.add_trace(go.Scatter(x=[float(ps)], y=[0.0], mode="markers", marker=dict(size=12, color=C.COLOR_RED), name="p*", hovertemplate="p* = %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Anteil der Bündelnden", range=[0, 1])
    fig.update_yaxes(title_text="Änderung dx/dt")
    return _base(fig, height=300, legend=False)


# ---------------------------------------------------------------------------------------------------------------------------- Experimente


def build_random_games(rows) -> go.Figure:
    """E1: Anteil der Spiele, in denen das auszahlungsdominante Gleichgewicht risikodominiert ist, je Ziehungsart."""
    fig = go.Figure(go.Bar(x=[r["mode"] for r in rows], y=[100 * r["risk_dominated"] for r in rows],
                           error_y=dict(type="data", array=[100 * r["risk_dominated_std"] for r in rows]), marker_color=[C.COLOR_LINE, C.COLOR_H], hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title_text="Auszahlungsdominant, aber risikodominiert (%)", range=[0, 100])
    fig.update_xaxes(title_text="Ziehung der Auszahlungen")
    return _base(fig, height=300, legend=False)


def build_sigma_sweep(rows, mu: int, s0: int) -> go.Figure:
    """E5: Endanteil der Läufe gegen die Streuung der Schwellen (Agenten gegen Fixpunkt)."""
    sel = [r for r in rows if r["mu"] == mu and r["s0"] == s0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["sigma"] for r in sel], y=[r["mean"] for r in sel], mode="lines+markers", name="Personen (Mittel der Ziehungen)", line=dict(color=C.COLOR_S, width=3)))
    fig.add_trace(go.Scatter(x=[r["sigma"] for r in sel], y=[r["fixed_point"] for r in sel], mode="lines+markers", name="Fixpunkt der Dichte", line=dict(color=C.COLOR_RED, dash="dash")))
    fig.update_xaxes(title_text="Streuung der Schwellen (%)")
    fig.update_yaxes(title_text="Endanteil", range=[-0.02, 1.02])
    return _base(fig, height=300)
