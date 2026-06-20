"""
Gera um portfólio HTML interativo de análise de risco de crédito.

Uso:
    python src/portfolio_report.py
    # Abre reports/portfolio_risco_credito.html no navegador
"""

import sys
from pathlib import Path

# garante que src/ está no path ao rodar como script
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from utils import load_data
from risk_analysis import (
    preparar_features_ml,
    criar_target,
    treinar_modelo_risco,
    calcular_score,
    _FEATURE_COLS,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Paleta ────────────────────────────────────────────────────────────────── #
C_LR  = "#4361EE"
C_RF  = "#F77F00"
C_OK  = "#2DC653"
C_ERR = "#E63946"
C_BG  = "#F8F9FA"
C_DARK = "#1B2A4A"


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _fig_to_div(fig: go.Figure, height: str = "420px") -> str:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, sans-serif",
        margin=dict(l=30, r=30, t=50, b=30),
    )
    html = pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"responsive": True, "displayModeBar": False})
    return f'<div style="height:{height}">{html}</div>'


def _kpi_card(titulo: str, valor: str, sub: str = "", color: str = C_DARK) -> str:
    return f"""
    <div class="kpi-card">
      <p class="kpi-label">{titulo}</p>
      <p class="kpi-value" style="color:{color}">{valor}</p>
      <p class="kpi-sub">{sub}</p>
    </div>"""


# ── Figuras ───────────────────────────────────────────────────────────────── #

def fig_roc(resultado, cv):
    X = resultado["df_features"][[c for c in _FEATURE_COLS
                                   if c in resultado["df_features"].columns]].values
    y = resultado["df_features"]["inadimplente"].values

    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dot", color="#aaa", width=1))

    for pipe, label, color in [
        (resultado["logistic_regression"], "Regressão Logística", C_LR),
        (resultado["random_forest"],        "Random Forest",       C_RF),
    ]:
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{label} (AUC={roc_auc:.2f})",
            line=dict(width=2.5, color=color),
            hovertemplate="FPR=%{x:.2f}<br>TPR=%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title="Curva ROC — Comparação de Modelos",
        xaxis_title="Taxa de Falso Positivo",
        yaxis_title="Taxa de Verdadeiro Positivo",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        legend=dict(x=0.55, y=0.1),
    )
    return fig


def fig_auc_bar(resultado):
    fig = go.Figure()
    modelos = ["Reg. Logística", "Random Forest"]
    valores = [resultado["lr_auc_cv"], resultado["rf_auc_cv"]]
    cores   = [C_LR, C_RF]

    fig.add_trace(go.Bar(
        x=modelos, y=valores, marker_color=cores,
        text=[f"{v:.3f}" for v in valores],
        textposition="outside",
        hovertemplate="%{x}<br>AUC=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="AUC-ROC — Validação Cruzada",
        yaxis=dict(range=[0, 1.15], title="AUC-ROC"),
        xaxis_title="Modelo",
        showlegend=False,
    )
    return fig


def fig_feature_importance(resultado):
    imp = resultado["feature_importances"].head(10).sort_values()
    cores = px.colors.sample_colorscale("RdYlGn", [i / len(imp) for i in range(len(imp))])

    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker_color=cores,
        hovertemplate="%{y}<br>Importância=%{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Importância das Features — Random Forest",
        xaxis_title="Importância",
        yaxis_title="",
    )
    return fig


def fig_confusion_matrices(resultado, cv):
    X = resultado["df_features"][[c for c in _FEATURE_COLS
                                   if c in resultado["df_features"].columns]].values
    y = resultado["df_features"]["inadimplente"].values

    labels = ["Adimplente", "Inadimplente"]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Regressão Logística", "Random Forest"],
        horizontal_spacing=0.12,
    )

    for col_idx, (pipe, colorscale) in enumerate([
        (resultado["logistic_regression"], [[0, C_BG], [0.5, C_LR], [1, C_DARK]]),
        (resultado["random_forest"],        [[0, C_BG], [0.5, C_RF], [1, C_DARK]]),
    ], start=1):
        pred = cross_val_predict(pipe, X, y, cv=cv)
        cm = confusion_matrix(y, pred)
        # normaliza para porcentagem
        cm_pct = cm.astype(float) / cm.sum() * 100

        z_text = [[f"{cm[r][c]}<br>({cm_pct[r][c]:.0f}%)"
                   for c in range(2)] for r in range(2)]

        fig.add_trace(go.Heatmap(
            z=cm[::-1], x=labels, y=labels[::-1],
            text=z_text[::-1], texttemplate="%{text}",
            colorscale=colorscale, showscale=False,
            hovertemplate="Real=%{y}<br>Previsto=%{x}<br>N=%{z}<extra></extra>",
        ), row=1, col=col_idx)

    fig.update_layout(title="Matrizes de Confusão (validação cruzada)")
    fig.update_xaxes(title_text="Previsto")
    fig.update_yaxes(title_text="Real", row=1, col=1)
    return fig


def fig_prob_dist(resultado, cv):
    X = resultado["df_features"][[c for c in _FEATURE_COLS
                                   if c in resultado["df_features"].columns]].values
    y = resultado["df_features"]["inadimplente"].values

    fig = go.Figure()
    for pipe, label, color in [
        (resultado["logistic_regression"], "Reg. Logística", C_LR),
        (resultado["random_forest"],        "Random Forest",  C_RF),
    ]:
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        for classe, width, name in [
            (0, 2,   f"{label} — Adimplente"),
            (1, 1,   f"{label} — Inadimplente"),
        ]:
            fig.add_trace(go.Violin(
                x=proba[y == classe],
                name=name,
                line_color=color,
                fillcolor=color,
                opacity=0.5,
                box_visible=True,
                meanline_visible=True,
                line=dict(width=width),
                hoverinfo="x+name",
            ))

    fig.update_layout(
        title="Distribuição de Probabilidade por Classe",
        xaxis_title="P(inadimplente)",
        violinmode="overlay",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def fig_risk_treemap(clientes, transacoes, demonstracoes):
    previsoes = __import__("risk_analysis").prever_risco_ml(
        clientes, transacoes, demonstracoes
    )
    merged = previsoes.merge(
        clientes[["id_cliente", "nome", "segmento", "regiao", "limite_credito"]],
        on="id_cliente",
    )
    merged["prob_pct"] = (merged["prob_inadimplencia"] * 100).round(1)

    fig = px.treemap(
        merged,
        path=["classificacao_ml", "segmento", "nome"],
        values="limite_credito",
        color="prob_inadimplencia",
        color_continuous_scale=["#2DC653", "#F4D35E", "#F77F00", "#E63946"],
        color_continuous_midpoint=0.5,
        title="Mapa de Risco — Exposição por Classificação e Segmento",
        hover_data={"prob_pct": True, "limite_credito": True},
        custom_data=["prob_pct", "limite_credito"],
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>P(inadimplência)=%{customdata[0]}%<br>Limite=R$%{customdata[1]:,.0f}<extra></extra>"
    )
    return fig


def fig_score_vs_ml(clientes, transacoes, demonstracoes):
    score_df = calcular_score(clientes, transacoes, demonstracoes)
    previsoes = __import__("risk_analysis").prever_risco_ml(
        clientes, transacoes, demonstracoes
    )
    merged = score_df.merge(previsoes, on="id_cliente")
    merged = merged.merge(clientes[["id_cliente", "segmento"]], on="id_cliente")

    fig = px.scatter(
        merged,
        x="pontos",
        y="prob_inadimplencia",
        color="segmento",
        size="media_dias_atraso",
        size_max=25,
        hover_data={"id_cliente": True, "classificacao_risco": True, "classificacao_ml": True},
        title="Score por Regras vs Probabilidade ML",
        labels={"pontos": "Score por Regras (0–80)", "prob_inadimplencia": "P(inadimplente) — ML"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Cliente %{customdata[0]}</b><br>"
            "Score regras: %{x}<br>"
            "P(inadimplência): %{y:.2%}<br>"
            "Regras: %{customdata[1]}<br>"
            "ML: %{customdata[2]}<extra></extra>"
        )
    )
    return fig


# ── HTML Template ─────────────────────────────────────────────────────────── #

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfólio — Análise de Risco de Crédito</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg: #F0F4F8;
    --surface: #FFFFFF;
    --dark: {C_DARK};
    --accent: {C_LR};
    --accent2: {C_RF};
    --text: #374151;
    --muted: #6B7280;
    --radius: 14px;
    --shadow: 0 2px 16px rgba(0,0,0,.07);
  }}

  body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}

  /* ── HERO ── */
  .hero {{
    background: linear-gradient(135deg, {C_DARK} 0%, #2A4A8A 60%, {C_LR} 100%);
    color: #fff;
    padding: 64px 48px 48px;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  }}
  .hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.3);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin-bottom: 16px;
  }}
  .hero h1 {{ font-size: clamp(1.8rem, 4vw, 2.8rem); font-weight: 800; line-height: 1.15; }}
  .hero p  {{ max-width: 640px; margin-top: 12px; opacity: .85; font-size: 1.05rem; }}
  .hero-tags {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 24px; }}
  .tag {{
    background: rgba(255,255,255,.15);
    border-radius: 99px;
    padding: 4px 12px;
    font-size: .8rem;
    font-weight: 500;
  }}

  /* ── LAYOUT ── */
  .container {{ max-width: 1280px; margin: 0 auto; padding: 0 24px; }}
  .section {{ padding: 48px 0 16px; }}
  .section-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--dark);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
  }}
  .section-title::before {{
    content: '';
    display: block;
    width: 4px;
    height: 22px;
    border-radius: 99px;
    background: var(--accent);
  }}
  .divider {{
    height: 1px;
    background: linear-gradient(90deg, var(--accent) 0%, transparent 60%);
    margin-bottom: 28px;
    opacity: .25;
  }}

  /* ── KPI CARDS ── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 16px;
    margin-bottom: 8px;
  }}
  .kpi-card {{
    background: var(--surface);
    border-radius: var(--radius);
    padding: 22px 20px;
    box-shadow: var(--shadow);
    border-top: 3px solid var(--accent);
    transition: transform .15s;
  }}
  .kpi-card:hover {{ transform: translateY(-2px); }}
  .kpi-label {{ font-size: .75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
  .kpi-value {{ font-size: 1.9rem; font-weight: 800; margin: 4px 0; line-height: 1; }}
  .kpi-sub   {{ font-size: .78rem; color: var(--muted); }}

  /* ── CHART CARDS ── */
  .grid-2 {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
  }}
  .grid-1 {{ margin-bottom: 20px; }}
  .card {{
    background: var(--surface);
    border-radius: var(--radius);
    padding: 20px 16px;
    box-shadow: var(--shadow);
  }}

  /* ── TABELA ── */
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  thead th {{
    background: var(--dark);
    color: #fff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: .8rem;
    letter-spacing: .04em;
  }}
  tbody tr:nth-child(odd) {{ background: #F9FAFB; }}
  tbody tr:hover {{ background: #EEF2FF; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #E5E7EB; }}
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: .75rem;
    font-weight: 700;
  }}
  .badge-alto     {{ background: #FFE4E6; color: #9B1C1C; }}
  .badge-medio-alto{{ background: #FEF3C7; color: #92400E; }}
  .badge-medio    {{ background: #FEF9C3; color: #713F12; }}
  .badge-baixo    {{ background: #DCFCE7; color: #14532D; }}

  /* ── FOOTER ── */
  footer {{
    text-align: center;
    padding: 40px 20px;
    color: var(--muted);
    font-size: .82rem;
    margin-top: 32px;
    border-top: 1px solid #E5E7EB;
  }}
  footer strong {{ color: var(--dark); }}
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <div class="container">
    <div class="hero-badge">📊 Portfólio de Dados</div>
    <h1>Análise de Risco de Crédito<br>com Machine Learning</h1>
    <p>Modelos preditivos de inadimplência combinando indicadores financeiros,
       comportamento de pagamento e dados cadastrais. Comparação entre
       Regressão Logística e Random Forest com validação cruzada.</p>
    <div class="hero-tags">
      <span class="tag">Python</span>
      <span class="tag">scikit-learn</span>
      <span class="tag">Pandas</span>
      <span class="tag">Plotly</span>
      <span class="tag">Random Forest</span>
      <span class="tag">Regressão Logística</span>
    </div>
  </div>
</div>

<div class="container">

  <!-- KPIs -->
  <div class="section">
    <div class="section-title">Visão Geral da Carteira</div>
    <div class="kpi-grid">
      {kpi_cards}
    </div>
  </div>

  <!-- ROC + AUC -->
  <div class="section">
    <div class="section-title">Performance dos Modelos</div>
    <div class="divider"></div>
    <div class="grid-2">
      <div class="card">{fig_roc}</div>
      <div class="card">{fig_auc}</div>
    </div>
  </div>

  <!-- CONFUSION + PROB DIST -->
  <div class="grid-2">
    <div class="card">{fig_cm}</div>
    <div class="card">{fig_prob}</div>
  </div>

  <!-- FEATURE IMPORTANCE -->
  <div class="section">
    <div class="section-title">Interpretabilidade do Modelo</div>
    <div class="divider"></div>
    <div class="grid-2">
      <div class="card">{fig_feat}</div>
      <div class="card">{fig_scatter}</div>
    </div>
  </div>

  <!-- TREEMAP -->
  <div class="section">
    <div class="section-title">Mapa de Exposição ao Risco</div>
    <div class="divider"></div>
    <div class="grid-1">
      <div class="card">{fig_treemap}</div>
    </div>
  </div>

  <!-- TABELA CLIENTES -->
  <div class="section">
    <div class="section-title">Ranking de Risco — Top 20 Clientes</div>
    <div class="divider"></div>
    <div class="card table-wrap">
      {tabela_html}
    </div>
  </div>

</div>

<footer>
  Gerado automaticamente com <strong>Python + Plotly</strong> &nbsp;·&nbsp;
  Dados sintéticos para demonstração &nbsp;·&nbsp;
  <strong>Portfólio de Análise de Crédito</strong>
</footer>

</body>
</html>
"""


# ── Tabela ────────────────────────────────────────────────────────────────── #

def _tabela_clientes(clientes, transacoes, demonstracoes):
    import risk_analysis as ra
    previsoes = ra.prever_risco_ml(clientes, transacoes, demonstracoes)
    score_df  = ra.calcular_score(clientes, transacoes, demonstracoes)

    merged = (
        previsoes
        .merge(score_df[["id_cliente", "pontos", "classificacao_risco"]], on="id_cliente")
        .merge(clientes[["id_cliente", "nome", "segmento", "regiao", "limite_credito"]], on="id_cliente")
    )
    merged = merged.sort_values("prob_inadimplencia", ascending=False).head(20)

    badge_map = {
        "Alto":       "badge-alto",
        "Médio-Alto": "badge-medio-alto",
        "Médio":      "badge-medio",
        "Baixo":      "badge-baixo",
    }

    rows = ""
    for _, r in merged.iterrows():
        cls = str(r["classificacao_ml"])
        badge_cls = badge_map.get(cls, "badge-baixo")
        rows += f"""<tr>
          <td><strong>{r['id_cliente']}</strong></td>
          <td>{r['nome']}</td>
          <td>{r['segmento']}</td>
          <td>{r['regiao']}</td>
          <td>R$ {r['limite_credito']:,.0f}</td>
          <td>{r['pontos']:.0f}</td>
          <td>{r['prob_inadimplencia']:.1%}</td>
          <td><span class="badge {badge_cls}">{cls}</span></td>
        </tr>"""

    return f"""
    <table>
      <thead><tr>
        <th>#</th><th>Cliente</th><th>Segmento</th><th>Região</th>
        <th>Limite</th><th>Score Regras</th><th>P(inadimplente)</th><th>Risco ML</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ── KPIs ──────────────────────────────────────────────────────────────────── #

def _calcular_kpis(clientes, transacoes, resultado):
    hoje = pd.Timestamp.today()

    n_clientes = len(clientes)
    total_limite = clientes["limite_credito"].sum()

    aberto = transacoes[transacoes["status"] != "Paga"].copy()
    aberto["data_vencimento"] = pd.to_datetime(aberto["data_vencimento"])
    aberto["dias_vencido"] = (hoje - aberto["data_vencimento"]).dt.days
    inadimplentes_ids = aberto[aberto["dias_vencido"] > 30]["id_cliente"].nunique()

    pct_inad = inadimplentes_ids / n_clientes
    valor_em_risco = aberto[aberto["dias_vencido"] > 30]["valor"].sum()

    return [
        _kpi_card("Clientes", f"{n_clientes}", "carteira total"),
        _kpi_card("Inadimplentes", f"{inadimplentes_ids}",
                  f"{pct_inad:.0%} da carteira", C_ERR),
        _kpi_card("Limite Total", f"R$ {total_limite/1e6:.1f}M", "exposição total"),
        _kpi_card("Valor em Risco", f"R$ {valor_em_risco/1e3:.0f}K",
                  "aberto +30 dias", C_RF),
        _kpi_card("AUC-ROC (RF)", f"{resultado['rf_auc_cv']:.3f}",
                  "validação cruzada", C_OK),
        _kpi_card("AUC-ROC (LR)", f"{resultado['lr_auc_cv']:.3f}",
                  "validação cruzada", C_LR),
    ]


# ── Main ──────────────────────────────────────────────────────────────────── #

def gerar_portfolio():
    print("⏳ Carregando dados...")
    clientes, transacoes, demonstracoes, _, _ = load_data()

    print("⏳ Treinando modelos...")
    resultado = treinar_modelo_risco(clientes, transacoes, demonstracoes)

    X = resultado["df_features"][[c for c in _FEATURE_COLS
                                   if c in resultado["df_features"].columns]].values
    y = resultado["df_features"]["inadimplente"].values
    n_splits = max(2, min(5, int(y.sum()), int(len(y) - y.sum())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("⏳ Gerando visualizações...")
    kpis = "".join(_calcular_kpis(clientes, transacoes, resultado))

    html = HTML_TEMPLATE.format(
        C_DARK=C_DARK, C_LR=C_LR, C_RF=C_RF,
        kpi_cards=kpis,
        fig_roc     =_fig_to_div(fig_roc(resultado, cv), "380px"),
        fig_auc     =_fig_to_div(fig_auc_bar(resultado), "380px"),
        fig_cm      =_fig_to_div(fig_confusion_matrices(resultado, cv), "380px"),
        fig_prob    =_fig_to_div(fig_prob_dist(resultado, cv), "380px"),
        fig_feat    =_fig_to_div(fig_feature_importance(resultado), "400px"),
        fig_scatter =_fig_to_div(fig_score_vs_ml(clientes, transacoes, demonstracoes), "400px"),
        fig_treemap =_fig_to_div(fig_risk_treemap(clientes, transacoes, demonstracoes), "480px"),
        tabela_html =_tabela_clientes(clientes, transacoes, demonstracoes),
    )

    out = OUTPUT_DIR / "portfolio_risco_credito.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ Portfólio salvo em: {out}")
    return out


if __name__ == "__main__":
    gerar_portfolio()
