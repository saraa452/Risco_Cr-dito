"""
Gera o Relatório Executivo de Diagnóstico de Risco de Crédito.

Uso:
    python src/executive_report.py
    # Abre reports/relatorio_executivo.html no navegador
"""

import sys
from pathlib import Path
from datetime import date

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from utils import load_data
from risk_analysis import (
    preparar_features_ml,
    criar_target,
    treinar_modelo_risco,
    calcular_score,
    _FEATURE_COLS,
)
from aging import aging_report, clientes_inadimplentes

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

TODAY     = date.today().strftime("%d/%m/%Y")
C_PRIMARY = "#1B3A6B"
C_DANGER  = "#C0392B"
C_WARN    = "#E67E22"
C_OK      = "#1A7A4A"
C_NEUTRAL = "#455A7A"
C_LIGHT   = "#F4F6FB"
C_BORDER  = "#D1D9E6"


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _div(fig: go.Figure, height="340px") -> str:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#2C3E50", size=12),
        margin=dict(l=30, r=20, t=46, b=28),
    )
    return f'<div style="height:{height}">' + pio.to_html(
        fig, full_html=False, include_plotlyjs=False,
        config={"responsive": True, "displayModeBar": False}
    ) + "</div>"


def _semaforo(valor, limites, labels=("🟢", "🟡", "🔴"), fmt=None):
    """Retorna ícone semáforo e valor formatado."""
    v = fmt(valor) if fmt else str(valor)
    if valor <= limites[0]:
        return labels[0], v
    elif valor <= limites[1]:
        return labels[1], v
    else:
        return labels[2], v


# ── Análises ──────────────────────────────────────────────────────────────── #

def _analise_carteira(clientes, transacoes):
    hoje = pd.Timestamp.today()
    t = transacoes.copy()
    t["data_vencimento"] = pd.to_datetime(t["data_vencimento"])
    t["data_pagamento"]  = pd.to_datetime(t["data_pagamento"])

    aberto = t[t["status"] != "Paga"].copy()
    aberto["dias_vencido"] = (hoje - aberto["data_vencimento"]).dt.days

    total_clientes   = len(clientes)
    total_limite     = clientes["limite_credito"].sum()
    total_faturas    = len(t)
    valor_total      = t["valor"].sum()
    valor_aberto     = aberto["valor"].sum()
    n_vencidas_30    = int((aberto["dias_vencido"] > 30).sum())
    valor_vencido_30 = aberto[aberto["dias_vencido"] > 30]["valor"].sum()
    inad_ids         = aberto[aberto["dias_vencido"] > 30]["id_cliente"].nunique()
    pct_inad         = inad_ids / total_clientes
    pct_valor_risco  = valor_vencido_30 / valor_total if valor_total else 0

    pagas = t[t["status"] == "Paga"].copy()
    pagas["dias_atraso"] = (pagas["data_pagamento"] - pagas["data_vencimento"]).dt.days.clip(lower=0)
    tma = pagas["dias_atraso"].mean()

    return dict(
        total_clientes=total_clientes,
        total_limite=total_limite,
        total_faturas=total_faturas,
        valor_total=valor_total,
        valor_aberto=valor_aberto,
        n_vencidas_30=n_vencidas_30,
        valor_vencido_30=valor_vencido_30,
        inad_ids=inad_ids,
        pct_inad=pct_inad,
        pct_valor_risco=pct_valor_risco,
        tma=tma,
    )


def _analise_pendencias(pendencias, pedidos):
    total_pendencias     = len(pendencias)
    valor_original       = pendencias["valor_original"].sum()
    valor_ajustado       = pendencias["valor_ajustado"].sum()
    perda_pendencias     = valor_original - valor_ajustado
    por_tipo             = pendencias.groupby("tipo")["valor_original"].agg(["sum", "count"])
    n_pedidos_bloqueados = len(pedidos)
    valor_bloqueado      = pedidos["valor_pedido"].sum()
    excesso_total        = pedidos["excede_limite"].sum()
    clientes_bloqueados  = pedidos["id_cliente"].nunique()
    return dict(
        total_pendencias=total_pendencias,
        valor_original=valor_original,
        valor_ajustado=valor_ajustado,
        perda_pendencias=perda_pendencias,
        por_tipo=por_tipo,
        n_pedidos_bloqueados=n_pedidos_bloqueados,
        valor_bloqueado=valor_bloqueado,
        excesso_total=excesso_total,
        clientes_bloqueados=clientes_bloqueados,
    )


def _gerar_achados(cart, pend, resultado):
    achados = []
    if cart["pct_inad"] > 0.20:
        achados.append(("danger", "Alta concentração de inadimplência",
            f"{cart['pct_inad']:.0%} dos clientes possuem faturas vencidas há mais de 30 dias "
            f"({cart['inad_ids']} clientes), representando R$ {cart['valor_vencido_30']:,.0f} em risco."))
    if cart["tma"] > 10:
        achados.append(("warn", "Tempo médio de atraso elevado",
            f"O tempo médio de atraso nos pagamentos é de {cart['tma']:.1f} dias, "
            f"indicando comportamento de pagamento tardio sistêmico."))
    if pend["perda_pendencias"] > 0:
        achados.append(("warn", "Perdas por ajustes em pendências",
            f"As {pend['total_pendencias']} pendências identificadas geraram R$ {pend['perda_pendencias']:,.0f} "
            f"em diferença entre valor original e ajustado."))
    if pend["n_pedidos_bloqueados"] > 0:
        achados.append(("danger", "Pedidos bloqueados por excesso de limite",
            f"{pend['n_pedidos_bloqueados']} pedidos bloqueados de {pend['clientes_bloqueados']} clientes "
            f"totalizando R$ {pend['valor_bloqueado']:,.0f} retidos. "
            f"Excesso de limite acumulado: R$ {pend['excesso_total']:,.0f}."))
    rf_auc = resultado["rf_auc_cv"]
    if rf_auc >= 0.80:
        achados.append(("ok", "Modelo ML com alta capacidade preditiva",
            f"O Random Forest obteve AUC-ROC de {rf_auc:.3f} na validação cruzada, "
            f"demonstrando forte separação entre adimplentes e inadimplentes."))
    top_feat = resultado["feature_importances"].index[0]
    achados.append(("info", f"Principal driver de risco: '{top_feat}'",
        f"A variável de maior importância no modelo é '{top_feat}' "
        f"({resultado['feature_importances'].iloc[0]:.1%}), "
        f"seguida de '{resultado['feature_importances'].index[1]}' "
        f"({resultado['feature_importances'].iloc[1]:.1%})."))
    return achados


# ── Gráficos ──────────────────────────────────────────────────────────────── #

def _fig_aging_donut(transacoes):
    aging_summary, _, _ = aging_report(transacoes)
    color_map = {
        "Pago":           "#2DC653",
        "A Vencer":       "#4361EE",
        "Vencido 1-30":   "#F4C430",
        "Vencido 31-60":  "#E67E22",
        "Vencido +60":    "#C0392B",
    }
    cores = [color_map.get(s, "#aaa") for s in aging_summary["status"]]
    fig = go.Figure(go.Pie(
        labels=aging_summary["status"],
        values=aging_summary["valor_total"],
        hole=0.58,
        marker_colors=cores,
        textinfo="label+percent",
        hovertemplate="%{label}<br>R$ %{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(title="Composição da Carteira por Status (Aging)", showlegend=False)
    return fig


def _fig_risco_barras(clientes, transacoes, demonstracoes):
    score_df = calcular_score(clientes, transacoes, demonstracoes)
    contagem = score_df["classificacao_risco"].value_counts().reindex(
        ["Alto", "Médio-Alto", "Médio", "Baixo"], fill_value=0
    )
    cores = [C_DANGER, C_WARN, "#F4C430", C_OK]
    fig = go.Figure(go.Bar(
        x=contagem.index, y=contagem.values,
        marker_color=cores,
        text=contagem.values, textposition="outside",
        hovertemplate="%{x}: %{y} clientes<extra></extra>",
    ))
    fig.update_layout(
        title="Distribuição de Risco por Regras",
        yaxis_title="Nº de Clientes",
        xaxis_title="Classificação",
        showlegend=False,
        yaxis=dict(range=[0, contagem.max() * 1.2]),
    )
    return fig


def _fig_pendencias_tipo(pendencias):
    pt = pendencias.groupby("tipo").agg(
        valor=("valor_original", "sum"),
        quantidade=("id_pendencia", "count"),
    ).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=pt["tipo"], y=pt["valor"], name="Valor (R$)",
        marker_color=C_WARN,
        hovertemplate="%{x}<br>R$ %{y:,.0f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=pt["tipo"], y=pt["quantidade"], name="Quantidade",
        mode="markers+lines",
        marker=dict(size=10, color=C_PRIMARY),
        line=dict(dash="dot", color=C_PRIMARY),
        hovertemplate="%{x}<br>%{y} ocorrências<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(title="Pendências por Tipo — Valor e Quantidade", legend=dict(x=0.7, y=1.1))
    fig.update_yaxes(title_text="Valor (R$)", secondary_y=False)
    fig.update_yaxes(title_text="Quantidade", secondary_y=True)
    return fig


def _fig_inad_segmento(clientes, transacoes):
    hoje = pd.Timestamp.today()
    t = transacoes.copy()
    t["data_vencimento"] = pd.to_datetime(t["data_vencimento"])
    aberto = t[t["status"] != "Paga"].copy()
    aberto["dias_vencido"] = (hoje - aberto["data_vencimento"]).dt.days
    inad = aberto[aberto["dias_vencido"] > 30][["id_cliente", "valor"]].copy()
    inad = inad.merge(clientes[["id_cliente", "segmento"]], on="id_cliente", how="left")
    grp = inad.groupby("segmento")["valor"].agg(["sum", "count"]).reset_index()
    grp.columns = ["segmento", "valor", "n_faturas"]

    fig = px.bar(
        grp, x="segmento", y="valor", color="segmento",
        text="n_faturas",
        title="Valor Inadimplente por Segmento (> 30 dias)",
        labels={"valor": "R$ em atraso", "segmento": "Segmento"},
        color_discrete_sequence=[C_DANGER, C_WARN, C_NEUTRAL, "#8E44AD"],
    )
    fig.update_traces(texttemplate="%{text} fatura(s)", textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


def _fig_ml_prob_ranking(clientes, transacoes, demonstracoes):
    import risk_analysis as ra
    prev = ra.prever_risco_ml(clientes, transacoes, demonstracoes)
    prev = prev.merge(clientes[["id_cliente", "nome", "segmento"]], on="id_cliente")

    color_map = {"Baixo": C_OK, "Médio": "#F4C430", "Médio-Alto": C_WARN, "Alto": C_DANGER}
    prev["cor"] = prev["classificacao_ml"].astype(str).map(color_map).fillna(C_NEUTRAL)
    prev = prev.sort_values("prob_inadimplencia")

    fig = go.Figure(go.Bar(
        x=prev["prob_inadimplencia"],
        y=prev["nome"],
        orientation="h",
        marker_color=prev["cor"],
        text=[f"{v:.0%}" for v in prev["prob_inadimplencia"]],
        textposition="outside",
        hovertemplate="%{y}<br>P(inadimplência)=%{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title="Probabilidade de Inadimplência por Cliente (Random Forest)",
        xaxis=dict(tickformat=".0%", range=[0, 1.15]),
        yaxis=dict(tickfont=dict(size=10)),
        height=max(400, len(prev) * 18),
    )
    return fig


def _fig_pedidos_bloqueados(pedidos, clientes):
    df = pedidos.merge(clientes[["id_cliente", "nome", "segmento"]], on="id_cliente", how="left")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Valor do Pedido",
        x=df["nome"], y=df["valor_pedido"],
        marker_color=C_DANGER, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Saldo Devedor",
        x=df["nome"], y=df["saldo_devedor"],
        marker_color=C_WARN, opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        name="Limite",
        x=df["nome"], y=df["limite"],
        mode="markers", marker=dict(symbol="line-ew", size=14, color=C_PRIMARY, line_width=2),
    ))
    fig.update_layout(
        title="Pedidos Bloqueados — Pedido vs Saldo vs Limite",
        barmode="group",
        xaxis_title="Cliente",
        yaxis_title="R$",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ── HTML ──────────────────────────────────────────────────────────────────── #

STYLE = f"""
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --primary: {C_PRIMARY};
    --danger: {C_DANGER};
    --warn: {C_WARN};
    --ok: {C_OK};
    --neutral: {C_NEUTRAL};
    --light: {C_LIGHT};
    --border: {C_BORDER};
    --text: #2C3E50;
    --muted: #6B7280;
    --radius: 8px;
    --shadow: 0 1px 6px rgba(0,0,0,.08);
  }}
  body {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: #EAEFF6;
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
  }}

  /* HEADER */
  .report-header {{
    background: var(--primary);
    color: #fff;
    padding: 0;
    border-bottom: 5px solid #F4C430;
  }}
  .header-inner {{
    max-width: 1160px;
    margin: 0 auto;
    padding: 32px 40px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 24px;
  }}
  .report-titulo {{ font-size: 1.55rem; font-weight: 800; letter-spacing: -.01em; }}
  .report-subtitulo {{ font-size: .92rem; opacity: .8; margin-top: 4px; }}
  .report-meta {{ text-align: right; font-size: .8rem; opacity: .75; line-height: 1.8; flex-shrink: 0; }}
  .confidencial {{
    display: inline-block;
    border: 1px solid rgba(255,255,255,.4);
    padding: 3px 10px;
    border-radius: 4px;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-top: 6px;
  }}

  /* LAYOUT */
  .container {{ max-width: 1160px; margin: 0 auto; padding: 0 40px 60px; }}

  /* SUMÁRIO EXECUTIVO */
  .exec-summary {{
    background: #fff;
    border-left: 5px solid var(--primary);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin: 32px 0 24px;
    box-shadow: var(--shadow);
  }}
  .exec-summary h2 {{ font-size: 1rem; font-weight: 700; color: var(--primary); margin-bottom: 12px; letter-spacing: .04em; text-transform: uppercase; }}
  .exec-summary p {{ color: #374151; font-size: .9rem; line-height: 1.7; }}

  /* SECTION */
  .section {{ margin-bottom: 32px; }}
  .section-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0 8px;
    border-bottom: 2px solid var(--primary);
    margin-bottom: 18px;
  }}
  .section-num {{
    background: var(--primary);
    color: #fff;
    font-size: .78rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 4px;
  }}
  .section-title {{ font-size: 1rem; font-weight: 700; color: var(--primary); text-transform: uppercase; letter-spacing: .05em; }}

  /* KPI STRIP */
  .kpi-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 18px; }}
  .kpi-box {{
    background: #fff;
    border-radius: var(--radius);
    padding: 16px 18px;
    box-shadow: var(--shadow);
    border-top: 3px solid var(--primary);
  }}
  .kpi-box.danger {{ border-color: var(--danger); }}
  .kpi-box.warn   {{ border-color: var(--warn); }}
  .kpi-box.ok     {{ border-color: var(--ok); }}
  .kpi-box .kpi-l {{ font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }}
  .kpi-box .kpi-v {{ font-size: 1.6rem; font-weight: 800; line-height: 1.1; margin: 4px 0 2px; }}
  .kpi-box .kpi-s {{ font-size: .72rem; color: var(--muted); }}
  .kpi-box.danger .kpi-v {{ color: var(--danger); }}
  .kpi-box.warn   .kpi-v {{ color: var(--warn); }}
  .kpi-box.ok     .kpi-v {{ color: var(--ok); }}

  /* ACHADOS */
  .achados-grid {{ display: flex; flex-direction: column; gap: 10px; margin-bottom: 18px; }}
  .achado {{
    display: flex;
    gap: 14px;
    padding: 14px 18px;
    border-radius: var(--radius);
    background: #fff;
    box-shadow: var(--shadow);
    border-left: 4px solid transparent;
    align-items: flex-start;
  }}
  .achado.danger {{ border-color: var(--danger); background: #FEF2F2; }}
  .achado.warn   {{ border-color: var(--warn);   background: #FFFBEB; }}
  .achado.ok     {{ border-color: var(--ok);     background: #F0FDF4; }}
  .achado.info   {{ border-color: var(--primary); background: #EFF6FF; }}
  .achado-icon   {{ font-size: 1.3rem; flex-shrink: 0; margin-top: 1px; }}
  .achado-titulo {{ font-weight: 700; font-size: .88rem; margin-bottom: 3px; }}
  .achado.danger .achado-titulo {{ color: var(--danger); }}
  .achado.warn   .achado-titulo {{ color: #92400E; }}
  .achado.ok     .achado-titulo {{ color: #14532D; }}
  .achado.info   .achado-titulo {{ color: var(--primary); }}
  .achado-desc   {{ font-size: .83rem; color: #374151; }}

  /* CHART GRID */
  .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; margin-bottom: 16px; }}
  .grid-1 {{ margin-bottom: 16px; }}
  .card {{
    background: #fff;
    border-radius: var(--radius);
    padding: 16px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
  }}

  /* RECOMENDAÇÕES */
  .rec-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
  .rec-card {{
    background: #fff;
    border-radius: var(--radius);
    padding: 18px 20px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
  }}
  .rec-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .rec-icon {{ font-size: 1.4rem; }}
  .rec-titulo {{ font-weight: 700; font-size: .9rem; color: var(--primary); }}
  .rec-lista {{ padding-left: 18px; }}
  .rec-lista li {{ font-size: .82rem; color: #374151; margin-bottom: 5px; line-height: 1.5; }}
  .prioridade {{
    display: inline-block;
    font-size: .65rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 99px;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-left: 6px;
    vertical-align: middle;
  }}
  .prioridade.alta   {{ background: #FEE2E2; color: #991B1B; }}
  .prioridade.media  {{ background: #FEF3C7; color: #92400E; }}
  .prioridade.baixa  {{ background: #DCFCE7; color: #14532D; }}

  /* TABELA */
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
  thead th {{ background: var(--primary); color: #fff; padding: 9px 12px; text-align: left; font-weight: 600; font-size: .78rem; letter-spacing: .04em; }}
  tbody tr:nth-child(odd) {{ background: #F8FAFC; }}
  tbody tr:hover {{ background: #EEF2FF; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  .badge {{ display: inline-block; padding: 2px 9px; border-radius: 99px; font-size: .72rem; font-weight: 700; }}
  .badge-alto      {{ background: #FEE2E2; color: #991B1B; }}
  .badge-medio-alto {{ background: #FFEDD5; color: #9A3412; }}
  .badge-medio     {{ background: #FEF9C3; color: #713F12; }}
  .badge-baixo     {{ background: #DCFCE7; color: #14532D; }}

  /* FOOTER */
  footer {{
    background: var(--primary);
    color: rgba(255,255,255,.6);
    text-align: center;
    padding: 18px 20px;
    font-size: .78rem;
  }}
  footer strong {{ color: #fff; }}

  @media print {{
    body {{ background: #fff; font-size: 12px; }}
    .report-header {{ border-bottom-width: 3px; }}
    .card {{ box-shadow: none; border: 1px solid #ccc; }}
  }}
</style>
"""


def _tabela_risco_clientes(clientes, transacoes, demonstracoes):
    import risk_analysis as ra
    prev   = ra.prever_risco_ml(clientes, transacoes, demonstracoes)
    score  = ra.calcular_score(clientes, transacoes, demonstracoes)
    merged = (
        prev
        .merge(score[["id_cliente", "pontos", "classificacao_risco", "media_dias_atraso"]], on="id_cliente")
        .merge(clientes[["id_cliente", "nome", "segmento", "regiao", "limite_credito"]], on="id_cliente")
        .sort_values("prob_inadimplencia", ascending=False)
        .head(15)
    )
    badge_map = {
        "Alto": "badge-alto", "Médio-Alto": "badge-medio-alto",
        "Médio": "badge-medio", "Baixo": "badge-baixo",
    }
    rows = ""
    for i, r in enumerate(merged.itertuples(), 1):
        cls   = str(r.classificacao_ml)
        badge = badge_map.get(cls, "badge-baixo")
        rows += f"""<tr>
          <td>{i}</td>
          <td><strong>{r.nome}</strong></td>
          <td>{r.segmento}</td>
          <td>{r.regiao}</td>
          <td>R$ {r.limite_credito:,.0f}</td>
          <td>{r.media_dias_atraso:.1f} dias</td>
          <td>{r.pontos:.0f}</td>
          <td>{r.prob_inadimplencia:.1%}</td>
          <td><span class="badge {badge}">{cls}</span></td>
        </tr>"""
    return f"""<table>
      <thead><tr>
        <th>#</th><th>Cliente</th><th>Segmento</th><th>Região</th>
        <th>Limite</th><th>Atraso Médio</th><th>Score</th>
        <th>P(inadimplente)</th><th>Risco ML</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _recomendacoes(cart, pend):
    return [
        dict(
            icon="🚨", titulo="Cobrança Ativa Imediata",
            prioridade="alta",
            itens=[
                f"Acionar os {cart['inad_ids']} clientes inadimplentes com faturas vencidas há +30 dias.",
                f"Priorizar os {min(5, cart['inad_ids'])} maiores exposições — responsáveis pela maior concentração de R$ {cart['valor_vencido_30']:,.0f}.",
                "Escalar para cobrança jurídica os casos com atraso superior a 90 dias.",
            ]
        ),
        dict(
            icon="🔒", titulo="Gestão de Pedidos Bloqueados",
            prioridade="alta",
            itens=[
                f"Revisar os {pend['n_pedidos_bloqueados']} pedidos bloqueados antes de liberar crédito adicional.",
                "Exigir regularização de saldo devedor como pré-requisito para desbloqueio.",
                f"Avaliar renegociação de limite para clientes com excesso de R$ {pend['excesso_total']:,.0f}.",
            ]
        ),
        dict(
            icon="📋", titulo="Regularização de Pendências",
            prioridade="media",
            itens=[
                f"Processar os {pend['total_pendencias']} ajustes pendentes para evitar acúmulo contábil.",
                f"Investigar R$ {pend['perda_pendencias']:,.0f} em diferença entre valor original e ajustado.",
                "Implementar reconciliação semanal para reduzir estoque de pendências.",
            ]
        ),
        dict(
            icon="🤖", titulo="Uso do Modelo Preditivo",
            prioridade="media",
            itens=[
                "Incorporar o score ML na rotina de análise de crédito para novos pedidos.",
                "Definir threshold de corte (ex.: P > 0.5 requer aprovação manual).",
                "Retroalimentar o modelo mensalmente com novos dados de pagamento.",
            ]
        ),
        dict(
            icon="📊", titulo="Monitoramento Contínuo",
            prioridade="baixa",
            itens=[
                "Estabelecer KPI mensal de inadimplência com meta ≤ 15% da carteira.",
                "Revisar política de limites de crédito semestralmente por segmento.",
                "Criar alerta automático quando TMA (tempo médio de atraso) ultrapassar 7 dias.",
            ]
        ),
    ]


def _html_achado(tipo, titulo, descricao):
    icon_map = {"danger": "⛔", "warn": "⚠️", "ok": "✅", "info": "ℹ️"}
    return f"""<div class="achado {tipo}">
      <span class="achado-icon">{icon_map.get(tipo, "•")}</span>
      <div>
        <div class="achado-titulo">{titulo}</div>
        <div class="achado-desc">{descricao}</div>
      </div>
    </div>"""


def _html_rec(rec):
    p = rec["prioridade"]
    itens = "".join(f"<li>{i}</li>" for i in rec["itens"])
    return f"""<div class="rec-card">
      <div class="rec-header">
        <span class="rec-icon">{rec['icon']}</span>
        <span class="rec-titulo">{rec['titulo']} <span class="prioridade {p}">{p}</span></span>
      </div>
      <ul class="rec-lista">{itens}</ul>
    </div>"""


def _html_kpi(label, valor, sub="", cls=""):
    return f"""<div class="kpi-box {cls}">
      <div class="kpi-l">{label}</div>
      <div class="kpi-v">{valor}</div>
      <div class="kpi-s">{sub}</div>
    </div>"""


# ── Main ──────────────────────────────────────────────────────────────────── #

def gerar_relatorio():
    print("⏳ Carregando dados...")
    clientes, transacoes, demonstracoes, pendencias, pedidos = load_data()

    print("⏳ Processando análises...")
    cart    = _analise_carteira(clientes, transacoes)
    pend    = _analise_pendencias(pendencias, pedidos)
    result  = treinar_modelo_risco(clientes, transacoes, demonstracoes)
    achados = _gerar_achados(cart, pend, result)
    recs    = _recomendacoes(cart, pend)

    print("⏳ Gerando visualizações...")

    sumario = (
        f"A análise diagnóstica da carteira de crédito, realizada em {TODAY}, identificou "
        f"<strong>{cart['inad_ids']} clientes com faturas vencidas há mais de 30 dias</strong> "
        f"({cart['pct_inad']:.0%} da carteira), representando <strong>R$ {cart['valor_vencido_30']:,.0f}</strong> "
        f"em risco imediato. O tempo médio de atraso nos pagamentos é de <strong>{cart['tma']:.1f} dias</strong>. "
        f"Adicionalmente, foram detectados <strong>{pend['n_pedidos_bloqueados']} pedidos bloqueados</strong> "
        f"por excesso de limite e <strong>{pend['total_pendencias']} pendências financeiras</strong> a regularizar. "
        f"O modelo de Machine Learning (Random Forest) obteve AUC-ROC de <strong>{result['rf_auc_cv']:.3f}</strong>, "
        f"demonstrando alta capacidade preditiva. São recomendadas ações imediatas de cobrança ativa "
        f"e revisão dos pedidos bloqueados."
    )

    kpis = "".join([
        _html_kpi("Total de Clientes",     f"{cart['total_clientes']}",                   "carteira ativa"),
        _html_kpi("Inadimplentes >30d",    f"{cart['inad_ids']}",                         f"{cart['pct_inad']:.0%} da carteira", "danger"),
        _html_kpi("Valor em Risco",        f"R$ {cart['valor_vencido_30']/1e3:.0f}K",     "faturas vencidas", "danger"),
        _html_kpi("Tempo Médio Atraso",    f"{cart['tma']:.1f} dias",                     "pagamentos realizados",
                  "warn" if cart["tma"] > 10 else "ok"),
        _html_kpi("Pedidos Bloqueados",    f"{pend['n_pedidos_bloqueados']}",              f"R$ {pend['valor_bloqueado']/1e3:.0f}K retido", "warn"),
        _html_kpi("Pendências",            f"{pend['total_pendencias']}",                  f"Δ R$ {pend['perda_pendencias']:,.0f}", "warn"),
        _html_kpi("AUC-ROC (RF)",          f"{result['rf_auc_cv']:.3f}",                  "modelo preditivo", "ok"),
    ])

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatório Executivo — Risco de Crédito</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
{STYLE}
</head>
<body>

<header class="report-header">
  <div class="header-inner">
    <div>
      <div class="report-titulo">Relatório Executivo — Análise Diagnóstica de Risco de Crédito</div>
      <div class="report-subtitulo">Carteira de Crédito e Cobrança · Diagnóstico de Inadimplência e Exposição</div>
    </div>
    <div class="report-meta">
      Data de emissão: <strong>{TODAY}</strong><br>
      Período analisado: <strong>Histórico completo</strong><br>
      Modelo: <strong>Random Forest + Reg. Logística</strong><br>
      <span class="confidencial">⚠ Uso Interno</span>
    </div>
  </div>
</header>

<div class="container">

  <!-- SUMÁRIO EXECUTIVO -->
  <div class="exec-summary">
    <h2>🗂 Sumário Executivo</h2>
    <p>{sumario}</p>
  </div>

  <!-- 1. KPIs -->
  <div class="section">
    <div class="section-header">
      <span class="section-num">01</span>
      <span class="section-title">Indicadores-Chave da Carteira</span>
    </div>
    <div class="kpi-strip">{kpis}</div>
  </div>

  <!-- 2. ACHADOS -->
  <div class="section">
    <div class="section-header">
      <span class="section-num">02</span>
      <span class="section-title">Principais Achados Diagnósticos</span>
    </div>
    <div class="achados-grid">
      {"".join(_html_achado(t, ti, d) for t, ti, d in achados)}
    </div>
  </div>

  <!-- 3. ANÁLISE DE CARTEIRA -->
  <div class="section">
    <div class="section-header">
      <span class="section-num">03</span>
      <span class="section-title">Análise da Carteira — Aging e Risco</span>
    </div>
    <div class="grid-2">
      <div class="card">{_div(_fig_aging_donut(transacoes), "340px")}</div>
      <div class="card">{_div(_fig_risco_barras(clientes, transacoes, demonstracoes), "340px")}</div>
    </div>
    <div class="grid-2">
      <div class="card">{_div(_fig_inad_segmento(clientes, transacoes), "320px")}</div>
      <div class="card">{_div(_fig_pedidos_bloqueados(pedidos, clientes), "320px")}</div>
    </div>
  </div>

  <!-- 4. MODELO ML -->
  <div class="section">
    <div class="section-header">
      <span class="section-num">04</span>
      <span class="section-title">Resultados do Modelo Preditivo</span>
    </div>
    <div class="grid-1">
      <div class="card">{_div(_fig_ml_prob_ranking(clientes, transacoes, demonstracoes), "520px")}</div>
    </div>
    <div class="grid-1">
      <div class="card">{_div(_fig_pendencias_tipo(pendencias), "320px")}</div>
    </div>
  </div>

  <!-- 5. TABELA -->
  <div class="section">
    <div class="section-header">
      <span class="section-num">05</span>
      <span class="section-title">Top 15 Clientes por Risco de Inadimplência</span>
    </div>
    <div class="card table-wrap">
      {_tabela_risco_clientes(clientes, transacoes, demonstracoes)}
    </div>
  </div>

  <!-- 6. RECOMENDAÇÕES -->
  <div class="section">
    <div class="section-header">
      <span class="section-num">06</span>
      <span class="section-title">Recomendações e Plano de Ação</span>
    </div>
    <div class="rec-grid">
      {"".join(_html_rec(r) for r in recs)}
    </div>
  </div>

</div>

<footer>
  <strong>Relatório Executivo — Análise de Risco de Crédito</strong> &nbsp;·&nbsp;
  Gerado em {TODAY} &nbsp;·&nbsp; Dados sintéticos para demonstração &nbsp;·&nbsp;
  Uso interno exclusivo
</footer>

</body>
</html>"""

    out = OUTPUT_DIR / "relatorio_executivo.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ Relatório salvo em: {out}")
    return out


if __name__ == "__main__":
    gerar_relatorio()
