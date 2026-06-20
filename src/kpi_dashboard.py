
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from . import utils, aging, risk_analysis


def kpi_report(transacoes: pd.DataFrame) -> dict:
    """Retorna um dicionário com KPIs chave calculados a partir das transações."""
    total_invoices = len(transacoes)
    total_amount = float(transacoes['valor'].sum()) if 'valor' in transacoes.columns else 0.0
    open_mask = transacoes.get('status', pd.Series([None] * len(transacoes))) != 'Paga'
    open_amount = float(transacoes.loc[open_mask, 'valor'].sum()) if 'valor' in transacoes.columns else 0.0
    overdue_mask = (pd.to_datetime(transacoes.get('data_vencimento')) < pd.Timestamp.today()) & (transacoes.get('data_pagamento').isna())
    overdue_ratio = float(overdue_mask.sum() / max(1, total_invoices))
    avg_days_past_due = float((pd.Timestamp.today() - pd.to_datetime(transacoes.get('data_vencimento'))).dt.days.mean())
    if 'valor' in transacoes.columns and transacoes['valor'].sum() > 0:
        dso = float(((pd.Timestamp.today() - pd.to_datetime(transacoes.get('data_vencimento'))).dt.days * transacoes.get('valor', 0)).sum() / transacoes['valor'].sum())
    else:
        dso = 0.0
    return {
        'total_invoices': total_invoices,
        'total_amount': total_amount,
        'open_amount': open_amount,
        'overdue_ratio': overdue_ratio,
        'avg_days_past_due': avg_days_past_due,
        'dso': dso,
    }


def export_kpis_to_excel(kpis: dict, path: str = 'reports/kpis.xlsx') -> None:
    df = pd.DataFrame(list(kpis.items()), columns=['metric', 'value'])
    df.to_excel(path, index=False)



# Inicializar app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de Crédito e Cobrança"),
    
    html.Div([
        html.H2("Visão Geral da Carteira"),
        html.Div([
            dcc.Graph(id='aging-pie'),
            dcc.Graph(id='risco-pie')
        ], style={'display': 'flex', 'flex-direction': 'row'})
    ]),
    
    html.Div([
        html.H2("Detalhamento de Inadimplência"),
        dcc.Graph(id='aging-bar')
    ])
])

@app.callback(
    Output('aging-pie', 'figure'),
    Output('aging-bar', 'figure'),
    Input('aging-pie', 'id')  # dummy input
)
def update_aging(_):
    # Carregar dados e recalcular aging para garantir dados atualizados
    _, transacoes, _, _, _ = utils.load_data()
    aging_summary, _, _ = aging.aging_report(transacoes)

    # Gráfico de pizza do aging
    fig_pie = px.pie(aging_summary, values='valor_total', names='status', title='Composição da Carteira por Status')

    # Gráfico de barras com valores por faixa
    fig_bar = px.bar(aging_summary, x='status', y='valor_total', title='Valores por Faixa de Vencimento')
    return fig_pie, fig_bar

@app.callback(
    Output('risco-pie', 'figure'),
    Input('risco-pie', 'id')
)
def update_risco(_):
    # Recalcula score para garantir que os dados estejam consistentes
    clientes, transacoes, demonstracoes, _, _ = utils.load_data()
    score_df = risk_analysis.calcular_score(clientes, transacoes, demonstracoes)

    risco_count = score_df['classificacao_risco'].value_counts().reset_index()
    risco_count.columns = ['classificacao', 'quantidade']
    fig = px.pie(risco_count, values='quantidade', names='classificacao', title='Distribuição de Risco dos Clientes')
    return fig

if __name__ == '__main__':
    app.run(debug=True)