import pandas as pd
from src import blocked_orders


def test_analisar_pedidos_bloqueados_suggests_actions():
    pedidos = pd.DataFrame({'id_pedido': [1, 2, 3], 'id_cliente': [1, 2, 3], 'valor_pedido': [100, 200, 300], 'excede_limite': [False, True, True]})
    score_df = pd.DataFrame({'id_cliente': [1, 2, 3], 'classificacao_risco': ['Baixo', 'Médio-Alto', 'Alto'], 'pontos': [90, 45, 10]})
    out = blocked_orders.analisar_pedidos_bloqueados(pedidos, score_df)
    assert 'acao_sugerida' in out.columns
    assert out.loc[out['id_cliente'] == 1, 'acao_sugerida'].iloc[0].startswith('Liberar')
    assert out.loc[out['id_cliente'] == 2, 'acao_sugerida'].iloc[0].startswith('Solicitar')
    assert 'Bloquear' in out.loc[out['id_cliente'] == 3, 'acao_sugerida'].iloc[0]
