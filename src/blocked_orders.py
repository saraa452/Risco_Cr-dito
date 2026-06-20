import pandas as pd

def analisar_pedidos_bloqueados(pedidos_df, score_df):
    """
    Para cada pedido bloqueado, consulta o score do cliente e sugere ação.
    """
    merged = pedidos_df.merge(score_df[['id_cliente', 'classificacao_risco', 'pontos']], on='id_cliente', how='left')
    
    def sugerir_acao(row):
        if row['classificacao_risco'] in ['Baixo', 'Médio']:
            return 'Liberar mediante análise rápida'
        elif row['classificacao_risco'] == 'Médio-Alto':
            return 'Solicitar análise de crédito'
        else:
            return 'Bloquear - risco alto'
    
    merged['acao_sugerida'] = merged.apply(sugerir_acao, axis=1)
    return merged

def gerar_relatorio_pedidos(pedidos_analise, output='reports/pedidos_bloqueados_analise.xlsx'):
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pedidos_analise.to_excel(writer, sheet_name='Pedidos', index=False)
    print(f"Relatório salvo em {output}")