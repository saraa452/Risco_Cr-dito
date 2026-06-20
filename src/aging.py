import pandas as pd
from datetime import datetime

def aging_report(transacoes, referencia=datetime.today()):
    """
    Gera relatório de aging com base nas transações.
    - 'Pago'
    - 'A Vencer' (dias até vencer > 0)
    - 'Vencido 1-30'
    - 'Vencido 31-60'
    - 'Vencido +60'
    """
    hoje = pd.to_datetime(referencia)
    df = transacoes.copy()
    # Converter datas para datetime se necessário
    df['data_vencimento'] = pd.to_datetime(df['data_vencimento'])
    df['data_pagamento'] = pd.to_datetime(df['data_pagamento'])
    
    # Calcular dias em atraso (negativo = ainda a vencer)
    df['dias_vencimento'] = (hoje - df['data_vencimento']).dt.days
    
    # Classificar status
    def classificar(row):
        if pd.notnull(row['data_pagamento']):
            return 'Pago'
        if row['dias_vencimento'] <= 0:
            return 'A Vencer'
        elif row['dias_vencimento'] <= 30:
            return 'Vencido 1-30'
        elif row['dias_vencimento'] <= 60:
            return 'Vencido 31-60'
        else:
            return 'Vencido +60'
    
    df['status_aging'] = df.apply(classificar, axis=1)
    
    # Resumo por status
    aging_summary = df.groupby('status_aging')['valor'].agg(['sum', 'count']).reset_index()
    aging_summary.columns = ['status', 'valor_total', 'quantidade']
    
    # Detalhamento por cliente (opcional)
    aging_cliente = df[df['status_aging'] != 'Pago'].groupby(['id_cliente', 'status_aging'])['valor'].sum().unstack(fill_value=0)
    
    return aging_summary, aging_cliente, df[['id_cliente', 'data_vencimento', 'valor', 'status_aging']]

def clientes_inadimplentes(transacoes, dias_minimo=1):
    """Lista clientes com pelo menos uma fatura vencida há mais de dias_minimo."""
    hoje = datetime.today()
    df = transacoes.copy()
    df['data_vencimento'] = pd.to_datetime(df['data_vencimento'])
    df['dias_atraso'] = (hoje - df['data_vencimento']).dt.days
    inadimplentes = df[(df['data_pagamento'].isna()) & (df['dias_atraso'] >= dias_minimo)]
    return inadimplentes.groupby('id_cliente').agg({'valor': 'sum', 'dias_atraso': 'max'}).reset_index()