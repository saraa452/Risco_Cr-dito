import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_clients(n=100):
    """Gera uma base de clientes."""
    clientes = []
    for i in range(1, n+1):
        cliente = {
            'id_cliente': i,
            'nome': f'Cliente {i}',
            'segmento': np.random.choice(['Varejo', 'Atacado', 'Indústria'], p=[0.5, 0.3, 0.2]),
            'regiao': np.random.choice(['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']),
            'data_cadastro': datetime.now() - timedelta(days=np.random.randint(30, 365*5)),
            'limite_credito': np.random.choice([5000, 10000, 20000, 50000, 100000], p=[0.3,0.3,0.2,0.15,0.05]),
            'score_inicial': np.random.randint(0, 101)
        }
        clientes.append(cliente)
    return pd.DataFrame(clientes)

def generate_transactions(clients_df, n=1000):
    """Gera transações (faturas) para os clientes."""
    transactions = []
    start_date = datetime.now() - timedelta(days=180)
    for i in range(1, n+1):
        cliente = clients_df.sample(1).iloc[0]
        data_vencimento = start_date + timedelta(days=np.random.randint(0, 180))
        valor = np.random.choice([500, 1000, 2000, 5000, 10000], p=[0.4,0.3,0.15,0.1,0.05])
        pago = np.random.choice([True, False], p=[0.7, 0.3])  # 70% pagas
        if pago:
            data_pagamento = data_vencimento + timedelta(days=np.random.randint(-5, 30))
            if data_pagamento < data_vencimento:
                data_pagamento = data_vencimento  # não pode pagar antes? pode, mas simplificamos
        else:
            data_pagamento = None
        trans = {
            'id_transacao': i,
            'id_cliente': cliente['id_cliente'],
            'data_vencimento': data_vencimento,
            'valor': valor,
            'data_pagamento': data_pagamento,
            'status': 'Paga' if pago else 'Em aberto'
        }
        transactions.append(trans)
    return pd.DataFrame(transactions)

def generate_financial_statements(clients_df):
    """Gera demonstrações financeiras simplificadas para cada cliente."""
    statements = []
    for _, cliente in clients_df.iterrows():
        ativo_circulante = np.random.randint(100000, 1000000)
        passivo_circulante = np.random.randint(50000, 800000)
        ativo_total = ativo_circulante + np.random.randint(200000, 2000000)
        passivo_total = passivo_circulante + np.random.randint(100000, 1500000)
        receita = np.random.randint(500000, 5000000)
        lucro_liquido = receita * np.random.uniform(0.05, 0.2)
        statements.append({
            'id_cliente': cliente['id_cliente'],
            'ano': datetime.now().year - 1,
            'ativo_circulante': ativo_circulante,
            'passivo_circulante': passivo_circulante,
            'ativo_total': ativo_total,
            'passivo_total': passivo_total,
            'receita_liquida': receita,
            'lucro_liquido': lucro_liquido
        })
    return pd.DataFrame(statements)

def generate_pending_items(transactions_df):
    """Gera itens que precisam de ajuste (ex.: abatimentos, créditos)."""
    pend = []
    # Seleciona algumas transações pagas para sugerir abatimento (ex.: valor pago menor)
    pagas = transactions_df[transactions_df['status'] == 'Paga'].sample(frac=0.05)
    for _, row in pagas.iterrows():
        pend.append({
            'id_pendencia': len(pend)+1,
            'id_cliente': row['id_cliente'],
            'id_transacao': row['id_transacao'],
            'tipo': np.random.choice(['Abatimento', 'Crédito', 'Estorno']),
            'descricao': 'Valor divergente' if np.random.rand()>0.5 else 'Duplicidade',
            'valor_original': row['valor'],
            'valor_ajustado': row['valor'] * np.random.uniform(0.8, 0.95)
        })
    return pd.DataFrame(pend)

def generate_blocked_orders(clients_df, transactions_df):
    """Gera pedidos bloqueados por limite de crédito."""
    blocked = []
    for i in range(20):
        cliente = clients_df.sample(1).iloc[0]
        # Verificar se cliente tem limite e saldo devedor
        saldo_devedor = transactions_df[(transactions_df['id_cliente']==cliente['id_cliente']) & (transactions_df['status']=='Em aberto')]['valor'].sum()
        valor_pedido = np.random.randint(1000, 50000)
        if saldo_devedor + valor_pedido > cliente['limite_credito']:
            blocked.append({
                'id_pedido': i+1,
                'id_cliente': cliente['id_cliente'],
                'valor_pedido': valor_pedido,
                'saldo_devedor': saldo_devedor,
                'limite': cliente['limite_credito'],
                'excede_limite': saldo_devedor + valor_pedido - cliente['limite_credito']
            })
    return pd.DataFrame(blocked)


def generate_synthetic_data(n: int = 1000, seed: int = 42):
    """Compatibilidade: gera um DataFrame de transações sintéticas.

    Retorna a tabela `transactions` com a mesma estrutura usada em notebooks/tests.
    """
    np.random.seed(seed)
    clients = generate_clients(50)
    return generate_transactions(clients, n)

# Função principal para gerar todos os dados
def generate_all_data():
    clients = generate_clients(50)
    transactions = generate_transactions(clients, 500)
    financial = generate_financial_statements(clients)
    pendencies = generate_pending_items(transactions)
    blocked = generate_blocked_orders(clients, transactions)
    # Salvar em CSV
    clients.to_csv('data/clientes.csv', index=False)
    transactions.to_csv('data/transacoes.csv', index=False)
    financial.to_csv('data/demonstracoes.csv', index=False)
    pendencies.to_csv('data/pendencias.csv', index=False)
    blocked.to_csv('data/pedidos_bloqueados.csv', index=False)
    print("Dados gerados com sucesso!")

if __name__ == '__main__':
    generate_all_data()