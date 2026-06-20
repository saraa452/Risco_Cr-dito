import pandas as pd
import numpy as np
from src import risk_analysis


# ---------------------------------------------------------------------------
# Fixtures reutilizáveis
# ---------------------------------------------------------------------------

def _make_clientes(n=10):
    return pd.DataFrame({
        'id_cliente': list(range(1, n + 1)),
        'nome': [f'Cliente {i}' for i in range(1, n + 1)],
        'segmento': ['Atacado' if i % 2 == 0 else 'Varejo' for i in range(1, n + 1)],
        'regiao': ['Sudeste', 'Sul', 'Norte', 'Nordeste', 'Centro-Oeste'] * 2,
        'data_cadastro': pd.date_range('2020-01-01', periods=n, freq='180D'),
        'limite_credito': [10000] * n,
        'score_inicial': [50] * n,
    })


def _make_transacoes(clientes_ids):
    rows = []
    hoje = pd.Timestamp.today()
    for i, cid in enumerate(clientes_ids):
        # Fatura paga no prazo
        rows.append({
            'id_transacao': i * 3 + 1,
            'id_cliente': cid,
            'data_vencimento': hoje - pd.Timedelta(days=90),
            'data_pagamento': hoje - pd.Timedelta(days=88),
            'valor': 1000.0,
            'status': 'Paga',
        })
        # Fatura paga com atraso
        rows.append({
            'id_transacao': i * 3 + 2,
            'id_cliente': cid,
            'data_vencimento': hoje - pd.Timedelta(days=60),
            'data_pagamento': hoje - pd.Timedelta(days=50),
            'valor': 500.0,
            'status': 'Paga',
        })
        # Fatura em aberto vencida há >30 dias para metade dos clientes
        status = 'Em aberto' if i % 2 == 0 else 'Paga'
        rows.append({
            'id_transacao': i * 3 + 3,
            'id_cliente': cid,
            'data_vencimento': hoje - pd.Timedelta(days=45),
            'data_pagamento': (hoje - pd.Timedelta(days=44)) if status == 'Paga' else pd.NaT,
            'valor': 2000.0,
            'status': status,
        })
    return pd.DataFrame(rows)


def _make_demonstracoes(clientes_ids):
    return pd.DataFrame({
        'id_cliente': clientes_ids,
        'ano': [2025] * len(clientes_ids),
        'ativo_circulante': [300_000.0] * len(clientes_ids),
        'passivo_circulante': [150_000.0] * len(clientes_ids),
        'ativo_total': [1_000_000.0] * len(clientes_ids),
        'passivo_total': [400_000.0] * len(clientes_ids),
        'receita_liquida': [2_000_000.0] * len(clientes_ids),
        'lucro_liquido': [200_000.0] * len(clientes_ids),
    })


# ---------------------------------------------------------------------------
# Testes existentes
# ---------------------------------------------------------------------------

def test_calcular_indicadores_outputs_expected_columns():
    demo = pd.DataFrame({
        'id_cliente': [1],
        'ativo_circulante': [200.0],
        'passivo_circulante': [100.0],
        'passivo_total': [100.0],
        'ativo_total': [200.0],
        'lucro_liquido': [10.0],
        'receita_liquida': [100.0]
    })
    out = risk_analysis.calcular_indicadores(demo)
    assert 'liquidez_corrente' in out.columns
    assert 'endividamento' in out.columns
    assert 'margem_liquida' in out.columns


def test_calcular_score_basic_flow():
    clientes = pd.DataFrame({'id_cliente': [1], 'data_cadastro': ['2020-01-01'], 'limite_credito': [1000], 'score_inicial': [50]})
    transacoes = pd.DataFrame({
        'id_cliente': [1],
        'data_vencimento': ['2024-01-01'],
        'data_pagamento': ['2024-01-02'],
        'valor': [100.0],
        'status': ['Paga']
    })
    demonstracoes = pd.DataFrame({
        'id_cliente': [1],
        'ativo_circulante': [200.0],
        'passivo_circulante': [100.0],
        'passivo_total': [50.0],
        'ativo_total': [200.0],
        'lucro_liquido': [10.0],
        'receita_liquida': [100.0]
    })
    df = risk_analysis.calcular_score(clientes, transacoes, demonstracoes)
    assert 'pontos' in df.columns
    assert 'classificacao_risco' in df.columns
    assert df['pontos'].iloc[0] >= 0


# ---------------------------------------------------------------------------
# Testes das novas funções ML
# ---------------------------------------------------------------------------

def test_preparar_features_ml_shape_e_colunas():
    clientes = _make_clientes(10)
    transacoes = _make_transacoes(list(range(1, 11)))
    demonstracoes = _make_demonstracoes(list(range(1, 11)))

    feat = risk_analysis.preparar_features_ml(clientes, transacoes, demonstracoes)

    assert len(feat) == 10
    for col in ['id_cliente', 'limite_credito', 'score_inicial', 'dias_relacionamento',
                'media_dias_atraso', 'total_faturas', 'pct_faturas_pagas',
                'valor_em_aberto', 'max_atraso_em_aberto',
                'liquidez_corrente', 'endividamento', 'margem_liquida',
                'segmento_enc', 'regiao_enc']:
        assert col in feat.columns, f"Coluna ausente: {col}"


def test_preparar_features_ml_sem_nulos():
    clientes = _make_clientes(10)
    transacoes = _make_transacoes(list(range(1, 11)))
    demonstracoes = _make_demonstracoes(list(range(1, 11)))

    feat = risk_analysis.preparar_features_ml(clientes, transacoes, demonstracoes)
    assert feat.isnull().sum().sum() == 0, "Existem valores nulos nas features"


def test_criar_target_classifica_inadimplentes():
    hoje = pd.Timestamp.today()
    transacoes = pd.DataFrame({
        'id_transacao': [1, 2, 3],
        'id_cliente': [1, 2, 3],
        # cliente 1: vencida há 40 dias (inadimplente)
        # cliente 2: vencida há 10 dias (não inadimplente por padrão dias_corte=30)
        # cliente 3: paga (adimplente)
        'data_vencimento': [
            hoje - pd.Timedelta(days=40),
            hoje - pd.Timedelta(days=10),
            hoje - pd.Timedelta(days=60),
        ],
        'data_pagamento': [pd.NaT, pd.NaT, hoje - pd.Timedelta(days=58)],
        'valor': [1000.0, 500.0, 2000.0],
        'status': ['Em aberto', 'Em aberto', 'Paga'],
    })
    target = risk_analysis.criar_target(transacoes, dias_corte=30)

    inadimplentes = dict(zip(target['id_cliente'], target['inadimplente']))
    assert inadimplentes.get(1) == 1, "Cliente 1 deve ser inadimplente"
    assert 2 not in inadimplentes, "Cliente 2 não deve aparecer (atraso < corte)"


def test_treinar_modelo_risco_retorna_estrutura():
    clientes = _make_clientes(10)
    transacoes = _make_transacoes(list(range(1, 11)))
    demonstracoes = _make_demonstracoes(list(range(1, 11)))

    resultado = risk_analysis.treinar_modelo_risco(clientes, transacoes, demonstracoes)

    assert 'logistic_regression' in resultado
    assert 'random_forest' in resultado
    assert 'lr_auc_cv' in resultado
    assert 'rf_auc_cv' in resultado
    assert 'feature_importances' in resultado
    assert 'df_features' in resultado
    assert 0.0 <= resultado['lr_auc_cv'] <= 1.0
    assert 0.0 <= resultado['rf_auc_cv'] <= 1.0


def test_treinar_modelo_risco_feature_importances_soma_1():
    clientes = _make_clientes(10)
    transacoes = _make_transacoes(list(range(1, 11)))
    demonstracoes = _make_demonstracoes(list(range(1, 11)))

    resultado = risk_analysis.treinar_modelo_risco(clientes, transacoes, demonstracoes)
    total = resultado['feature_importances'].sum()
    assert abs(total - 1.0) < 1e-6


def test_prever_risco_ml_retorna_todos_clientes():
    clientes = _make_clientes(10)
    transacoes = _make_transacoes(list(range(1, 11)))
    demonstracoes = _make_demonstracoes(list(range(1, 11)))

    resultado = risk_analysis.prever_risco_ml(clientes, transacoes, demonstracoes)

    assert len(resultado) == 10
    assert 'prob_inadimplencia' in resultado.columns
    assert 'inadimplente_predito' in resultado.columns
    assert 'classificacao_ml' in resultado.columns


def test_prever_risco_ml_probabilidades_validas():
    clientes = _make_clientes(10)
    transacoes = _make_transacoes(list(range(1, 11)))
    demonstracoes = _make_demonstracoes(list(range(1, 11)))

    resultado = risk_analysis.prever_risco_ml(clientes, transacoes, demonstracoes)

    assert resultado['prob_inadimplencia'].between(0, 1).all()
    assert resultado['inadimplente_predito'].isin([0, 1]).all()


def test_prever_risco_ml_com_modelo_preexistente():
    clientes = _make_clientes(10)
    transacoes = _make_transacoes(list(range(1, 11)))
    demonstracoes = _make_demonstracoes(list(range(1, 11)))

    modelo_treinado = risk_analysis.treinar_modelo_risco(clientes, transacoes, demonstracoes)
    resultado = risk_analysis.prever_risco_ml(
        clientes, transacoes, demonstracoes,
        modelo_treinado=modelo_treinado,
    )
    assert len(resultado) == 10

