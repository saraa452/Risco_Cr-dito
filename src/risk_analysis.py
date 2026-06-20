
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

def calcular_indicadores(demonstracoes):
    """Calcula índices financeiros para cada cliente."""
    df = demonstracoes.copy()
    df['liquidez_corrente'] = df['ativo_circulante'] / df['passivo_circulante']
    df['endividamento'] = df['passivo_total'] / df['ativo_total']
    df['margem_liquida'] = df['lucro_liquido'] / df['receita_liquida']
    # Evitar divisões por zero ou infinito
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def calcular_score(clientes, transacoes, demonstracoes):
    """
    Calcula score de risco combinando dados financeiros e comportamentais.
    Regras simples:
    - Com base no histórico de pagamentos (média de atraso)
    - Índices financeiros
    - Tempo de relacionamento
    """
    # Agregar histórico de pagamentos por cliente
    hoje = pd.Timestamp.today()
    pag = transacoes.copy()
    pag['data_vencimento'] = pd.to_datetime(pag['data_vencimento'])
    pag['data_pagamento'] = pd.to_datetime(pag['data_pagamento'])
    
    # Dias de atraso (para pagamentos realizados)
    pag['dias_atraso_pagamento'] = (pag['data_pagamento'] - pag['data_vencimento']).dt.days
    pag.loc[pag['dias_atraso_pagamento'] < 0, 'dias_atraso_pagamento'] = 0  # pagamento antecipado = 0
    
    # Média de atraso por cliente (considerando apenas pagos)
    media_atraso = pag[pag['status'] == 'Paga'].groupby('id_cliente')['dias_atraso_pagamento'].mean().reset_index()
    media_atraso.columns = ['id_cliente', 'media_dias_atraso']
    
    # Calcular indicadores financeiros
    indicadores = calcular_indicadores(demonstracoes)
    
    # Unir tudo
    score_df = clientes[['id_cliente', 'data_cadastro', 'limite_credito', 'score_inicial']].merge(media_atraso, on='id_cliente', how='left')
    score_df = score_df.merge(indicadores[['id_cliente', 'liquidez_corrente', 'endividamento', 'margem_liquida']], on='id_cliente', how='left')
    
    # Preencher valores faltantes (clientes sem pagamentos ou sem demonstrações)
    score_df['media_dias_atraso'] = score_df['media_dias_atraso'].fillna(0)
    score_df['liquidez_corrente'] = score_df['liquidez_corrente'].fillna(1.0)
    score_df['endividamento'] = score_df['endividamento'].fillna(0.5)
    score_df['margem_liquida'] = score_df['margem_liquida'].fillna(0.1)
    
    # Calcular tempo de relacionamento em dias
    score_df['dias_relacionamento'] = (pd.Timestamp.today() - pd.to_datetime(score_df['data_cadastro'])).dt.days
    
    # Pontuação (quanto maior, menor risco)
    # Regras simples:
    score_df['pontos'] = 0
    # - Média de atraso: até 5 dias: +30, 5-15: +15, >15: 0
    score_df.loc[score_df['media_dias_atraso'] <= 5, 'pontos'] += 30
    score_df.loc[(score_df['media_dias_atraso'] > 5) & (score_df['media_dias_atraso'] <= 15), 'pontos'] += 15
    # - Liquidez corrente: >1.5: +20, >1: +10, <=1: 0
    score_df.loc[score_df['liquidez_corrente'] > 1.5, 'pontos'] += 20
    score_df.loc[(score_df['liquidez_corrente'] > 1) & (score_df['liquidez_corrente'] <= 1.5), 'pontos'] += 10
    # - Endividamento: <0.5: +20, 0.5-0.7: +10, >0.7: 0
    score_df.loc[score_df['endividamento'] < 0.5, 'pontos'] += 20
    score_df.loc[(score_df['endividamento'] >= 0.5) & (score_df['endividamento'] <= 0.7), 'pontos'] += 10
    # - Tempo de relacionamento: >2 anos: +10, >1 ano: +5
    score_df.loc[score_df['dias_relacionamento'] > 730, 'pontos'] += 10
    score_df.loc[(score_df['dias_relacionamento'] > 365) & (score_df['dias_relacionamento'] <= 730), 'pontos'] += 5
    
    # Classificar risco
    score_df['classificacao_risco'] = pd.cut(score_df['pontos'], bins=[0, 30, 50, 70, 100], labels=['Alto', 'Médio-Alto', 'Médio', 'Baixo'])
    
    return score_df[['id_cliente', 'media_dias_atraso', 'liquidez_corrente', 'endividamento', 'dias_relacionamento', 'pontos', 'classificacao_risco']]


# ---------------------------------------------------------------------------
# Funções de Machine Learning
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    'limite_credito', 'score_inicial', 'dias_relacionamento',
    'media_dias_atraso', 'total_faturas', 'pct_faturas_pagas',
    'valor_em_aberto', 'max_atraso_em_aberto',
    'liquidez_corrente', 'endividamento', 'margem_liquida',
    'segmento_enc', 'regiao_enc',
]


def preparar_features_ml(clientes, transacoes, demonstracoes):
    """
    Monta a matriz de features para modelagem ML de risco de crédito.

    Retorna DataFrame com id_cliente + colunas de features numéricas.
    """
    hoje = pd.Timestamp.today()

    pag = transacoes.copy()
    pag['data_vencimento'] = pd.to_datetime(pag['data_vencimento'])
    pag['data_pagamento'] = pd.to_datetime(pag['data_pagamento'])
    mask_paga = pag['status'] == 'Paga'

    # Dias de atraso para faturas já pagas
    pag.loc[mask_paga, 'dias_atraso_pagamento'] = (
        (pag.loc[mask_paga, 'data_pagamento'] - pag.loc[mask_paga, 'data_vencimento'])
        .dt.days.clip(lower=0)
    )

    # Faturas em aberto
    pag_aberto = pag[~mask_paga].copy()
    pag_aberto['dias_vencido'] = (hoje - pag_aberto['data_vencimento']).dt.days.clip(lower=0)

    # Agregações por cliente
    media_atraso = (
        pag[mask_paga].groupby('id_cliente')['dias_atraso_pagamento']
        .mean().rename('media_dias_atraso')
    )
    total_faturas = pag.groupby('id_cliente').size().rename('total_faturas')
    n_pagas = pag[mask_paga].groupby('id_cliente').size().rename('n_faturas_pagas')
    valor_aberto = pag_aberto.groupby('id_cliente')['valor'].sum().rename('valor_em_aberto')
    max_atraso_aberto = (
        pag_aberto.groupby('id_cliente')['dias_vencido']
        .max().rename('max_atraso_em_aberto')
    )

    # Montar base de features dos clientes
    feat = clientes[['id_cliente', 'data_cadastro', 'limite_credito', 'score_inicial', 'segmento', 'regiao']].copy()
    feat['dias_relacionamento'] = (hoje - pd.to_datetime(feat['data_cadastro'])).dt.days
    feat = feat.drop(columns=['data_cadastro'])

    feat = (
        feat
        .merge(media_atraso, on='id_cliente', how='left')
        .merge(total_faturas.to_frame(), on='id_cliente', how='left')
        .merge(n_pagas.to_frame(), on='id_cliente', how='left')
        .merge(valor_aberto.to_frame(), on='id_cliente', how='left')
        .merge(max_atraso_aberto.to_frame(), on='id_cliente', how='left')
    )

    # Indicadores financeiros
    ind = calcular_indicadores(demonstracoes)
    feat = feat.merge(
        ind[['id_cliente', 'liquidez_corrente', 'endividamento', 'margem_liquida']],
        on='id_cliente', how='left'
    )

    # Preencher NAs com defaults razoáveis
    feat['media_dias_atraso'] = feat['media_dias_atraso'].fillna(0)
    feat['total_faturas'] = feat['total_faturas'].fillna(0)
    feat['n_faturas_pagas'] = feat['n_faturas_pagas'].fillna(0)
    feat['valor_em_aberto'] = feat['valor_em_aberto'].fillna(0)
    feat['max_atraso_em_aberto'] = feat['max_atraso_em_aberto'].fillna(0)
    feat['liquidez_corrente'] = feat['liquidez_corrente'].fillna(1.0)
    feat['endividamento'] = feat['endividamento'].fillna(0.5)
    feat['margem_liquida'] = feat['margem_liquida'].fillna(0.1)
    feat['pct_faturas_pagas'] = (
        feat['n_faturas_pagas'] / feat['total_faturas'].replace(0, np.nan)
    ).fillna(1.0)
    feat = feat.drop(columns=['n_faturas_pagas'])

    # Codificação ordinal de categóricas
    feat['segmento_enc'] = pd.Categorical(feat['segmento']).codes
    feat['regiao_enc'] = pd.Categorical(feat['regiao']).codes
    feat = feat.drop(columns=['segmento', 'regiao'])

    return feat


def criar_target(transacoes, dias_corte=30):
    """
    Cria a variável target binária por cliente:
      1 = inadimplente (ao menos 1 fatura em aberto vencida há > dias_corte dias)
      0 = adimplente
    """
    hoje = pd.Timestamp.today()
    pag = transacoes.copy()
    pag['data_vencimento'] = pd.to_datetime(pag['data_vencimento'])

    aberto = pag[pag['status'] != 'Paga'].copy()
    aberto['dias_vencido'] = (hoje - aberto['data_vencimento']).dt.days

    inadimplente = (
        aberto[aberto['dias_vencido'] > dias_corte]
        .groupby('id_cliente')
        .size()
        .gt(0)
        .astype(int)
        .rename('inadimplente')
        .reset_index()
    )
    return inadimplente


def treinar_modelo_risco(clientes, transacoes, demonstracoes, dias_corte=30, cv_folds=5):
    """
    Treina dois modelos de classificação de risco (Regressão Logística e Random Forest).

    Parâmetros
    ----------
    dias_corte : int
        Quantidade de dias vencidos para classificar o cliente como inadimplente.
    cv_folds : int
        Número de folds para validação cruzada estratificada.

    Retorna
    -------
    dict com chaves:
        - 'logistic_regression' : Pipeline treinado
        - 'random_forest'       : Pipeline treinado
        - 'feature_names'       : lista de nomes de features
        - 'lr_auc_cv'           : AUC-ROC médio (LR) na validação cruzada
        - 'rf_auc_cv'           : AUC-ROC médio (RF) na validação cruzada
        - 'feature_importances' : Series com importâncias do RF
        - 'df_features'         : DataFrame completo features + target
    """
    feat = preparar_features_ml(clientes, transacoes, demonstracoes)
    target = criar_target(transacoes, dias_corte)

    df = feat.merge(target, on='id_cliente', how='left')
    df['inadimplente'] = df['inadimplente'].fillna(0).astype(int)

    feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df['inadimplente'].values

    n_splits = min(cv_folds, int(y.sum()), int((len(y) - y.sum())))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
    ])
    lr_scores = cross_val_score(lr_pipe, X, y, cv=cv, scoring='roc_auc')

    rf_pipe = Pipeline([
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
    ])
    rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring='roc_auc')

    # Treinar no dataset completo
    lr_pipe.fit(X, y)
    rf_pipe.fit(X, y)

    importancias = pd.Series(
        rf_pipe.named_steps['clf'].feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=False)

    return {
        'logistic_regression': lr_pipe,
        'random_forest': rf_pipe,
        'feature_names': feature_cols,
        'lr_auc_cv': float(lr_scores.mean()),
        'rf_auc_cv': float(rf_scores.mean()),
        'feature_importances': importancias,
        'df_features': df[['id_cliente'] + feature_cols + ['inadimplente']],
    }


def prever_risco_ml(clientes, transacoes, demonstracoes, modelo_treinado=None, dias_corte=30):
    """
    Gera previsões de risco de crédito por cliente usando ML.

    Se `modelo_treinado` (resultado de treinar_modelo_risco) não for fornecido,
    treina automaticamente um novo Random Forest.

    Retorna DataFrame com: id_cliente, prob_inadimplencia, inadimplente_predito,
    classificacao_ml.
    """
    if modelo_treinado is None:
        modelo_treinado = treinar_modelo_risco(clientes, transacoes, demonstracoes, dias_corte)

    modelo = modelo_treinado['random_forest']
    feature_cols = modelo_treinado['feature_names']

    feat = preparar_features_ml(clientes, transacoes, demonstracoes)
    X = feat[feature_cols].values

    proba = modelo.predict_proba(X)[:, 1]
    pred = modelo.predict(X)

    result = feat[['id_cliente']].copy()
    result['prob_inadimplencia'] = np.round(proba, 4)
    result['inadimplente_predito'] = pred
    result['classificacao_ml'] = pd.cut(
        proba,
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Baixo', 'Médio', 'Médio-Alto', 'Alto'],
        include_lowest=True,
    )
    return result.sort_values('prob_inadimplencia', ascending=False).reset_index(drop=True)