import pandas as pd
from src import pendencies


def test_identificar_pendencias_classifies_priority():
    df = pd.DataFrame({
        'id_pendencia': [1, 2],
        'tipo': ['Estorno', 'Abatimento'],
        'descricao': ['x', 'y'],
        'valor_original': [100, 200]
    })
    out = pendencies.identificar_pendencias(df.copy())
    assert out.loc[out['tipo'] == 'Estorno', 'prioridade'].iloc[0] == 'Alta'
    assert out.loc[out['tipo'] == 'Abatimento', 'prioridade'].iloc[0] == 'Média'
