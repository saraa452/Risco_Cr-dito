import pytest
import pandas as pd
from src import aging, utils


def test_aging_summary_totals_match_transacoes():
    _, transacoes, _, _, _ = utils.load_data()
    aging_summary, aging_cliente, df = aging.aging_report(transacoes)
    assert pytest.approx(aging_summary['valor_total'].sum(), rel=1e-6) == transacoes['valor'].sum()


def test_clientes_inadimplentes_basic():
    _, transacoes, _, _, _ = utils.load_data()
    inadimplentes = aging.clientes_inadimplentes(transacoes, dias_minimo=1)
    assert isinstance(inadimplentes, pd.DataFrame)
    # espera colunas mínimas
    assert 'id_cliente' in inadimplentes.columns
    assert 'valor' in inadimplentes.columns
