import pandas as pd
from src import utils


def test_load_data_returns_dataframes():
    clientes, transacoes, demonstracoes, pendencias, pedidos = utils.load_data()
    assert not clientes.empty
    assert 'id_cliente' in clientes.columns
    assert 'valor' in transacoes.columns
    assert isinstance(demonstracoes, pd.DataFrame)
    assert isinstance(pendencias, pd.DataFrame)
    assert isinstance(pedidos, pd.DataFrame)
