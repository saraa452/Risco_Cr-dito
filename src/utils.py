import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

def load_data(data_dir: Optional[str] = None):
    """Carrega todos os CSVs da pasta `data/`.

    - Por padrão procura `../data` relativo à raiz do projeto (um nível acima de `src/`).
    - Aceita `data_dir` explícito (string ou Path).
    - Gera mensagens claras se arquivos estiverem ausentes.
    """
    base = Path(__file__).resolve().parents[1]
    data_dir = Path(data_dir) if data_dir else base / "data"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    def _read(fname, **kwargs):
        p = data_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")
        return pd.read_csv(p, **kwargs)

    clientes = _read("clientes.csv", parse_dates=["data_cadastro"]) 
    transacoes = _read("transacoes.csv", parse_dates=["data_vencimento", "data_pagamento"])
    demonstracoes = _read("demonstracoes.csv")
    pendencias = _read("pendencias.csv")
    pedidos = _read("pedidos_bloqueados.csv")
    return clientes, transacoes, demonstracoes, pendencias, pedidos

def format_currency(value):
    return f"R$ {value:,.2f}"