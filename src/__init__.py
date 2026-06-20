"""
Pacote `src` do projeto de análise de crédito e cobrança.
Exporta funções utilitárias para uso direto em notebooks e scripts.
"""
from .data_generator import generate_synthetic_data
from .aging import aging_report, clientes_inadimplentes
from .pendencies import identificar_pendencias
from .risk_analysis import calcular_indicadores, calcular_score
from .blocked_orders import analisar_pedidos_bloqueados, gerar_relatorio_pedidos
from .kpi_dashboard import kpi_report

__all__ = [
    "generate_synthetic_data",
    "aging_report",
    "clientes_inadimplentes",
    "identificar_pendencias",
    "calcular_indicadores",
    "calcular_score",
    "analisar_pedidos_bloqueados",
    "gerar_relatorio_pedidos",
    "kpi_report",
]
