# analise_credito_cobranca

[![CI](https://img.shields.io/github/actions/workflow/status/USERNAME/REPO/ci.yml?branch=main&label=ci&logo=github)](https://github.com/USERNAME/REPO/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)

Projeto exemplo de *Credit & Collections* com: geração de dados sintéticos, cálculo de aging, identificação de pendências, heurística de score de risco e um dashboard em Dash. Ideal para demonstrar fluxo de dados e técnicas de análise em um portfólio.

---

## O que tem aqui

- `data/` — dados de entrada/saída (exemplos gerados)
- `notebooks/` — notebook de exploração `01-exploracao.ipynb`
- `src/` — código fonte (módulos reaproveitáveis)
- `reports/` — relatórios gerados (Excel, etc.)

---

## Quickstart

1) Crie e ative o ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # para dev (tests/format)
```

2) Gerar dados de exemplo (opcional):

```bash
 # ou use a função generate_synthetic_data
```

3) Rodar o dashboard localmente:

```bash
python -m src.kpi_dashboard
# abrir http://127.0.0.1:8050
```

4) Executar testes:

```bash
pytest -q
```

5) Build Docker (opcional):

```bash
docker build -t analise_credito_cobranca:latest .
docker run -p 8050:8050 analise_credito_cobranca:latest
```

---

## Desenvolvimento & qualidade

- Formatação automática: `black` + `isort` via `pre-commit`
- Testes: `pytest` (tests/)
- CI: GitHub Actions (`.github/workflows/ci.yml`)

Para formatar e testar localmente:

```bash
pre-commit run --all-files
pytest -q
```

---

## Contribuir

Abra uma issue ou envie um PR com uma branch clara (`feat/`, `fix/`, `chore/`). Incluir testes para novas funcionalidades é obrigatório.

---

## License

Este projeto está licenciado sob a Apache License 2.0 — veja o arquivo `LICENSE` para detalhes.

---

(Atualize os badges no topo trocando `USERNAME/REPO` pelo seu usuário e repositório no GitHub.)
