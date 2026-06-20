.PHONY: test lint docker-build run

test:
	pip install -r requirements-dev.txt
	pytest -q

lint:
	pre-commit run --all-files

docker-build:
	docker build -t analise_credito_cobranca:latest .

run:
	python -m src.kpi_dashboard
