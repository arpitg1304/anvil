.PHONY: install sync test lint typecheck fmt check all

install:
	uv sync --all-extras --group dev

sync:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

check: lint typecheck test

all: check
