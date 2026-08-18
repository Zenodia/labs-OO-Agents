FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "service.py"]
