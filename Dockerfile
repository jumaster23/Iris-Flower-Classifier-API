FROM python:3.13-slim

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copiar archivos de dependencias primero (cache de Docker)
COPY pyproject.toml uv.lock ./

# Instalar dependencias sin dev
RUN uv sync --frozen --no-dev

# Copiar el resto del código
COPY . .

# Entrenar el modelo si no existen los archivos
RUN uv run train.py

EXPOSE 8001

CMD ["uv", "run", "main.py"]
```

Y crea un `.dockerignore`:
```
.venv/
__pycache__/
*.pyc
.env
.pytest_cache/
tests/