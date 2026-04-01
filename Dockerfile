FROM python:3.12-slim
LABEL org.opencontainers.image.source=https://github.com/Lukombo-JDS/face-detect-laforge

# 1. Dépendances système (Regroupées pour limiter les layers)
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    make \
    && rm -rf /var/lib/apt/lists/*

# 2. Installation de uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
ENV PYTHONPATH=/app

# 3. Cache des dépendances : on ne copie QUE ces deux fichiers d'abord
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 4. Copie du reste du projet (seulement maintenant)
COPY . .

# 5. Initialisation
RUN uv run make init env download

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "view/display.py", "--server.port=8501", "--server.address=0.0.0.0"]
