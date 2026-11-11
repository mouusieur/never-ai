FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEVER_AI_BOOTSTRAP_GITHUB_BRAIN=1 \
    NEVER_AI_CLOUD_AUTO_OTA=1 \
    NEVER_AI_DATA_DIR=/home/neverai/app \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user uid=1000 (required by Hugging Face Spaces Docker & secure on Render/Koyeb/Railway)
RUN useradd -m -u 1000 neverai
WORKDIR /home/neverai/app

COPY --chown=neverai:neverai pyproject.toml README.md DIRECTIVES.md ./
COPY --chown=neverai:neverai src ./src
COPY --chown=neverai:neverai assets ./assets
COPY --chown=neverai:neverai mobile_pwa ./mobile_pwa
COPY --chown=neverai:neverai scripts ./scripts

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -e . && \
    python scripts/generate_brand_assets.py && \
    mkdir -p checkpoints memory_store logs && \
    chown -R neverai:neverai /home/neverai/app

USER neverai
EXPOSE 7860

CMD ["python", "scripts/cloud_entrypoint.py"]
