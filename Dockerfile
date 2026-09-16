# Root Dockerfile for Hugging Face Spaces (Docker SDK). The docker-compose based
# local/self-hosted setup uses docker/Dockerfile instead -- this one specifically
# matches Spaces' requirements: run as uid 1000, WORKDIR under that user's home,
# listen on the port Spaces expects (7860 by default, see README.md's app_port).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    DATA_DIR=/home/user/app/data \
    INDEX_DIR=/home/user/app/.indexes

WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
COPY --chown=user data ./data
COPY --chown=user entrypoint.sh ./

RUN pip install --no-cache-dir . && chmod +x entrypoint.sh

EXPOSE 7860
CMD ["./entrypoint.sh"]
