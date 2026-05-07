# syntax=docker/dockerfile:1.7

# CUDA-enabled PyTorch base image. faster-qwen3-tts requires torch >= 2.5.1 and CUDA.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    DATA_DIR=/data

# libsndfile for soundfile, ffmpeg for optional mp3/opus/aac encoding,
# git for any pip-from-git deps that faster-qwen3-tts may pull transitively.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Application code.
COPY app /app/app
COPY pyproject.toml /app/pyproject.toml

# Persistent data (mount as a volume in production).
RUN mkdir -p /data /root/.cache/huggingface
VOLUME ["/data", "/root/.cache/huggingface"]

EXPOSE 8080
EXPOSE 10200

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
