# Kicks Backend Dockerfile
# FastAPI + BigVGAN vocoder with GPU support (falls back to CPU)

FROM python:3.12-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir \
    torch \
    torchaudio \
    soundfile \
    numpy \
    scikit-learn \
    fastapi \
    uvicorn \
    typer \
    rich \
    matplotlib \
    pyloudnorm \
    bigvgan \
    huggingface_hub

COPY kicks/ ./kicks/

# Pre-download BigVGAN model on build (optional, speeds first startup)
RUN python -c "from kicks.vocoder import _patch_bigvgan_from_pretrained; \
    _patch_bigvgan_from_pretrained(); \
    import bigvgan; \
    bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_256x', use_cuda_kernel=False)" \
    || echo "BigVGAN pre-download skipped (will download on first run)"

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "kicks.server:app", "--host", "0.0.0.0", "--port", "8080"]
