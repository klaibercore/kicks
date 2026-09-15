# The kicks REST API with the BigVGAN vocoder. The website in web/ is a static
# site (GitHub Pages) that talks to this container. GPU is recommended; falls
# back to CPU.

FROM python:3.12-slim

WORKDIR /app

# System deps for audio I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY kicks/ ./kicks/
RUN pip install --no-cache-dir .

# Pre-download BigVGAN so the first request does not pay for it.
RUN python -c "from kicks.audio.vocoder import patch_bigvgan_from_pretrained; \
    patch_bigvgan_from_pretrained(); \
    import bigvgan; \
    bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_256x', use_cuda_kernel=False)" \
    || echo "BigVGAN pre-download skipped (will download on first run)"

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "kicks.api:app", "--host", "0.0.0.0", "--port", "8080"]
