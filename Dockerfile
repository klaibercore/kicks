# The kicks REST API with the DisCoder and BigVGAN vocoders. The website in web/ is a static
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

# Vocoder weights download on startup into the persistent models volume.
# This also lets KICKS_VOCODER select a backend without baking another model
# into every image.

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "kicks.api:app", "--host", "0.0.0.0", "--port", "8080"]
