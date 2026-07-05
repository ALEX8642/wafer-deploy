# wafer-deploy service image — CPU base for the quickstart.
#
# Phase 4 builds the arm64 variant for the GB10 co-tenant deploy; this default
# is the CPU-reproducible image the docker-compose quickstart uses. The 45 MB
# checkpoint is NOT baked in — it is bind-mounted from the sibling wafer-mixed
# checkout at run time (see docker-compose.yml), matching the repo policy that
# no *.pt is vendored here.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

# CPU-only torch keeps the image lean (the default wheel bundles multi-GB CUDA).
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.7.1,<2.9" "torchvision>=0.22.1,<0.24" \
 && pip install \
        "numpy>=2.0" "scipy>=1.11" "scikit-learn>=1.5" "tqdm>=4.66" "pyyaml>=6.0" \
        "matplotlib>=3.8" \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "prometheus-client>=0.20"

# Application code + committed reference snapshot.
COPY pyproject.toml ./
COPY src/ ./src/
COPY serve/ ./serve/
COPY configs/ ./configs/
COPY reference/ ./reference/

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if 'ok' in urllib.request.urlopen('http://localhost:8000/healthz').read().decode() else 1)"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
