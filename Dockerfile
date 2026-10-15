# CPU-only, key-free image: `docker build -t vsm . && docker run --rm vsm make repro`
FROM python:3.11-slim

# No API key, no GPU, no network needed at runtime. Agg backend for matplotlib.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    VLM_BACKEND=mock \
    MAX_FRAMES=120 \
    TORCH_HOME=/app/out/cache/torch \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# opencv-python needs libGL / libgthread even though we never open a window, and the
# `make` binary drives the repro target.
RUN apt-get update \
    && apt-get install -y --no-install-recommends make libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the pinned CPU torch/torchvision wheels first: the PyPI defaults are the
# CUDA builds, which are ~10x larger and useless here.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 torchvision==0.29.1

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Pre-download the resnet18 weights used by the dedup stage so the container runs
# fully offline.
RUN python -c "from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"

COPY . .

# Smoke check that the package imports and the CLI is wired up.
RUN python -c "import vsm; print('vsm', vsm.__version__)" \
    && python scripts/make_dataset.py --n 4 --data-dir /tmp/vsm-smoke >/dev/null \
    && rm -rf /tmp/vsm-smoke \
    && make lint

CMD ["make", "repro"]
