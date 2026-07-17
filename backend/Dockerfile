# One image, two roles (API vs worker) -- docker-compose overrides `command` per service so
# both share the exact same dependency set and never drift out of sync with each other.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies layer cached separately from source so code changes don't force a full
# torch/transformers/whisperx reinstall.
COPY requirements.txt .
# CPU-only torch build: the default PyPI wheel bundles the full CUDA runtime (nvidia-*-cu12
# packages, several GB) which is dead weight on a GPU-less instance and exhausted the disk
# during an earlier attempt. --index-url (not a separate install step) makes the CPU
# wheelhouse primary for THIS WHOLE install -- installing torch as a one-off first didn't
# stick, because whisperx/pyannote pin a specific torch version and pip re-resolved it from
# the default (CUDA) index when processing the rest of requirements.txt. --extra-index-url
# lets everything else still come from regular PyPI.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Default command runs the API; the worker service in docker-compose.yml overrides this.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
