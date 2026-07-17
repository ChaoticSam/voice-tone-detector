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
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Default command runs the API; the worker service in docker-compose.yml overrides this.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
