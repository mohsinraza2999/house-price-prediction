# =========================
# Base image
# =========================
FROM python:3.9-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# =========================
# Builder (deps compilation)
# =========================
FROM base AS builder


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


#COPY requirements.txt .



# =========================
# Test image
# =========================
FROM base AS test

COPY . /app
# Install project in editable mode + PyTorch CPU
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -e .

RUN pip install --no-cache-dir -r requirements-dev.txt

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY tests /app/tests

# Tests only — no server startup
#CMD ["pytest", "-q"]
CMD ["sh", "-c", "\
    uvicorn src.routes.router:app --host 0.0.0.0 --port 8000 & \
    sleep 5 && \
    python -m unittest discover -s tests -p test_*.py\
"]

# =========================
# Production image
# =========================
FROM base AS production

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app

# Copy only what is needed
COPY pyproject.toml /app/
COPY src /app/src
COPY configs /app/configs

COPY --from=builder /wheels /wheels
RUN pip wheel --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu . -w /wheels \
    && pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

USER app

EXPOSE 8000
CMD ["uvicorn", "src.routes.router:app", "--host", "0.0.0.0", "--port", "8000"]
# Add a healthcheck to verify FastAPI is responding
#HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
#  CMD curl --fail http://127.0.0.1:8000/ || exit 1