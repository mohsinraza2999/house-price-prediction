# =========================
# Base image
# =========================
FROM python:3.11-slim AS base

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
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

# =========================
# Test image
# =========================
FROM base AS test

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY . .

# Tests only — no server startup
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]

# =========================
# Production image
# =========================
FROM base AS production

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

COPY . .

USER app

EXPOSE 8000
CMD ["uvicorn", "src.routes.router:app", "--host", "0.0.0.0", "--port", "8000"]
# Add a healthcheck to verify FastAPI is responding
#HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
#  CMD curl --fail http://127.0.0.1:8000/ || exit 1