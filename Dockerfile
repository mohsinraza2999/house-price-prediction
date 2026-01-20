# =========================
# Base image
# =========================
FROM python:3.9-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*
#    curl \

RUN pip install --upgrade pip setuptools wheel
# =========================
# Dependencies stage
# =========================
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# =========================
# Test stage
# =========================
FROM deps AS test

COPY . .
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]


# =========================
# Production stage
# =========================
FROM deps AS production

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.routes.router:app", "--host", "0.0.0.0", "--port", "8000"]