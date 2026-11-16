FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -c requirements.txt \
        fastapi joblib numpy pandas scikit-learn uvicorn

FROM python:3.12-slim AS runtime

LABEL maintainer="MLOps Student"
LABEL description="FastAPI churn prediction API"

WORKDIR /app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv
COPY app.py model_pipeline.py ./
COPY artifacts/classifier.joblib ./artifacts/classifier.joblib

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
