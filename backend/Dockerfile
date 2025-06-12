# /backend/ Dockerfile
# builder stage
FROM python:3.9-slim-bullseye AS builder

# set the working directory
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# production stage
FROM python:3.9-slim-bullseye

WORKDIR /app

RUN addgroup --system appgroup && adduser --system --no-create-home --ingroup appgroup appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

RUN mkdir -p /app/data /app/results /app/logs

RUN chown -R appuser:appgroup /app

USER appuser

ENV PATH="/opt/venv/bin:$PATH"
ENV MPLCONFIGDIR=/app/.cache

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
