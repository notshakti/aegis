FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire aegis package into /app/aegis/
COPY . ./aegis/

# The aegis package imports use `from aegis.models import ...`
# so /app must be on PYTHONPATH (aegis/ is the package dir under /app/)
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "aegis.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
