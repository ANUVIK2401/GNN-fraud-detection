FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY fraudgnn ./fraudgnn
COPY artifacts/bundle ./bundle
ENV BUNDLE_DIR=/app/bundle PORT=8080

# One worker on purpose: neighbor state lives in process memory.
CMD exec uvicorn fraudgnn.serve.app:app --host 0.0.0.0 --port ${PORT} --workers 1
