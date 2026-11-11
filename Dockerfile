FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt requirements-server.txt ./
RUN python -m pip install --no-cache-dir -r requirements-server.txt && useradd --uid 10001 --create-home appuser
COPY healthdl/__init__.py healthdl/data.py healthdl/inference.py healthdl/runtime.py ./healthdl/
COPY main.py ./
COPY templates/ ./templates/
COPY public/ ./public/
COPY data/conditions.json data/provenance.json ./data/
COPY artifacts/metadata.json artifacts/inference.npz artifacts/training_patterns.npz artifacts/report.json ./artifacts/
COPY deploy/gunicorn.conf.py ./deploy/
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2)"
CMD ["gunicorn", "--config", "deploy/gunicorn.conf.py", "main:app"]
