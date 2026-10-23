FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY auspicious ./auspicious
COPY data ./data
COPY examples ./examples
RUN pip install --no-cache-dir '.[gcp]'
ENTRYPOINT ["python", "-m", "auspicious.cloud"]
