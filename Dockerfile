FROM python:3.12-slim

WORKDIR /app

# Install system deps (if needed for NLP libs like numpy, torch, etc.)
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

CMD ["python", "main.py"]