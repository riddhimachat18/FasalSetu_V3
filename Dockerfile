FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF, Pillow, and audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Pre-create required dirs
RUN mkdir -p logs data/schemes data/chroma_db models

# Ingest seed scheme data at build time (no PDFs needed)
RUN python scripts/ingest_schemes.py || true

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
