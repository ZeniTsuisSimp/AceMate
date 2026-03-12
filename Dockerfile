FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV (rapidocr) and general use
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST to prevent sentence-transformers
# from pulling the 2GB CUDA build
RUN pip install --no-cache-dir \
        torch==2.4.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies (torch already present, pip will skip it)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
