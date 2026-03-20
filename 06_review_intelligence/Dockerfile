# ─── Base Dockerfile (copy into each project folder as Dockerfile) ─────────────
# Usage: docker build -t ai_projects/<name> .
#        docker run -p <PORT>:<PORT> ai_projects/<name>

FROM python:3.11-slim

# System deps (ffmpeg needed for project 05)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# HuggingFace cache dir
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

# Expose the service port (override per project)
EXPOSE 8005

# Start the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
