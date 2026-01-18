# Dark Subnet - FHE Bittensor Demo
# 
# This Dockerfile creates an environment with Concrete ML for
# running the full FHE (Fully Homomorphic Encryption) demo.
#
# Build:   docker build -t dark-subnet .
# Run:     docker run -it dark-subnet python demo.py
# Shell:   docker run -it dark-subnet bash

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    pydantic>=2.0.0 \
    rich>=13.0.0 \
    requests>=2.28.0 \
    concrete-ml>=1.5.0

# Copy project files
COPY . .

# Train the FHE model during build (optional - speeds up demo)
# Uncomment to pre-train: RUN python fhe_models/train_model.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Default command
CMD ["python", "demo.py"]
