# syntax=docker/dockerfile:1.7-labs

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install first for better caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy source
COPY app ./app
COPY README.md ./

# Default env (can be overridden at runtime)
ENV OPENAI_MODEL=gpt-4o-mini

EXPOSE 8501

# Use streamlit to run the app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
