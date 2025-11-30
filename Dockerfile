# ────────────── Base Image ──────────────
FROM python:3.10-slim

# Avoid tzdata prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ────────────── Install system dependencies ──────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ────────────── Working directory ──────────────
WORKDIR /app
COPY . /app

# ────────────── Python deps ──────────────
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    deepface==0.0.79 \
    tensorflow-cpu==2.12.0 \
    opencv-python-headless \
    pillow \
    requests \
    numpy

# ────────────── Expose & Run ──────────────
EXPOSE 8000
CMD ["uvicorn", "face_service_v2:app", "--host", "0.0.0.0", "--port", "8000"]
