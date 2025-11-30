# ────────────── Base image ──────────────
FROM python:3.10-slim

# Prevent interactive tzdata setup
ENV DEBIAN_FRONTEND=noninteractive

# ────────────── System dependencies ──────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ────────────── Copy project ──────────────
WORKDIR /app
COPY . /app

# ────────────── Install Python packages ──────────────
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ────────────── Expose & run ──────────────
EXPOSE 8000
CMD ["uvicorn", "face_service_v2:app", "--host", "0.0.0.0", "--port", "8000"]
