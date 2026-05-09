FROM python:3.11-slim

# System deps for PyBullet + OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects PORT env var
ENV PORT=5000
ENV PYTHONPATH=/app

EXPOSE $PORT

CMD ["python", "demo/backend/app.py"]
