# Free Cloud Deployment Dockerfile (Render.com & Hugging Face Spaces compatible)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces defaults to port 7860; Render injects $PORT automatically
ENV PORT=7860
ENV HOST=0.0.0.0
ENV DEMO_CAPITAL=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "demo_bot_server.py"]
