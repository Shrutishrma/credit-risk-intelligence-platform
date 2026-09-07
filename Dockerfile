FROM python:3.12-slim

WORKDIR /app

# Basic system packages needed by Python ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better Docker caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY src ./src
COPY models ./models

# Data is mounted through docker-compose
RUN mkdir -p /app/data/processed

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]