# Stage 1: Base image with heavy dependencies
FROM python:3.8-slim as base

RUN apt update -y && apt install awscli -y

# Install heavy ML packages first (better caching)
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 pytorch-lightning==2.4.0

# Stage 2: Application
FROM base as app
WORKDIR /app

# Install remaining dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code (this changes frequently)
COPY . /app

CMD ["python3", "app.py"]