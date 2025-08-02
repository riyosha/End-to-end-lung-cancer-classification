FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set non-interactive mode and US timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

RUN apt update -y && apt install awscli -y
WORKDIR /app

# Copy package files (needed for -e . in requirements.txt)
COPY setup.py .
COPY src/ ./src/
COPY README.md .

# Only install non-PyTorch packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python3", "app.py"]