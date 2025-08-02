FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt update -y && apt install awscli -y
WORKDIR /app

# Only install non-PyTorch packages
COPY requirements-light.txt .
RUN pip install --no-cache-dir -r requirements-light.txt

COPY . .
CMD ["python3", "app.py"]