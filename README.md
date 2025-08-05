# 🫁 End-to-End Lung Cancer Classification

A production-ready MLOps pipeline for Non-Small Cell Lung Cancer (NSCLC) classification using deep learning. This project implements a complete end-to-end workflow from data ingestion to model deployment with automated CI/CD, model versioning, and cloud-based inference.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.12+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)
![AWS](https://img.shields.io/badge/AWS-ECR%20%7C%20S3-orange.svg)

## 🎯 Project Overview

This project uses transfer learning with pre-trained CNN models (ResNet50 and VGG16) to classify chest CT scan images for lung cancer subtype classification. The system can distinguish between normal tissue and three major types of Non-Small Cell Lung Cancer (NSCLC): adenocarcinoma, large cell carcinoma, and squamous cell carcinoma. The implementation follows MLOps best practices with automated pipelines, experiment tracking, and containerized deployment.

**Live Demo:** [http://34.221.223.5:8080/](http://34.221.223.5:8080/)

## Key Features

- **🤖 Deep Learning Model**: Transfer learning with ResNet/VGG architectures
- **📊 MLflow Integration**: Experiment tracking and model versioning with MLFlow
- **🐳 Docker Containerization**: Scalable deployment with Docker
- **☁️ AWS Cloud Integration**: ECR for container registry, S3 for model storage
- **CI/CD Pipeline**: Automated building, testing, and deployment
- **DVC Pipeline**: Data versioning and reproducible ML pipelines
- **Web Interface**: Flask-based UI for real-time predictions
- **Comprehensive Logging**: Structured logging throughout the pipeline

## Technology Stack

| Component | Technology |
|-----------|------------|
| **ML Framework** | PyTorch, PyTorch Lightning |
| **Web Framework** | Flask |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **Containerization** | Docker |
| **Cloud Platform** | AWS (ECR, S3, EC2) |
| **CI/CD** | GitHub Actions |
| **Configuration** | YAML, Python dataclasses |

## 📁 Project Structure

```
├── .github/workflows/     # CI/CD pipelines
├── config/               # Configuration files
│   └── config.yaml      # Main configuration
├── src/cvClassifier/     # Main package
│   ├── components/       # Core ML components
│   ├── pipeline/         # Training & prediction pipelines
│   ├── utils/           # Utility functions
│   └── __init__.py      # Package initialization & logging
├── templates/           # Web UI templates
├── research/           # Jupyter notebooks for experimentation
├── artifacts/          # Generated artifacts (models, data)
├── model/             # Trained model storage
├── logs/              # Application logs
├── app.py             # Flask web application
├── main.py            # Training pipeline entry point
├── dvc.yaml           # DVC pipeline definition
├── params.yaml        # Model hyperparameters
├── Dockerfile         # Container configuration
└── requirements.txt   # Python dependencies
```

## Quick Start for locally deploying the project

### Prerequisites
- Python 3.8+
- Docker (for containerized deployment)
- AWS CLI (for cloud deployment)
- Git

### 1. Clone Repository

```bash
git clone https://github.com/riyosha/End-to-end-lung-cancer-classification.git
cd End-to-end-chest-cancer-classification
```

### 2. Setup Environment
```bash
# Create virtual environment
conda create -n chest-cancer python=3.8 -y
conda activate chest-cancer

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables and Github Secrets

Create a `.env` file with your credentials:

```bash
# MLflow Tracking
MLFLOW_TRACKING_URI=your_mlflow_uri
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
```

Set these github secrets with your AWS credentials:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-west-2
AWS_ECR_LOGIN_URI=you_ECR_login
```

### 4. Run Training Pipeline

```bash
# Run the complete training pipeline
dvc repro
```

### 5. Launch Web Application

```bash
# Launch Flask app
python app.py
```
Visit `http://localhost:8080` to access the locally deployed web interface.

## AWS Cloud Deployment
The project includes automated Docker image building and pushing to AWS ECR via GitHub Actions.

### Prerequisites for AWS Deployment

- AWS CLI configured with appropriate permissions
- AWS account with ECR, EC2, and S3 access
- GitHub repository with secrets configured

### 1. Setup AWS Infrastructure

#### Create ECR Repository
```bash
# Create ECR repository for your Docker images
aws ecr create-repository --repository-name chest-cancer-classifier --region us-west-2

# Get login token and authenticate Docker to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-west-2.amazonaws.com
```

#### Create S3 Bucket for Model Storage
```bash
# Create S3 bucket for storing trained models and artifacts
aws s3 mb s3://chest-cancer-models-bucket --region us-west-2
```
### 2. Setup EC2 Instance

Launch EC2 Instance, then configure it by running these commands

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure GitHub Actions Runner
# Follow GitHub's instructions to add a self-hosted runner
```

In Security Group rules of your EC2 instance,
1. Allow HTTP traffic on port 8080
2. Allow SSH access on port 22

### 3. Setup GitHub Actions Runner
Follow GitHub's instructions to add a self-hosted runner to your EC2 instance.

### 4. Deploy with GitHub Actions
```bash
# Push your code to trigger automated deployment
git add .
git commit -m "Deploy to AWS"
git push origin main
```

The GitHub Actions workflow will automatically:
- Build the Docker image
- Push to ECR  
- Deploy to your EC2 instance

### 5. Access Your Deployed Application

Once deployed, your application will be accessible at:
```
http://your-ec2-public-ip:8080
```


### DVC Pipeline
```bash
# Reproduce the entire pipeline
dvc repro

# Check pipeline status
dvc status
```

### Experiment Tracking
- All experiments are tracked in MLflow
- Model metrics, parameters, and artifacts are logged
- Easy comparison between different runs

## Docker Deployment

### Local Docker Build
```bash
docker build -t chest-cancer-classifier .
docker run -p 8080:8080 chest-cancer-classifier
```


## Configuration

### Hyperparameters Search Space (`params.yaml`)
Change these as per your requirements while before training:
```yaml
LEARNING_RATE_RANGE: [0.001, 0.01]
BATCH_SIZE_OPTIONS: [16, 32, 64]  
EPOCHS_OPTIONS: [25, 100, 200] 
N_TRIALS: 10
TIMEOUT: 7200 
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset providers for chest CT images
- PyTorch and PyTorch Lightning communities
- MLflow for experiment tracking capabilities
- AWS for cloud infrastructure

---

**Note**: This project is for educational and research purposes. Always consult healthcare professionals for medical diagnosis.
