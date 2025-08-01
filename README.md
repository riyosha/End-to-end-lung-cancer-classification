# End to end chest cancer classification

AWS URI - 473548817706.dkr.ecr.us-west-2.amazonaws.com/cv-cancer-classification
http://35.91.236.2:8080/

## 🫁 About This Project

This project tackles the critical challenge of automated chest cancer detection using deep learning. It classifies CT scan images into four categories:
- **Adenocarcinoma** (left lower lobe, T2 N0 M0 Ib)
- **Large Cell Carcinoma** (left hilum, T2 N2 M0 IIIa) 
- **Normal** (healthy tissue)
- **Squamous Cell Carcinoma** (left hilum, T1 N2 M0 IIIa)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | PyTorch, PyTorch Lightning |
| **MLOps** | MLflow, DVC |
| **Web Framework** | Flask |
| **Cloud** | AWS (S3, EC2, ECR) |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker |

## MLOps Highlights

- **Automated Pipeline**: End-to-end training from data ingestion to deployment
- **Lightweight Containers**: 56MB smaller Docker images (models stored on S3)
- **Model Versioning**: DVC integration for reproducible ML workflows
- **Cloud-Native**: S3 model storage with runtime downloading
- **Experiment Tracking**: MLflow for metrics and model management
- **CI/CD Automation**: GitHub Actions for seamless deployments

## Deployment Flow

1. **Local Training** → Model trained and evaluated locally
2. **S3 Upload** → Best model automatically pushed to S3 bucket
3. **Docker Build** → Lightweight container (no model included)
4. **ECR Push** → Image uploaded to AWS ECR
5. **EC2 Deploy** → Container deployed to EC2 instance
6. **Runtime Loading** → Model downloaded from S3 on first prediction
