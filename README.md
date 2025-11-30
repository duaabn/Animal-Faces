# 🐾 Animal-Faces: EfficientNetB0-Powered Animal Classification

## 🌟 Project Overview

This project is dedicated to building a robust and reliable classification system for three animal categories: **Cats**, **Dogs**, and **Wild animals**. We leverage the state-of-the-art, pre-trained **EfficientNetB0** deep learning model, fine-tuned on an animal face dataset to deliver excellent accuracy and performance.

## 🛠️ Key Technologies and Libraries

* **Core Model:** EfficientNetB0 (using a 224x224 input size).
* **Optimization:** Implementation of the **BN Unfreeze** (Batch Normalization Unfreeze) technique during fine-tuning to improve model convergence and final accuracy.
* **MLOps & Deployment:**
    * **API Framework:** **FastAPI** for serving predictions quickly and efficiently.
    * **Web Interface:** **Gradio** for creating a fast, user-friendly graphical interface for testing and demonstration.
* **Containerization:** **Docker** to encapsulate the entire environment (model, FastAPI/Gradio, dependencies) ensuring seamless reproducibility and deployment across different platforms. 
* **Development Environment:** Python 3.x, PyTorch/TensorFlow (specify the exact framework used).

## 🚀 Setup and Deployment Guide

To run the project locally, please follow these steps:

### 1. Prerequisites

Ensure you have **Docker** and **Docker Compose** installed on your system.

### 2. Build and Run the Container

Use the provided `Dockerfile` and/or `docker-compose.yml` (if available) to start the application.

```bash
# Build the Docker image
docker build -t animal-faces-classifier .

# Run the container (assuming deployment is on port 8000)
docker run -d -p 8000:8000 --name animal_app animal-faces-classifier
