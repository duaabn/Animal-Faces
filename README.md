# 🐾 Animal-Faces: High-Reliability Classification using EfficientNetB0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)]

## 🌟 Project Overview

This project provides a **high-reliability animal classification system** capable of distinguishing between **Cats**, **Dogs**, and **Wild** animals. It uses the pre-trained **EfficientNetB0** deep learning architecture and features a robust deployment strategy based on **MLOps** principles.

The model is served using **FastAPI** (as a backend) and demonstrated with a user-friendly **Gradio** interface, all packaged within a **Docker** container for consistent and reproducible deployment.

---

## 🛠️ Key Technologies and Components

| Component | Technology | Purpose |
| :---: | :---: | :---: |
| **Model** | EfficientNetB0 (224x224) | High-performance base classifier. |
| **Optimization** | BN Unfreeze | Fine-tuning technique used in the notebook (`ipynb`). |
| **API** | FastAPI | Fast, asynchronous backend for prediction serving. |
| **Interface** | Gradio | Simple web interface for live testing. |
| **Deployment** | Docker | Containerization for reproducible MLOps workflow. |

---

## ⬇️ Prerequisites: Model Weights Download

**⚠️ IMPORTANT:** The actual trained model weights (`best_efficientnetb0_224_model.keras`) are **not** stored on GitHub due to file size limits.

**Before running the Docker commands, you must:**

1.  **Download the Model:** Get the file from the external host: **[PLACE YOUR DOWNLOAD LINK HERE]**
    *(Make sure this link is publicly accessible.)*
2.  **Place the File:** Save the downloaded file (`best_efficientnetb0_224_model.keras`) directly into the **root directory** of this repository.

---

## 🚀 Deployment Guide (Using Docker)

The easiest way to run the classifier is by using Docker, which ensures all dependencies are correctly handled.

### 1. Requirements

* **Docker** installed and running on your system.
* **Model Weights** file placed in the root directory (as described above).

### 2. Build and Run

Execute these commands from the root directory of the repository:

```bash
# 1. Build the Docker Image
# This step uses the Dockerfile and installs all dependencies from requirements.txt
docker build -t animal-faces-classifier .

# 2. Run the Container
# We map the container's internal port 7860 (where Gradio runs) to your host port 7860
docker run -d -p 7860:7860 --name animal_app animal-faces-classifier
# Run the container (assuming deployment is on port 8000)
docker run -d -p 8000:8000 --name animal_app animal-faces-classifier
