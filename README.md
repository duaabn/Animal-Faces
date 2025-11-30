# 🐾 Animal-Faces: High-Reliability Classification Client (MLOps via External API)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)]

## 🌟 Project Overview

This repository hosts the **Client Interface** for a high-reliability animal classification model (Cats, Dogs, Wild).

The core classification logic is **not run locally**. Instead, this application is designed to act as a **Gradio Client** that sends image data to an **externally deployed FastAPI API** on Hugging Face Spaces and displays the prediction result. This is a common pattern in MLOps for managing model serving separately from the front-end application.

---

## 🛠️ Key Technologies and Components

| Component | Technology | Role in this Repository |
| :---: | :---: | :---: |
| **Deployed Model** | EfficientNetB0 | Runs on the external server (Hugging Face Spaces). |
| **External API** | **FastAPI** | Provides the prediction endpoint for image submission. |
| **Local Interface** | Gradio | Simple web interface for local user interaction. |
| **Containerization** | Docker | Ensures the local Gradio client runs reliably. |

---

## 🔗 External API Endpoint

The local application connects to the following **live API endpoint** for all predictions:

* **API Documentation (OpenAPI/Swagger):** [https://duaabn555-animalfacesv2.hf.space/docs#/default/predict_animal_type_predict_animal__post](https://duaabn555-animalfacesv2.hf.space/docs#/default/predict_animal_type_predict_animal__post)

---

## 🚀 Deployment Guide (Using Docker)

Running this project involves building and starting the local Gradio client container.

### 1. Requirements

* **Docker** installed and running on your system.

### 2. Build and Run

Execute these commands from the root directory of the repository:

```bash
# 1. Build the Docker Image
# This creates the environment for the Gradio Client application.
docker build -t animal-faces-client .

# 2. Run the Container
# Map the container's internal port 7860 to your host machine.
docker run -d -p 7860:7860 --name animal_client animal-faces-client
