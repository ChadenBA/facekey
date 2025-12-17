# FaceKey 🔐🤖

**Secure & Intelligent Face Recognition System**

FaceKey is a production-ready **AI-powered face recognition application** designed with a strong focus on **security, scalability, and observability**.

It combines **Computer Vision, MLOps, DevOps, and Cloud-native practices** to deliver a robust authentication system.


---

## 🚀 Features

### 🧠 AI & Computer Vision Engine

The core authentication logic utilizes a **Multi-Layer Perceptron (MLP)** with a custom 3-4-3-1 architecture.

* **Optimized Architecture**:
* **Input Layer**: 3 neurons (x_1, x_2, x_3) representing facial feature vectors.
* **Hidden Layer 1**: 4 neurons using **ReLU** activation for non-linear feature extraction.


* **Hidden Layer 2**: 3 neurons for deep pattern recognition.
* **Output Layer**: 1 neuron with **Sigmoid** activation to produce a probability score between 0 and 1.




* **Fine-tuned Recognition**: Accurately identifies users with or without glasses.
* **Anti-Spoofing**: Presentation attack detection (photos, videos, screen replays).

---

## 🏗️ Core Algorithm: Forward Propagation

The system processes authentication requests through a sequential forward pass :

1. **Stage A (Input to Hidden 1)**:
* Calculation: z^{(1)} = W^{(1)}x + b^{(1)}
* Activation: a^{(1)} = \text{ReLU}(z^{(1)})


2. **Stage B (Hidden 1 to Hidden 2)**:
* Calculation: z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}
* Activation: a^{(2)} = \text{ReLU}(z^{(2)})


3. **Stage C (Hidden 2 to Output)**:
* Calculation: z^{(3)} = W^{(3)}a^{(2)} + b^{(3)}
* Prediction: \hat{y} = \sigma(z^{(3)}) = \frac{1}{1 + e^{-z^{(3)}}}



**Numerical Example Logic**:
Given an input vector x = [1, 0.5, -1], the internal pass follows this verified path:

* z^{(1)} = [0.8, -1.2, 0.4, 2.0] \implies a^{(1)} = [0.8, 0, 0.4, 2.0]
* z^{(2)} = [1.1, -0.3, 0.5] \implies a^{(2)} = [1.1, 0, 0.5]
* z^{(3)} = 1.2 \implies \text{Authentication Probability} = 0.77

---

## 🛠️ Tech Stack

**Frontend**

* React + Vite
* JavaScript / TypeScript

**Backend**

* Python + FastAPI
* Uvicorn (High-performance inference)

**DevOps & Cloud**

* Docker (Containerization)
* Kubernetes (Orchestration & Scalability)
* CI/CD (GitHub Actions / GitLab CI)

**Monitoring & Security**

* **Prometheus**: System and model metrics collection.
* **Grafana**: Real-time visualization dashboards.
* **Trivy**: Container and dependency vulnerability scanning.

---



## ⚙️ Running the Project Locally

### 🔹 Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```

📍 **API Docs**: `http://localhost:8000/docs`

### 🔹 Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev

```

### 🐳 Docker & ☸️ Kubernetes

```bash
# Build images
docker build -t facekey-backend./backend
docker build -t facekey-frontend./frontend

# Deploy manifests
kubectl apply -f k8s/

```

---

## 📊 Observability & Security

* **Security Scanning**: Trivy is integrated into the CI/CD pipeline to prevent deployments with critical vulnerabilities.
* **Email Alerts**: Automatic notifications triggered upon detection of spoofing/presentation attacks.
* **Performance Metrics**: Inference latency and model accuracy tracking via Grafana.

---

## 📄 License

This project is licensed under the MIT License.
