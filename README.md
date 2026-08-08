# 🛡️ Intent Detection System using Computer Vision & Deep Learning

<p align="center">
  <img src="assets/banner.png" alt="Intent Detection System Banner" width="100%">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green.svg)
![License](https://img.shields.io/badge/License-MIT-success.svg)

</p>

---

# 📌 Overview

The **Intent Detection System** is an AI-powered surveillance solution designed to **predict suspicious human intentions before an incident occurs** rather than simply detecting events after they happen.

Unlike traditional CCTV analytics that only identify objects or actions, this system continuously analyzes human movement, interaction patterns, trajectories, dwell time, and temporal behavior to estimate the probability of suspicious intent.

The project combines **Computer Vision**, **Object Detection**, **Multi-Object Tracking**, **Behavior Analysis**, and **Deep Learning** to provide proactive security monitoring for smart surveillance environments.

---

# 🎯 Problem Statement

Conventional surveillance systems suffer from several limitations:

* Detect incidents only after they occur.
* Require continuous human monitoring.
* Cannot understand behavioral intent.
* Fail to recognize suspicious activity before escalation.

This project addresses these limitations by introducing an intelligent system capable of learning behavioral patterns and predicting potentially suspicious intentions in real time.

---

# 🚀 Key Features

* Real-time person detection
* Multi-object tracking
* Human trajectory analysis
* Behavioural feature extraction
* Temporal sequence modeling
* AI-based intent prediction
* Risk score generation
* Live surveillance monitoring
* Real-time alert generation
* Modular deep learning architecture

---

# 🏗 System Architecture

```
Input Video
      │
      ▼
Frame Extraction
      │
      ▼
YOLOv8 Person Detection
      │
      ▼
Centroid Based Tracking
      │
      ▼
Trajectory Generation
      │
      ▼
Behaviour Feature Extraction
      │
      ▼
Sequence Generation
      │
      ▼
LSTM Intent Prediction Model
      │
      ▼
Risk Score Estimation
      │
      ▼
Real-Time Alert System
```

---

# 🧠 Workflow

## Step 1 — Video Input

The system accepts

* Live CCTV Stream
* Recorded Video
* IP Camera Feed

Every video is processed frame-by-frame.

---

## Step 2 — Person Detection

Each frame is passed through a **YOLOv8** object detection model.

The detector identifies

* Person
* Bounding Box
* Confidence Score

Example

```
Frame

+--------------------+
|        Person      |
|      ███████       |
|      ███████       |
+--------------------+
```

---

## Step 3 — Multi-Object Tracking

Detection alone cannot identify individuals across multiple frames.

A centroid-based tracking algorithm assigns a unique ID to every detected person.

Example

```
Person A → ID 1

Person B → ID 2

Person C → ID 3
```

This enables continuous behavioral analysis.

---

## Step 4 — Trajectory Generation

Every tracked individual generates a movement trajectory.

Example

```
Frame 1 → (120,210)

Frame 2 → (125,216)

Frame 3 → (132,220)

Frame 4 → (140,228)
```

These trajectories represent movement patterns over time.

---

## Step 5 — Behaviour Feature Extraction

Instead of using raw images, meaningful behavioural features are extracted for each tracked individual.

Features include:

* X Position
* Y Position
* Velocity
* Direction
* Acceleration
* Distance Travelled
* Dwell Time
* Turning Rate
* Interaction Duration
* Movement Consistency

These features form a compact numerical representation of human behaviour.

---

## Step 6 — Sequence Generation

Intent cannot be predicted from a single frame.

The system creates temporal sequences of behavioural data.

Example

```
T1

T2

T3

T4

...

T30
```

Each sequence represents a person's behaviour over time.

---

## Step 7 — LSTM Intent Prediction

The generated sequences are fed into an **LSTM (Long Short-Term Memory)** neural network.

The model learns

* Normal behaviour
* Abnormal movement
* Suspicious trajectories
* Behaviour evolution over time

The LSTM outputs an intent probability.

Example

```
Normal : 0.08

Suspicious : 0.92
```

---

## Step 8 — Risk Score Calculation

The prediction is converted into a risk score.

Example

| Probability | Risk Level |
| ----------- | ---------- |
| 0–0.30      | Low        |
| 0.30–0.60   | Medium     |
| 0.60–0.80   | High       |
| 0.80–1.00   | Critical   |

---

## Step 9 — Real-Time Alerts

When the risk exceeds a predefined threshold, the system generates alerts.

Example

```
⚠ Suspicious Behaviour Detected

Person ID : 4

Risk Score : 0.91

Location : Camera 2

Timestamp : 12:41:28
```

---

# 🤖 Deep Learning Model

Model Used

* LSTM Network

Input

Behaviour Feature Sequence

Output

Intent Probability

Loss Function

* Binary Cross Entropy

Optimizer

* Adam Optimizer

Framework

* PyTorch

---

# 🛠 Technology Stack

## Programming

* Python

## Computer Vision

* OpenCV

## Object Detection

* YOLOv8

## Deep Learning

* PyTorch

## Tracking

* Centroid Tracking

## Numerical Computing

* NumPy
* Pandas

## Visualization

* Matplotlib

---

# 📂 Project Structure

```
Intent-Detection-System/

│

├── dataset/

├── models/

│ ├── yolov8/

│ └── lstm/

│

├── tracking/

│

├── feature_extraction/

│

├── preprocessing/

│

├── inference/

│

├── utils/

│

├── outputs/

│

├── assets/

│

├── train.py

├── predict.py

├── requirements.txt

└── README.md
```

---

# ⚙ Installation

Clone repository

```bash
git clone https://github.com/yourusername/Intent-Detection-System.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python predict.py
```

---

# 📈 Applications

* Smart Surveillance
* Airport Security
* Railway Stations
* Shopping Malls
* Public Events
* Smart Cities
* Industrial Security
* Campus Monitoring

---

# 🔮 Future Improvements

* Transformer-based temporal models
* Action recognition integration
* Pose estimation
* Edge AI deployment
* Multi-camera tracking
* Person re-identification
* Real-time dashboard
* Cloud deployment
* Mobile monitoring application

---

# 📚 Research Contribution

This project demonstrates how combining **computer vision**, **multi-object tracking**, and **temporal deep learning** can move surveillance systems beyond simple object detection toward **predictive intent analysis**, enabling earlier identification of potentially suspicious behaviour.

---

# 👨‍💻 Author

**Divyam Goel**

B.Tech Computer Science (AI & ML)

Artificial Intelligence • Deep Learning • Computer Vision • Machine Learning

---

# ⭐ If you found this project useful

Please consider giving this repository a ⭐ to support the project.
