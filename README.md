# 🚀 AI-Powered Video Analytics System

A complete end-to-end video processing pipeline built with Computer Vision, AI, and Backend Engineering.

This project goes beyond object detection — it transforms raw video into structured, usable data for analysis.

---

## 🎯 Features

- 🎥 Object detection using YOLOv8  
- 🔄 Object tracking across frames (unique IDs)  
- ⚙️ FastAPI backend for video processing  
- 📊 Structured outputs (JSON, CSV, Excel)  
- 🌐 Simple web interface for interaction  
- 🎬 Annotated output video generation  

---

## 🧠 Key Concept

One of the main challenges addressed in this project is **tracking consistency**:

> Ensuring that the same object is not counted multiple times across frames.

This is critical for turning raw detections into reliable data.

---

## 🏗️ Architecture

1. User uploads a video  
2. Backend processes it frame-by-frame  
3. YOLOv8 performs detection + tracking  
4. Data is structured and stored  
5. Outputs are generated:
   - Annotated video  
   - CSV / Excel files  
   - JSON summary  

---

## ⚙️ Tech Stack

- Python  
- FastAPI  
- YOLOv8 (Ultralytics)  
- OpenCV  
- Pandas  

---

## 📂 Project Structure

```bash
project/
│
├── api.py                # FastAPI application
├── object_detector.py   # Core detection & processing logic
├── uploads/             # Uploaded videos
├── outputs/             # Generated results
└── README.md
