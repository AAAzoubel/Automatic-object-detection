
# 🚀 AI-Powered Video Analytics System

End-to-end video processing pipeline using Computer Vision, AI, and Backend Engineering.

This project goes beyond object detection — it transforms raw video into structured, usable data.

---



---

## 🔍 Features

- Object detection using YOLOv8  
- Object tracking with persistent IDs (avoiding duplicate counts)  
- FastAPI backend for video processing  
- Structured outputs: JSON, CSV, Excel  
- Basic web interface for interaction  

---

## 🧠 Tech Stack

- Python  
- FastAPI  
- YOLOv8 (Ultralytics)  
- OpenCV  
- Pandas  

---

## ⚙️ How It Works

1. Upload a video  
2. Process frame-by-frame  
3. Detect and track objects  
4. Generate structured outputs  
5. Return results via API  

---

## 📁 Project Structure


├── api.py
├── object_detector.py
├── requirements.txt
├── README.md


---

## ▶️ How to Run

```bash
git clone https://github.com/AAAzoubel/Automatic-object-detection
cd Automatic-object-detection

pip install -r requirements.txt

uvicorn api:app --reload

Then open:
http://127.0.0.1:8000/ui

🚧 Next Steps
Database integration
Real-time processing
Performance optimization
Docker containerization
🤝 Contributing

Feel free to fork and test locally. Feedback is always welcome!
