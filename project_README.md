# 🔐 AI-Based Real-Time Intruder Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/InsightFace-512D_Embeddings-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenCV-Real--Time-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-KNN_Classifier-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SMTP-Email_Alerts-00C851?style=for-the-badge&logo=gmail&logoColor=white"/>
</p>

<p align="center">
  A computer vision security system that uses deep learning face embeddings to distinguish authorized users from intruders in real time — with automatic email alerts containing a photo of the intruder.
</p>

---

## 🎯 What This Does

| Feature | Description |
|--------|-------------|
| 🧠 **Face Embedding Extraction** | Uses InsightFace (ArcFace backbone) to generate 512-dimensional face vectors |
| 📐 **Cosine Similarity Matching** | Compares live face embeddings against an authorized user profile |
| 🤖 **KNN ML Evaluation** | Offline classifier trained on embeddings with full accuracy/precision/recall/F1 metrics |
| 📊 **Radar Chart Output** | Auto-generates a performance visualization of the ML model |
| 📧 **Automated Email Alerts** | Sends an email with the intruder's cropped face photo via Gmail SMTP |
| 🎥 **Threaded Video Stream** | Frame-skipping + multithreaded webcam for efficient real-time processing |
| 🔒 **Intruder Memory Tracking** | Prevents duplicate alerts for the same person using embedding memory |

---

## 🏗️ System Architecture

```
Webcam Feed
    │
    ▼
[VideoStream Thread] ──► Frame Queue
    │
    ▼
[InsightFace Model] ──► 512D Face Embedding
    │
    ├──► [Cosine Similarity] ──► Authorized ✅  (green box)
    │         vs                 Intruder   ❌  (red box)
    │    Known Embedding
    │
    └──► [Intruder Memory] ──► Persist > 2s? ──► [Email Alert Thread]
                                                        │
                                                   Gmail SMTP
                                              (cropped face attached)

Offline Evaluation (runs first):
[authorized_faces/] + [intruder_faces/]
    │
    ▼
[InsightFace Embeddings] ──► [KNN Classifier] ──► Accuracy / Precision / Recall / F1
                                                  ──► Radar Chart (matplotlib)
                                                  ──► ml_evaluation_report.txt
```

---

## 📁 Project Structure

```
intruder-detection/
│
├── main.py                     # Main application (detection + ML evaluation)
├── requirements.txt            # Python dependencies
├── .env.example                # Template for email credentials
├── .gitignore                  # Excludes .env, face images, model cache
│
├── authorized_faces/           # ⚠️ Add your own images here (not in repo)
│   ├── face1.jpg
│   └── face2.jpg
│
├── intruder_faces/             # ⚠️ Add intruder samples here (not in repo)
│   ├── unknown1.jpg
│   └── unknown2.jpg
│
├── assets/
│   ├── radar_chart.png         # Auto-generated ML performance chart
│   └── demo_screenshot.png     # Live detection screenshot
│
└── ml_evaluation_report.txt    # Auto-generated ML metrics report
```

---

## ⚙️ How It Works

### Part 1 — Offline ML Evaluation

Before launching the live feed, the system runs a full ML pipeline:

1. Loads images from `authorized_faces/` (label = 1) and `intruder_faces/` (label = 0)
2. Passes each image through InsightFace → extracts 512D embeddings
3. Splits data 70/30 train/test using stratified sampling
4. Trains a **K-Nearest Neighbors (KNN)** classifier on the embeddings
5. Evaluates with **Accuracy, Precision, Recall, F1-Score**
6. Generates a **radar chart** and saves a full text report

### Part 2 — Real-Time Detection

1. Loads all authorized face images → averages their embeddings into a single **authorized profile vector**
2. Starts a **background thread** for the webcam (no frame drops)
3. Every 4th frame: resizes to 50% → runs InsightFace detection → extracts embeddings
4. Computes **cosine similarity** between detected face and authorized profile
   - `similarity > 0.5` → **Authorized** (green bounding box)
   - `similarity ≤ 0.5` → **Intruder** (red bounding box)
5. If an intruder persists for **2+ seconds**, crops their face and sends an **email alert**
6. **Intruder memory** prevents duplicate alerts using secondary embedding comparison (threshold: 0.7)

---

## 📊 Sample Output

### Terminal
```
🧠 Loading pre-trained deep learning face model (InsightFace)...
✅ InsightFace model ready.

🔎 Running offline ML evaluation...
✅ Loaded 24 samples — Authorized: 12, Intruders: 12

========================== 📊 MACHINE LEARNING EVALUATION ==========================
✅ Model Accuracy   : 91.67%
✅ Precision Score  : 93.33%
✅ Recall Score     : 90.00%
✅ F1 Score         : 91.63%

🔹 Detailed Classification Report:
              precision    recall  f1-score   support
    Intruder       0.90      0.90      0.90         10
  Authorized       0.94      0.92      0.93        14
    accuracy                           0.92        24
   macro avg       0.92      0.91      0.92        24
weighted avg       0.92      0.92      0.92        24

📁 Results saved as 'ml_evaluation_report.txt'
🎥 Starting real-time intruder detection. Press 'q' to quit.
✅ Intruder email sent for intruder 0!
```

### Live Feed
```
┌────────────────────────────────────┐
│  Intruder Detection CCTV           │
│                                    │
│   ┌──────────┐                     │
│   │ Khushboo │  ← green box        │
│   │  (0.87)  │  Authorized ✅      │
│   └──────────┘                     │
│                                    │
│         ┌──────────┐               │
│         │ Unknown  │  ← red box    │
│         │  (0.31)  │  Intruder ❌  │
│         └──────────┘               │
│                                    │
│  Press 'q' to quit                 │
└────────────────────────────────────┘
```

### Email Alert
```
Subject: 🚨 Intruder Detected!
Body:    An intruder has been detected! See attached photo.
Attach:  intruder.jpg  (cropped face from live frame)
```

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Khushii0213/intruder-detection.git
cd intruder-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure email credentials
```bash
cp .env.example .env
# Edit .env with your Gmail address and App Password
```
> 🔑 Generate a Gmail App Password at: https://myaccount.google.com/apppasswords

### 5. Add face images
```
authorized_faces/     ← Add 5–10 clear photos of the authorized person
intruder_faces/       ← Add 5–10 photos of other people (for ML evaluation)
```

### 6. Run
```bash
python main.py
```

---

## 🔧 Key Parameters (in `main.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DETECTION_THRESHOLD` | `0.5` | Cosine similarity cutoff. Raise to be stricter. |
| `EMAIL_DELAY` | `2` sec | How long intruder must persist before alert is sent |
| `CLEANUP_TIMEOUT` | `10` sec | How long to remember a tracked intruder |
| `SKIP_FRAMES` | `4` | Process every Nth frame (lower = more CPU usage) |
| `SCALE_FACTOR` | `2` | Downscale factor for detection (higher = faster) |

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Face Detection & Embeddings | InsightFace (ArcFace, SCRFD) |
| Real-time Video | OpenCV + Python threading |
| ML Classifier | scikit-learn KNeighborsClassifier |
| Similarity Metric | Cosine Similarity on 512D vectors |
| Email Alerts | smtplib + Gmail SMTP SSL |
| Visualization | Matplotlib (radar chart) |
| Credentials Management | python-dotenv |

---

## 🔮 Future Improvements

- [ ] Add support for multiple authorized users
- [ ] Replace KNN with a neural network-based classifier
- [ ] Integrate with Raspberry Pi for edge deployment
- [ ] Add a web dashboard (Flask/FastAPI) for live monitoring
- [ ] Store intruder logs in a database
- [ ] Support for IP cameras (RTSP streams)
- [ ] Push notifications via Telegram Bot API

---

## 👩‍💻 Author

**Khushboo** — Robotics & AI Engineering Student, Thapar Institute  
🌐 [Portfolio](https://khushii0213.github.io) · 💼 [LinkedIn](https://linkedin.com/in/khushboo-ab365a283) · 🐙 [GitHub](https://github.com/Khushii0213)

---

## ⚠️ Security Note

This project uses environment variables for email credentials. **Never commit your `.env` file.** The `.gitignore` is pre-configured to exclude it. See `.env.example` for the required format.

---

<p align="center">Made with 🤖 for computer vision security research</p>
