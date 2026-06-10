import cv2                      # For image capture and processing
import insightface              # For deep learning-based face detection & embeddings
import numpy as np              # For numerical computations
from threading import Thread, Lock  # For parallel webcam thread handling
import smtplib                  # For sending emails
from email.message import EmailMessage
import time, os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv  # pip install python-dotenv

# ─── Load credentials from .env file (never hardcode these!) ───
load_dotenv()
EMAIL_SENDER   = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# ---------------------------
# CONFIGURATIONS
# ---------------------------
AUTHORIZED_FOLDER = "authorized_faces"
INTRUDER_FOLDER   = "intruder_faces"
ML_REPORT_FILE    = "ml_evaluation_report.txt"

# ---------------------------
# LOAD PRETRAINED FACE MODEL (InsightFace)
# ---------------------------
print("🧠 Loading pre-trained deep learning face model (InsightFace)...")
model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(224, 224))
print("✅ InsightFace model ready.\n")

# ---------------------------
# FUNCTION: Extract face embeddings from folder
# ---------------------------
def extract_embeddings(folder, label):
    """
    Converts all images in a folder into 512D face embeddings using InsightFace.
    label = 1 for Authorized, 0 for Intruder.
    """
    X, y = [], []
    if not os.path.exists(folder):
        return X, y
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = model.get(img)
        if len(faces) > 0:
            X.append(faces[0].embedding)
            y.append(label)
    return X, y

# ---------------------------
# PART 1 — OFFLINE MACHINE LEARNING EVALUATION
# ---------------------------
def run_evaluation():
    print("🔎 Running offline ML evaluation...")

    auth_X, auth_y = extract_embeddings(AUTHORIZED_FOLDER, 1)
    intr_X, intr_y = extract_embeddings(INTRUDER_FOLDER, 0)

    X = np.array(auth_X + intr_X)
    y = np.array(auth_y + intr_y)

    if len(X) < 4 or len(np.unique(y)) < 2:
        msg = (
            "⚠️ Skipping ML evaluation: Not enough labeled samples or only one class found.\n"
            "Please add images to both 'authorized_faces' and 'intruder_faces' folders.\n"
        )
        print(msg)
        with open(ML_REPORT_FILE, "w") as f:
            f.write(msg)
        return

    print(f"✅ Loaded {len(X)} samples — Authorized: {sum(y)}, Intruders: {len(y)-sum(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    report    = classification_report(y_test, y_pred, target_names=["Intruder", "Authorized"], zero_division=0)

    print("\n========================== 📊 MACHINE LEARNING EVALUATION ==========================")
    print(f"✅ Model Accuracy   : {accuracy * 100:.2f}%")
    print(f"✅ Precision Score  : {precision * 100:.2f}%")
    print(f"✅ Recall Score     : {recall * 100:.2f}%")
    print(f"✅ F1 Score         : {f1 * 100:.2f}%")
    print("\n🔹 Detailed Classification Report:")
    print(report)
    print("===============================================================================\n")

    with open(ML_REPORT_FILE, "w") as f:
        f.write("=================== MACHINE LEARNING EVALUATION ===================\n")
        f.write(f"Accuracy  : {accuracy * 100:.2f}%\n")
        f.write(f"Precision : {precision * 100:.2f}%\n")
        f.write(f"Recall    : {recall * 100:.2f}%\n")
        f.write(f"F1 Score  : {f1 * 100:.2f}%\n\n")
        f.write(report)
        f.write("\n==================================================================\n")

    print(f"📁 Results saved as '{ML_REPORT_FILE}'\n")

    import matplotlib.pyplot as plt
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values  = [v * 100 for v in [accuracy, precision, recall, f1]]
    values += values[:1]

    angles  = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], metrics, color='grey', size=10)
    ax.set_rlabel_position(30)
    plt.yticks([20, 40, 60, 80, 100], ["20","40","60","80","100"], color="grey", size=8)
    plt.ylim(0, 100)
    ax.plot(angles, values, color="dodgerblue", linewidth=2, linestyle='solid')
    ax.fill(angles, values, color="skyblue", alpha=0.4)
    plt.title("Model Performance Overview", size=14, color="navy", y=1.08)
    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/radar_chart.png")
    plt.show()

run_evaluation()

# ============================================================
# PART 2 — REAL-TIME INTRUDER DETECTION (COSINE SIMILARITY)
# ============================================================
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()

def load_known_faces(folder_path='authorized_faces'):
    """
    Loads authorized images, extracts embeddings, returns averaged profile vector.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ Folder not found: '{folder_path}'.")

    known_embeddings = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not read image: {filename}. Skipping.")
            continue
        faces = model.get(img)
        if len(faces) > 0:
            known_embeddings.append(faces[0].embedding)
            print(f"✅ Processed {filename}")
        else:
            print(f"⚠️ No face detected in {filename}. Skipping.")

    if len(known_embeddings) == 0:
        raise ValueError("❌ No valid faces found in the folder.")

    average_embedding = np.mean(known_embeddings, axis=0)
    print(f"✅ Average face embedding created from {len(known_embeddings)} images.")
    return average_embedding

sending_lock = Lock()
sending_in_progress = set()

def send_intruder_email(image_path, intruder_id):
    """
    Sends email alert with intruder photo attached.
    Credentials loaded from .env — never hardcoded.
    """
    def _send():
        try:
            msg = EmailMessage()
            msg['Subject'] = "🚨 Intruder Detected!"
            msg['From']    = EMAIL_SENDER
            msg['To']      = EMAIL_RECEIVER
            msg.set_content("An intruder has been detected! See attached photo.")

            with open(image_path, 'rb') as f:
                img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='intruder.jpg')

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
                smtp.send_message(msg)
            print(f"✅ Intruder email sent for intruder {intruder_id}!")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
        finally:
            with sending_lock:
                sending_in_progress.discard(intruder_id)

    with sending_lock:
        if intruder_id not in sending_in_progress:
            sending_in_progress.add(intruder_id)
            Thread(target=_send, daemon=True).start()

def cosine_similarity(a, b):
    """Cosine similarity between two 512D face embeddings (1=identical, 0=different)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def run_realtime():
    known_face_embedding = load_known_faces(AUTHORIZED_FOLDER)
    vs = VideoStream(0).start()

    intruder_memory    = []
    DETECTION_THRESHOLD = 0.5
    EMAIL_DELAY         = 2
    CLEANUP_TIMEOUT     = 10
    SCALE_FACTOR        = 2
    SKIP_FRAMES         = 4
    frame_counter       = 0
    stored_results      = []

    print("🎥 Starting real-time intruder detection. Press 'q' to quit.")
    while True:
        ret, frame = vs.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame_counter += 1

        if frame_counter % SKIP_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=1/SCALE_FACTOR, fy=1/SCALE_FACTOR)
            faces = model.get(small_frame)
            current_intruders, stored_results = [], []

            for face in faces:
                bbox = face.bbox * SCALE_FACTOR
                x1, y1, x2, y2 = bbox.astype(int)
                embedding = face.embedding

                try:
                    similarity = cosine_similarity(known_face_embedding, embedding)
                    if similarity > DETECTION_THRESHOLD:
                        label, color = f"Authorized ({similarity:.2f})", (0, 255, 0)
                    else:
                        label, color = f"Intruder ({similarity:.2f})", (0, 0, 255)
                        current_intruders.append({'embedding': embedding, 'bbox': np.array([x1, y1, x2, y2])})
                except Exception as e:
                    label, color = "Unknown", (255, 255, 0)
                    print(f"⚠️ Similarity computation failed: {e}")

                stored_results.append({'bbox': (x1, y1, x2, y2), 'label': label, 'color': color})

            now = time.time()
            for idx, intruder in enumerate(current_intruders):
                matched = False
                for mem in intruder_memory:
                    sim = np.dot(mem['embedding'], intruder['embedding']) / (
                        np.linalg.norm(mem['embedding']) * np.linalg.norm(intruder['embedding']) + 1e-10
                    )
                    if sim > 0.7:
                        matched = True
                        mem['last_seen'] = now
                        if not mem['email_sent']:
                            if 'start_time' not in mem:
                                mem['start_time'] = now
                            elif now - mem['start_time'] >= EMAIL_DELAY:
                                x1i, y1i, x2i, y2i = intruder['bbox'].astype(int)
                                face_crop = frame[y1i:y2i, x1i:x2i]
                                intruder_image_path = f"intruder_{idx}.jpg"
                                cv2.imwrite(intruder_image_path, face_crop)
                                send_intruder_email(intruder_image_path, idx)
                                mem['email_sent'] = True
                        break
                if not matched:
                    intruder_memory.append({'embedding': intruder['embedding'], 'start_time': now,
                                            'email_sent': False, 'last_seen': now})

            intruder_memory = [
                mem for mem in intruder_memory
                if now - mem.get('last_seen', now) <= CLEANUP_TIMEOUT
            ]

        for result in stored_results:
            x1, y1, x2, y2 = result['bbox']
            label, color = result['label'], result['color']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Intruder Detection CCTV", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

run_realtime()
