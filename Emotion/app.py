import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from datetime import datetime
from deepface import DeepFace

# ---------------- APP SETUP ----------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- FEEDBACK ----------------
def get_feedback(emotion):
    return {
        "Happy": "Great! Keep smiling 😊",
        "Sad": "Take a break and relax 💙",
        "Angry": "Try deep breathing 😌",
        "Fear": "Stay calm 💪",
        "Surprise": "Interesting 😲",
        "Disgust": "Stay positive 😐",
        "Neutral": "Have a nice day 🙂"
    }.get(emotion, "")

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        return redirect("/dashboard")
    return render_template("login.html")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/camera")
def camera():
    return render_template("camera.html")

# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if request.method == "POST":
        file = request.files.get("image")

        if file is None or file.filename == "":
            return redirect("/upload")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Read image safely
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "Error loading image. Please upload JPG or PNG."

        try:
            # 🔥 FAST EMOTION ONLY
            analysis = DeepFace.analyze(
                img_path=filepath,
                actions=['emotion'],
                enforce_detection=False
            )

            # Handle list output
            if isinstance(analysis, list):
                analysis = analysis[0]

            # ---------------- EMOTION ----------------
            dominant_emotion = analysis.get('dominant_emotion', 'neutral')
            emotion = dominant_emotion.capitalize()

            confidence = round(
                analysis['emotion'].get(dominant_emotion, 0), 2
            )

            feedback = get_feedback(emotion)

            # ---------------- LOG HISTORY ----------------
            with open("history.txt", "a") as f:
                f.write(f"{datetime.now()} - {emotion}\n")

            # ---------------- RESULT ----------------
            return render_template(
                "upload.html",
                emotion=emotion,
                confidence=confidence,
                feedback=feedback,
                image_file=filename
            )

        except Exception as e:
            print(e)
            return render_template(
                "upload.html",
                emotion="No face detected",
                confidence=0,
                feedback="No feedback",
                image_file=filename
            )

    return render_template("upload.html")


@app.route("/start_camera")
def start_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list):
                result = result[0]

            emotion = result['dominant_emotion']

            # Show emotion on screen
            cv2.putText(frame, emotion, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        except:
            pass

        cv2.imshow("Emotion Detection - Press Q to Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return redirect("/dashboard")

# ---------------- HISTORY ----------------
@app.route("/history")
def history():
    try:
        with open("history.txt", "r") as f:
            data = f.readlines()
    except:
        data = []
    return render_template("history.html", data=data)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=False)