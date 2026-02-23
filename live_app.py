"""
EmotiScan Live - Real-time Face Detection & Emotion Recognition
Owner: Kriskumar Gadara

Works on any device (phone, laptop, PC) via browser.
Access from other devices on the same network using http://<your-ip>:5000
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

# Add EmotiEffLib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "EmotiEffLib"))

from emotiefflib.facial_analysis import EmotiEffLibRecognizer

# ── App config ───────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Face detector (OpenCV Haar Cascade) ──────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Emotion recognizer (ONNX) ───────────────────────────────────────────────
MODEL_NAME = "enet_b0_8_best_vgaf"
print(f"[EmotiScan] Loading model: {MODEL_NAME} ...")
emotion_model = EmotiEffLibRecognizer(engine="onnx", model_name=MODEL_NAME)
print("[EmotiScan] Model loaded successfully!")

# Emotion -> colour map (BGR for OpenCV)
EMOTION_COLORS = {
    "Anger": (0, 0, 255),
    "Contempt": (128, 0, 128),
    "Disgust": (0, 128, 0),
    "Fear": (0, 165, 255),
    "Happiness": (0, 255, 0),
    "Neutral": (200, 200, 200),
    "Sadness": (255, 0, 0),
    "Surprise": (0, 255, 255),
}

# ── Shared state for the video stream ────────────────────────────────────────
output_frame = None
lock = threading.Lock()
camera_active = False


def process_camera():
    """Capture frames from webcam, detect faces, classify emotions, and stream."""
    global output_frame, camera_active

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[EmotiScan] ERROR: Cannot open webcam. Trying index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[EmotiScan] ERROR: No camera found.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera_active = True
    print("[EmotiScan] Camera started. Streaming live...")

    fps_time = time.time()
    fps = 0
    frame_count = 0

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            continue

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        for x, y, w, h in faces:
            # Extract face ROI and convert to RGB for model
            face_roi = frame[y : y + h, x : x + w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            try:
                emotions, scores = emotion_model.predict_emotions(face_rgb, logits=False)
                emotion = emotions[0]
                confidence = float(np.max(scores[0]))

                # Get top 3 emotions for display
                class_names = list(emotion_model.idx_to_emotion_class.values())
                sorted_idx = np.argsort(scores[0])[::-1]
                top3 = [(class_names[i], scores[0][i]) for i in sorted_idx[:3]]
            except Exception:
                emotion = "Unknown"
                confidence = 0.0
                top3 = []

            # Draw bounding box
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label background + text
            label = f"{emotion} {confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 14), (x + tw + 4, y), color, -1)
            cv2.putText(
                frame, label, (x + 2, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
            )

            # Draw top-3 emotion bars on the right side of the face box
            bar_x = x + w + 5
            bar_y_start = y
            for rank, (emo_name, emo_score) in enumerate(top3):
                bar_y = bar_y_start + rank * 22
                bar_w = int(emo_score * 100)
                bar_color = EMOTION_COLORS.get(emo_name, (180, 180, 180))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16), bar_color, -1)
                cv2.putText(
                    frame, f"{emo_name[:3]} {emo_score:.0%}", (bar_x + bar_w + 4, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                )

        # Draw header bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (40, 40, 40), -1)
        cv2.putText(
            frame, f"EmotiScan Live | FPS: {fps:.1f} | Faces: {len(faces)}",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1,
        )

        with lock:
            output_frame = frame.copy()

    cap.release()
    print("[EmotiScan] Camera stopped.")


def generate_stream():
    """Yield MJPEG frames for the browser."""
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)  # ~30 fps cap


# ── HTML template (mobile-friendly) ─────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>EmotiScan Live</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a0a;
            color: #e0e0e0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        header {
            width: 100%;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            padding: 16px 20px;
            text-align: center;
            border-bottom: 2px solid #00ffc8;
        }
        header h1 {
            font-size: 1.6rem;
            color: #00ffc8;
            margin-bottom: 4px;
        }
        header p {
            font-size: 0.85rem;
            color: #8892b0;
        }
        .stream-container {
            margin: 20px auto;
            max-width: 960px;
            width: 95%;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid #00ffc8;
            box-shadow: 0 0 30px rgba(0, 255, 200, 0.15);
        }
        .stream-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        .info-bar {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 12px;
            padding: 16px;
            max-width: 960px;
            width: 95%;
        }
        .emotion-tag {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid;
        }
        .emotion-tag.anger { background: rgba(255,0,0,0.15); border-color: #f00; color: #ff4444; }
        .emotion-tag.contempt { background: rgba(128,0,128,0.15); border-color: #800080; color: #c060c0; }
        .emotion-tag.disgust { background: rgba(0,128,0,0.15); border-color: #080; color: #44cc44; }
        .emotion-tag.fear { background: rgba(255,165,0,0.15); border-color: #ffa500; color: #ffbb44; }
        .emotion-tag.happiness { background: rgba(0,255,0,0.15); border-color: #0f0; color: #44ff44; }
        .emotion-tag.neutral { background: rgba(200,200,200,0.15); border-color: #aaa; color: #ccc; }
        .emotion-tag.sadness { background: rgba(0,0,255,0.15); border-color: #00f; color: #4488ff; }
        .emotion-tag.surprise { background: rgba(0,255,255,0.15); border-color: #0ff; color: #44ffff; }
        footer {
            margin-top: auto;
            padding: 14px;
            text-align: center;
            font-size: 0.75rem;
            color: #555;
            border-top: 1px solid #222;
            width: 100%;
        }
        footer a { color: #00ffc8; text-decoration: none; }
        @media (max-width: 600px) {
            header h1 { font-size: 1.2rem; }
            .info-bar { gap: 6px; }
            .emotion-tag { font-size: 0.7rem; padding: 4px 10px; }
        }
    </style>
</head>
<body>
    <header>
        <h1>&#127917; EmotiScan Live</h1>
        <p>Real-time Face Detection &amp; Emotion Recognition &mdash; by Kriskumar Gadara</p>
    </header>

    <div class="stream-container">
        <img src="/video_feed" alt="Live Camera Stream">
    </div>

    <div class="info-bar">
        <span class="emotion-tag anger">Anger</span>
        <span class="emotion-tag contempt">Contempt</span>
        <span class="emotion-tag disgust">Disgust</span>
        <span class="emotion-tag fear">Fear</span>
        <span class="emotion-tag happiness">Happiness</span>
        <span class="emotion-tag neutral">Neutral</span>
        <span class="emotion-tag sadness">Sadness</span>
        <span class="emotion-tag surprise">Surprise</span>
    </div>

    <footer>
        EmotiScan &copy; 2026 Kriskumar Gadara |
        <a href="https://github.com/Kris-gadara/EmotiScan" target="_blank">GitHub</a>
    </footer>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    # Start camera processing in a background thread
    cam_thread = threading.Thread(target=process_camera, daemon=True)
    cam_thread.start()

    # Get local IP for cross-device access
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"\n{'='*60}")
    print(f"  EmotiScan Live - Real-time Emotion Detection")
    print(f"  Owner: Kriskumar Gadara")
    print(f"{'='*60}")
    print(f"  Open in browser:")
    print(f"    Local:   http://127.0.0.1:5000")
    print(f"    Network: http://{local_ip}:5000")
    print(f"")
    print(f"  Access from phone/tablet/other PC using the Network URL")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
