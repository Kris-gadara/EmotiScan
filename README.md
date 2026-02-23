<div align="center">

# 🎭 EmotiScan

### Real-time Face Detection & Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Detect faces and classify 8 emotions in real time — from your browser, on any device.**

Built by **[Kriskumar Gadara](https://github.com/Kris-gadara)**

---

</div>

## 📸 Live Demo

<div align="center">

<img src="images/Screenshot 2026-02-23 221020.png" alt="EmotiScan Live Demo" width="900"/>

_EmotiScan detecting faces and recognizing emotions in real time_

</div>

---

## ✨ Features

| Feature                    | Description                                                                  |
| -------------------------- | ---------------------------------------------------------------------------- |
| 🎥 **Live Camera Stream**  | Real-time webcam emotion detection via browser (MJPEG stream)                |
| 🖼️ **Image Upload**        | Upload any photo for face detection and emotion analysis                     |
| 📱 **Cross-Device Access** | Access from phone, tablet, or any PC on the same network                     |
| 🧠 **8 Emotion Classes**   | Anger · Contempt · Disgust · Fear · Happiness · Neutral · Sadness · Surprise |
| 📊 **Confidence Scores**   | Per-emotion probability scores with Top-3 visual bars                        |
| ⚡ **ONNX Inference**      | Fast CPU-only inference — no GPU required                                    |
| 🎨 **Color-Coded Output**  | Each emotion has a unique color for instant visual feedback                  |
| 🌐 **Mobile-Friendly UI**  | Responsive HTML5 interface with dark theme                                   |

---

## 🏗️ Architecture

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────────┐
│   Browser    │◄───►│  Flask / Gradio   │◄───►│  OpenCV + ONNX   │
│  (Any Device)│     │   Web Server      │     │  Emotion Engine  │
└──────────────┘     └───────────────────┘     └──────────────────┘
                              │
                     ┌────────┴────────┐
                     │  EmotiEffLib    │
                     │  EfficientNet   │
                     │  B0 (AffectNet) │
                     └─────────────────┘
```

**Model:** EfficientNet-B0 trained on **AffectNet + VGAF** datasets, served via ONNX Runtime for lightweight CPU inference.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- Webcam (for live mode)
- ~500 MB disk space (model + dependencies)

### 1. Clone the Repository

```bash
git clone https://github.com/Kris-gadara/EmotiScan.git
cd EmotiScan
```

### 2. Clone the Emotion Library (EmotiEffLib)

```bash
git clone https://github.com/sb-ai-lab/EmotiEffLib.git
```

### 3. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install opencv-python numpy flask gradio onnxruntime Pillow mediapipe tf-keras
pip install -r EmotiEffLib/requirements.txt
```

### 5. Download ONNX Models

```bash
cd EmotiEffLib/models/affectnet_emotions/onnx
python ../../../emotiefflib/facial_analysis.py   # auto-downloads on first run
cd ../../../..
```

> **Tip:** Models are auto-downloaded on first inference if not present.

---

## 🎮 Usage

### Option A — Live Camera Stream (Flask)

Stream your webcam to the browser with real-time emotion overlays:

```bash
python live_app.py
```

Then open:

- **Local:** http://127.0.0.1:5000
- **Network:** http://\<your-ip\>:5000 _(access from phone/tablet)_

### Option B — Image Upload (Gradio)

Upload images or use webcam snapshots with a Gradio UI:

```bash
python app.py
```

Opens automatically in your default browser at http://127.0.0.1:7860

---

## 📂 Project Structure

```
EmotiScan/
├── app.py              # Gradio-based image upload & webcam UI
├── live_app.py         # Flask-based real-time MJPEG camera stream
├── README.md
├── .gitignore
├── images/             # Screenshots & demo media
│   └── Screenshot 2026-02-23 221020.png
└── EmotiEffLib/        # Emotion recognition library (cloned separately)
    ├── emotiefflib/
    │   ├── facial_analysis.py
    │   └── ...
    └── models/
        └── affectnet_emotions/
            └── onnx/
```

---

## 🧠 Supported Emotions

<div align="center">

| Emotion          | Color           | Example                         |
| ---------------- | --------------- | ------------------------------- |
| 😡 **Anger**     | 🔴 Red          | Furrowed brows, tight lips      |
| 😤 **Contempt**  | 🟣 Purple       | Asymmetric lip corner raise     |
| 🤢 **Disgust**   | 🟢 Green        | Wrinkled nose, raised upper lip |
| 😨 **Fear**      | 🟠 Orange       | Wide eyes, open mouth           |
| 😊 **Happiness** | 💚 Bright Green | Smile, raised cheeks            |
| 😐 **Neutral**   | ⚪ Gray         | Relaxed face                    |
| 😢 **Sadness**   | 🔵 Blue         | Drooping eyelids, frown         |
| 😲 **Surprise**  | 🩵 Cyan         | Raised eyebrows, open mouth     |

</div>

---

## ⚙️ Configuration

| Parameter      | Default               | Description                           |
| -------------- | --------------------- | ------------------------------------- |
| `MODEL_NAME`   | `enet_b0_8_best_vgaf` | ONNX model for emotion classification |
| `scaleFactor`  | `1.1`                 | Haar cascade scale factor             |
| `minNeighbors` | `5`                   | Minimum detections for a valid face   |
| `minSize`      | `(50, 50)`            | Minimum face size in pixels           |
| `JPEG Quality` | `80`                  | MJPEG stream compression quality      |
| `FPS Cap`      | `~30`                 | Maximum streaming frame rate          |

---

## 🛠️ Tech Stack

- **[OpenCV](https://opencv.org/)** — Face detection (Haar Cascade) & image processing
- **[ONNX Runtime](https://onnxruntime.ai/)** — Fast, portable model inference
- **[EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib)** — Emotion recognition backbone (EfficientNet-B0)
- **[Flask](https://flask.palletsprojects.com/)** — Lightweight web server for live MJPEG streaming
- **[Gradio](https://gradio.app/)** — Interactive ML web UI for image upload mode
- **[NumPy](https://numpy.org/)** — Numerical processing

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m "Add amazing feature"`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[sb-ai-lab/EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib)** — Emotion recognition models & library
- **[AffectNet](http://mohammadmahoor.com/affectnet/)** — Facial expression dataset
- **[VGAF](https://github.com/ControlNet/VGAF)** — Video-level emotion labels

---

<div align="center">

**Made with ❤️ by [Kriskumar Gadara](https://github.com/Kris-gadara)**

⭐ **Star this repo** if you found it useful!

</div>
