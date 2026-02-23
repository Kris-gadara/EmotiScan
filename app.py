"""
EmotiScan - Web-based Face Detection & Emotion Recognition
Owner: Kriskumar Gadara
"""

import os
import sys

import cv2
import gradio as gr
import numpy as np

# Add EmotiEffLib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "EmotiEffLib"))

from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

# ── Face detector (OpenCV Haar Cascade) ──────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ── Emotion recognizer (ONNX - no torch needed) ─────────────────────────────
MODEL_NAME = "enet_b0_8_best_vgaf"
print(f"Loading EmotiScan model: {MODEL_NAME} ...")
emotion_model = EmotiEffLibRecognizer(engine="onnx", model_name=MODEL_NAME)
print("Model loaded successfully!")

# Emotion → colour map
EMOTION_COLORS = {
    "Anger": (0, 0, 255),
    "Contempt": (128, 0, 128),
    "Disgust": (0, 128, 0),
    "Fear": (255, 165, 0),
    "Happiness": (0, 255, 0),
    "Neutral": (200, 200, 200),
    "Sadness": (255, 0, 0),
    "Surprise": (0, 255, 255),
}


def detect_and_recognize(image):
    """Detect faces, classify emotions, and annotate the image."""
    if image is None:
        return None, "No image provided."

    img = np.array(image)
    if img.ndim == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    result_text_parts = []
    annotated = img.copy()

    if len(faces) == 0:
        result_text_parts.append("No faces detected.")
    else:
        for i, (x, y, w, h) in enumerate(faces, 1):
            # Extract face ROI
            face_roi = img[y : y + h, x : x + w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Predict emotion
            try:
                emotions, scores = emotion_model.predict_emotions(face_rgb, logits=False)
                emotion = emotions[0]
                confidence = float(np.max(scores[0]))

                # Build per-emotion score string
                class_names = list(emotion_model.idx_to_emotion_class.values())
                score_lines = "  ".join(
                    f"{name}: {scores[0][j]:.1%}" for j, name in enumerate(class_names)
                )

                result_text_parts.append(
                    f"Face {i}: **{emotion}** ({confidence:.1%})\n{score_lines}"
                )
            except Exception as e:
                emotion = "Error"
                confidence = 0.0
                result_text_parts.append(f"Face {i}: Error - {e}")

            # Draw bounding box + label
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"{emotion} {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x, y - th - 10), (x + tw, y), color, -1)
            cv2.putText(
                annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
            )

    result_text = "\n\n".join(result_text_parts)
    # Convert back to RGB for Gradio display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, result_text


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="EmotiScan - Face Emotion Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎭 EmotiScan — Face Detection & Emotion Recognition
        **Owner:** Kriskumar Gadara

        Upload an image or use your webcam to detect faces and recognize emotions in real time.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", sources=["upload", "webcam"])
            run_btn = gr.Button("🔍 Detect & Recognize", variant="primary", size="lg")
        with gr.Column(scale=1):
            output_image = gr.Image(label="Detected Faces & Emotions")
            output_text = gr.Markdown(label="Results")

    run_btn.click(fn=detect_and_recognize, inputs=input_image, outputs=[output_image, output_text])
    input_image.change(
        fn=detect_and_recognize, inputs=input_image, outputs=[output_image, output_text]
    )

    gr.Markdown(
        """
        ---
        **Models:** EmotiEffLib (ONNX) — EfficientNet-B0 trained on AffectNet + VGAF  
        **Emotions:** Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise  
        **Project:** [EmotiScan on GitHub](https://github.com/Kris-gadara/EmotiScan)
        """
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
