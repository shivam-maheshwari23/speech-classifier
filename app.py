from flask import Flask, render_template, jsonify, request
from classifier import load_model, build_label_embeddings, classify
from data import get_training_data
import numpy as np
from sentence_transformers import SentenceTransformer

app: Flask = Flask(__name__)

# Load models once when server starts
print("Loading models...")
model = load_model()
texts, labels = get_training_data()
train_embeddings, train_labels = build_label_embeddings(model, texts, labels)
print("Models ready!")

# Store history of classifications
history: list[dict] = []


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify_text():
    """Classify text input and return result."""
    data = request.get_json()
    text: str = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    predicted, score = classify(text, model, train_embeddings, train_labels)

    result = {
        "input": text,
        "predicted": predicted,
        "score": round(float(score), 2),
        "is_oos": predicted == "OUT_OF_SCOPE"
    }

    history.append(result)
    return jsonify(result)


@app.route("/classify_mic", methods=["POST"])
def classify_mic():
    """Record from mic, transcribe with Whisper, classify."""
    import whisper
    import sounddevice as sd
    import soundfile as sf
    import tempfile
    import os

    SAMPLE_RATE: int = 16000
    DURATION: int = 5

    try:
        # Record audio
        print("Recording from mic...")
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        audio = audio.flatten()
        print("Recording done!")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio, SAMPLE_RATE)

        # Load whisper and transcribe
        print("Transcribing...")
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(tmp_path, language="en", fp16=False)
        spoken_text = result["text"].strip().lower()
        os.unlink(tmp_path)
        print(f"Whisper heard: {spoken_text}")

        if not spoken_text:
            return jsonify({"error": "Nothing detected"}), 400

        # Classify
        predicted, score = classify(
            spoken_text, model, train_embeddings, train_labels
        )

        result = {
            "input": spoken_text,
            "predicted": predicted,
            "score": round(float(score), 2),
            "is_oos": predicted == "OUT_OF_SCOPE"
        }
        history.append(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def get_history():
    """Return classification history."""
    return jsonify(history[-20:])


@app.route("/stats")
def get_stats():
    """Return statistics about classifications."""
    if not history:
        return jsonify({
            "total": 0,
            "successful": 0,
            "rejected": 0,
            "avg_confidence": 0
        })

    total = len(history)
    successful = sum(1 for h in history if not h["is_oos"])
    rejected = sum(1 for h in history if h["is_oos"])
    avg_confidence = round(
        sum(h["score"] for h in history if not h["is_oos"]) / max(successful, 1), 2
    )

    return jsonify({
        "total": total,
        "successful": successful,
        "rejected": rejected,
        "avg_confidence": avg_confidence
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)