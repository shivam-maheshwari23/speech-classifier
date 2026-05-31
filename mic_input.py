import whisper
import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
import os
from classifier import load_model, build_label_embeddings, classify
from data import get_training_data


# Sample rate for recording
SAMPLE_RATE: int = 16000
# Duration to record in seconds
RECORD_SECONDS: int = 5


def record_audio(duration: int = RECORD_SECONDS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from microphone and return as numpy array."""
    print(f"🎤 Recording for {duration} seconds... speak now!")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()  # Wait until recording is finished
    print("Recording done!")
    return audio.flatten()


def transcribe_audio(audio: np.ndarray, whisper_model: whisper.Whisper) -> str:
    """Convert recorded audio to text using Whisper offline."""
    # Save audio to a temp file because Whisper reads from file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, audio, SAMPLE_RATE)

    # Transcribe using Whisper
    result = whisper_model.transcribe(tmp_path, language="en", fp16=False)
    text = result["text"].strip().lower()

    # Clean up temp file
    os.unlink(tmp_path)

    return text


def run_mic_pipeline() -> None:
    """Full offline pipeline: mic → whisper → classifier → command."""
    print("=== Offline Automotive Speech Command Classifier ===")
    print("Loading classifier model...")
    classifier_model = load_model()
    texts, labels = get_training_data()
    train_embeddings, train_labels = build_label_embeddings(
        classifier_model, texts, labels
    )

    print("Loading Whisper model (first time downloads ~150MB)...")
    whisper_model = whisper.load_model("base")
    print("✅ All models loaded! Fully offline from here.\n")

    print("Say a command after the prompt. Say 'quit' to exit.\n")

    while True:
        input("Press Enter when ready to speak...")

        # Step 1: Record from mic
        audio = record_audio()

        # Step 2: Speech to text using Whisper offline
        print("Transcribing...")
        spoken_text = transcribe_audio(audio, whisper_model)
        print(f"Whisper heard: '{spoken_text}'")

        if not spoken_text:
            print("Nothing detected, try again.\n")
            continue

        if "quit" in spoken_text:
            print("Goodbye!")
            break

        # Step 3: Classify the text
        predicted, score = classify(
            spoken_text, classifier_model, train_embeddings, train_labels
        )

        if predicted == "OUT_OF_SCOPE":
            print(f"Result: ❌ OUT OF SCOPE (score: {score:.2f})\n")
        else:
            print(f"Result: ✅ {predicted} (confidence: {score:.2f})\n")


if __name__ == "__main__":
    run_mic_pipeline()