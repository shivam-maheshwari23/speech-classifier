from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from classifier import load_model, build_label_embeddings, classify
from data import get_training_data, get_all_labels

# Test set - clean inputs the model has NOT seen during training
CLEAN_TEST_DATA: list[tuple[str, str]] = [
    ("turn it down", "decrease_volume"),
    ("make it louder", "increase_volume"),
    ("next track please", "play_next_song"),
    ("go back a track", "play_previous_song"),
    ("answer the phone", "pick_up_call"),
    ("reject the call", "decline_call"),
    ("start playing music", "play_music"),
    ("hold the music", "pause_music"),
    ("enable do not disturb", "activate_do_not_disturb"),
    ("turn off do not disturb", "deactivate_do_not_disturb"),
    ("volume up please", "increase_volume"),
    ("skip this track", "play_next_song"),
]

# Noisy inputs - simulating real ASR output
NOISY_TEST_DATA: list[tuple[str, str]] = [
    ("um turn it down", "decrease_volume"),
    ("uh next song", "play_next_song"),
    ("pause uh the music", "pause_music"),
    ("like answer the call", "pick_up_call"),
    ("volume up uh", "increase_volume"),
    ("um decline call", "decline_call"),
    ("play music please uh", "play_music"),
    ("previous song um", "play_previous_song"),
]

# Out of scope inputs - must all be rejected
OOS_TEST_DATA: list[str] = [
    "what is the weather today",
    "navigate to the nearest petrol station",
    "call John",
    "hello",
    "turn off the air conditioning",
    "what time is it",
    "open google maps",
]


def evaluate_clean(
    model: SentenceTransformer,
    train_embeddings: np.ndarray,
    train_labels: list[str]
) -> None:
    """Evaluate on clean test inputs and print classification report."""
    print("\n========== CLEAN INPUT EVALUATION ==========")
    true_labels: list[str] = []
    predicted_labels: list[str] = []

    for text, true_label in CLEAN_TEST_DATA:
        predicted, score = classify(text, model, train_embeddings, train_labels)
        true_labels.append(true_label)
        predicted_labels.append(predicted)
        status = "✅" if predicted == true_label else "❌"
        print(f"{status} '{text}' → {predicted} (score: {score:.2f})")

    print("\n--- Classification Report ---")
    print(classification_report(true_labels, predicted_labels))

    # Confusion matrix
    all_labels = get_all_labels()
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=all_labels,
                yticklabels=all_labels, cmap="Blues")
    plt.title("Confusion Matrix - Clean Inputs")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix_clean.png")
    print("Confusion matrix saved as confusion_matrix_clean.png")


def evaluate_noisy(
    model: SentenceTransformer,
    train_embeddings: np.ndarray,
    train_labels: list[str]
) -> None:
    """Evaluate on noisy ASR-like inputs."""
    print("\n========== NOISY INPUT EVALUATION ==========")
    correct: int = 0
    total: int = len(NOISY_TEST_DATA)

    for text, true_label in NOISY_TEST_DATA:
        predicted, score = classify(text, model, train_embeddings, train_labels)
        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"{status} '{text}' → {predicted} (score: {score:.2f})")

    accuracy = correct / total * 100
    print(f"\nNoisy Input Accuracy: {correct}/{total} = {accuracy:.1f}%")


def evaluate_oos(
    model: SentenceTransformer,
    train_embeddings: np.ndarray,
    train_labels: list[str]
) -> None:
    """Evaluate out-of-scope rejection."""
    print("\n========== OUT OF SCOPE EVALUATION ==========")
    correct_rejections: int = 0
    total: int = len(OOS_TEST_DATA)

    for text in OOS_TEST_DATA:
        predicted, score = classify(text, model, train_embeddings, train_labels)
        is_rejected = predicted == "OUT_OF_SCOPE"
        if is_rejected:
            correct_rejections += 1
        status = "✅" if is_rejected else "❌"
        print(f"{status} '{text}' → {predicted} (score: {score:.2f})")

    rejection_rate = correct_rejections / total * 100
    print(f"\nOOS Rejection Rate: {correct_rejections}/{total} = {rejection_rate:.1f}%")


def run_evaluation() -> None:
    """Run all evaluations."""
    model = load_model()
    texts, labels = get_training_data()
    train_embeddings, train_labels = build_label_embeddings(model, texts, labels)

    evaluate_clean(model, train_embeddings, train_labels)
    evaluate_noisy(model, train_embeddings, train_labels)
    evaluate_oos(model, train_embeddings, train_labels)


if __name__ == "__main__":
    run_evaluation()