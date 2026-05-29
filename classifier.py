from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data import get_training_data, get_all_labels, OUT_OF_SCOPE

# Load a lightweight model that works offline after first download
MODEL_NAME: str = "all-MiniLM-L6-v2"

def load_model() -> SentenceTransformer:
    """Load the sentence transformer model."""
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded!")
    return model


def build_label_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    labels: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Create embeddings for all training phrases."""
    print("Building embeddings for training data...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, labels


def classify(
    input_text: str,
    model: SentenceTransformer,
    train_embeddings: np.ndarray,
    train_labels: list[str],
    threshold: float = 0.58
) -> tuple[str, float]:
    """
    Classify input text into a command or reject as out-of-scope.
    Returns the predicted label and confidence score.
    """
    # Convert input text to embedding
    input_embedding = model.encode([input_text])

    # Compare with all training embeddings
    similarities = cosine_similarity(input_embedding, train_embeddings)[0]

    # Get the best match
    best_index: int = int(np.argmax(similarities))
    best_score: float = float(similarities[best_index])
    best_label: str = train_labels[best_index]

    # If score is too low, reject as out of scope
    if best_score < threshold:
        return "OUT_OF_SCOPE", best_score

    return best_label, best_score


def run_pipeline() -> None:
    """Main pipeline: load model, build embeddings, classify user input."""
    # Setup
    model = load_model()
    texts, labels = get_training_data()
    train_embeddings, train_labels = build_label_embeddings(model, texts, labels)

    print("\n--- Speech Command Classifier Ready ---")
    print("Type a command and press Enter. Type 'quit' to exit.\n")

    while True:
        user_input: str = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        predicted_label, score = classify(
            user_input,
            model,
            train_embeddings,
            train_labels
        )

        if predicted_label == "OUT_OF_SCOPE":
            print(f"Result: ❌ OUT OF SCOPE (score: {score:.2f})\n")
        else:
            print(f"Result: ✅ {predicted_label} (confidence: {score:.2f})\n")


if __name__ == "__main__":
    run_pipeline()