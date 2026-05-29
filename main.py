import numpy as np
from sentence_transformers import SentenceTransformer
from classifier import load_model, build_label_embeddings, classify
from data import get_training_data

# Five example runs required by the assignment
EXAMPLE_RUNS: list[dict] = [
    {"input": "turn it down", "type": "clean"},
    {"input": "um pause music", "type": "noisy"},
    {"input": "what is the weather today", "type": "oos"},
    {"input": "increase the brightness", "type": "extension"},
    {"input": "start the vehicle", "type": "extension"},
]


def run_examples(
    model: SentenceTransformer,
    train_embeddings: np.ndarray,
    train_labels: list[str]
) -> None:
    """Run the 5 required example runs."""
    print("\n========== EXAMPLE RUNS ==========")
    for example in EXAMPLE_RUNS:
        text = example["input"]
        kind = example["type"]
        predicted, score = classify(text, model, train_embeddings, train_labels)
        if predicted == "OUT_OF_SCOPE":
            print(f"[{kind.upper()}] '{text}' → ❌ OUT_OF_SCOPE (score: {score:.2f})")
        else:
            print(f"[{kind.upper()}] '{text}' → ✅ {predicted} (score: {score:.2f})")


def run_interactive(
    model: SentenceTransformer,
    train_embeddings: np.ndarray,
    train_labels: list[str]
) -> None:
    """Interactive mode for testing."""
    print("\n========== INTERACTIVE MODE ==========")
    print("Type any command and press Enter. Type 'quit' to exit.\n")
    while True:
        user_input: str = input("You: ").strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if not user_input:
            continue
        predicted, score = classify(user_input, model,
                            train_embeddings, train_labels)
        if predicted == "OUT_OF_SCOPE":
            print(f"Result: ❌ OUT OF SCOPE (score: {score:.2f})\n")
        else:
            print(f"Result: ✅ {predicted} (confidence: {score:.2f})\n")


def main() -> None:
    """Full end to end pipeline."""
    print("=== Automotive Speech Command Classifier ===")
    print("14 commands | Offline | Edge Ready\n")

    model = load_model()
    texts, labels = get_training_data()
    train_embeddings, train_labels = build_label_embeddings(model, texts, labels)

    run_examples(model, train_embeddings, train_labels)
    run_interactive(model, train_embeddings, train_labels)


if __name__ == "__main__":
    main()
