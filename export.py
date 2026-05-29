import numpy as np
import onnxruntime as ort
import time
import os
import torch
from sentence_transformers import SentenceTransformer
from classifier import load_model, build_label_embeddings, classify
from data import get_training_data


def export_to_onnx(model: SentenceTransformer, output_path: str = "classifier_model.onnx") -> None:
    """Export the sentence transformer to ONNX format."""
    print("Exporting model to ONNX...")
    transformer = model[0].auto_model
    tokenizer = model[0].tokenizer
    dummy_text = "increase the volume"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    torch.onnx.export(
        transformer,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        verbose=False
    )
    print(f"Model exported to {output_path}")


def check_model_size(path: str) -> float:
    """Return model size in MB."""
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def benchmark_inference(
    model: SentenceTransformer,
    train_embeddings: np.ndarray,
    train_labels: list[str],
    runs: int = 100
) -> float:
    """Measure average inference latency in milliseconds."""
    test_input = "increase the volume"
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        classify(test_input, model, train_embeddings, train_labels)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    avg_ms = float(np.mean(times))
    return avg_ms


def run_export() -> None:
    """Main export and benchmark pipeline."""
    model = load_model()
    texts, labels = get_training_data()
    train_embeddings, train_labels = build_label_embeddings(model, texts, labels)

    onnx_path = "classifier_model.onnx"
    export_to_onnx(model, onnx_path)

    size_mb = check_model_size(onnx_path)
    print(f"\n--- Model Size Report ---")
    print(f"ONNX model size: {size_mb:.2f} MB")
    if size_mb <= 25:
        print(f"✅ Within 25 MB limit")
    else:
        print(f"❌ Exceeds 25 MB limit")

    print(f"\n--- Latency Benchmark (100 runs) ---")
    avg_latency = benchmark_inference(model, train_embeddings, train_labels)
    print(f"Average inference latency: {avg_latency:.2f} ms")
    if avg_latency < 1000:
        print(f"✅ Within 1 second limit")
    else:
        print(f"❌ Exceeds 1 second limit")

    print(f"\n--- Summary ---")
    print(f"Model size: {size_mb:.2f} MB {'✅' if size_mb <= 25 else '❌'}")
    print(f"Latency: {avg_latency:.2f} ms {'✅' if avg_latency < 1000 else '❌'}")


if __name__ == "__main__":
    run_export()