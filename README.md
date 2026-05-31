# Automotive Speech Command Classifier

A lightweight, fully offline semantic command classifier for automotive voice control systems.

## What it does
Classifies spoken commands into 14 predefined automotive commands and rejects out-of-scope inputs. Runs fully offline on-device — no internet required.

## Full Pipeline
Microphone → Whisper (offline STT) → Semantic Classifier → Command Output

## Commands Supported
**Core (10):** activate/deactivate do not disturb, decline/pick up call, play/pause music, next/previous song, increase/decrease volume

**Extension (4):** increase/decrease brightness, start/stop vehicle

## Results
| Metric | Score |
|--------|-------|
| Clean input accuracy | 100% (12/12) |
| Noisy input accuracy | 100% (8/8) |
| OOS rejection rate | 100% (7/7) |
| Model size (ONNX) | 0.67 MB (limit: 25 MB) |
| Inference latency | 7.61 ms (limit: 1000 ms) |

## Setup
```bash
pip install -r requirements.txt
```

## How to Run

### Option 1 — Web Dashboard (recommended)
```bash
python app.py
```
Then open browser at: http://localhost:5000

### Option 2 — Mic Input (terminal)
```bash
python mic_input.py
```

### Option 3 — Text Input (terminal)
```bash
python main.py
```

### Option 4 — Evaluation
```bash
python evaluate.py
```

### Option 5 — ONNX Export
```bash
python export.py
```

## Files
| File | Purpose |
|------|---------|
| `data.py` | 14 commands with training phrases |
| `classifier.py` | Embedding and classification logic |
| `evaluate.py` | Full evaluation with metrics and confusion matrix |
| `export.py` | ONNX export and latency benchmark |
| `main.py` | End-to-end pipeline entry point |
| `mic_input.py` | Offline mic input using Whisper STT |
| `app.py` | Flask web dashboard |
| `templates/index.html` | Dashboard UI |

## Architecture
Text Input → SentenceTransformer (all-MiniLM-L6-v2) → Cosine Similarity → Threshold (0.58) → Predicted Command or OUT_OF_SCOPE

## Key Design Decisions
- **Semantic embeddings** over keyword matching — handles paraphrases naturally
- **Cosine similarity threshold** for OOS rejection — no wrong guesses
- **Embedding-based architecture** — new commands added without retraining
- **Whisper** for fully offline speech-to-text

## Tools Used
Python 3.14, sentence-transformers, scikit-learn, ONNX, PyTorch, OpenAI Whisper, Flask, sounddevice, numpy, pandas, matplotlib, seaborn

## AI Assistance
Built with help from Claude AI as permitted by the problem statement. All code tested and fully understood by the author.