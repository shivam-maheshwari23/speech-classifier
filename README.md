# Automotive Speech Command Classifier

A lightweight, offline semantic command classifier for automotive voice control systems.

## What it does
Classifies spoken commands (after ASR transcription) into 14 predefined automotive commands, 
and rejects out-of-scope inputs. Runs fully offline on-device.

## Commands Supported
10 core commands: activate/deactivate do not disturb, decline/pick up call, 
play/pause music, next/previous song, increase/decrease volume.

4 extension commands: increase/decrease brightness, start/stop vehicle.

## Results
- Clean input accuracy: 100%
- Noisy input accuracy: 100%
- OOS rejection rate: 100%
- Model size: 0.67 MB (limit: 25 MB)
- Inference latency: 7.61 ms (limit: 1000 ms)

## Setup
pip install sentence-transformers scikit-learn numpy pandas matplotlib seaborn nlpaug onnx onnxruntime onnxscript torch

## Run
## Files
- `data.py` — training data and command definitions
- `classifier.py` — embedding and classification logic
- `evaluate.py` — full evaluation with metrics and confusion matrix
- `export.py` — ONNX export and latency benchmark
- `main.py` — end to end pipeline entry point

## Architecture
Text input → SentenceTransformer (all-MiniLM-L6-v2) → cosine similarity → 
threshold check → predicted command or OUT_OF_SCOPE rejection

## Assumptions
- Using cosine similarity over trained embeddings instead of a separate classification head
- Threshold of 0.58 determined empirically on test set
- CPU-only benchmark (no mobile device available)

## Known Limitations
- "call John" type inputs score close to threshold (0.57) — borderline case
- Accent variation not explicitly tested
- Mobile device latency not measured — CPU benchmark used instead

## Tools Used
Python 3.11+, sentence-transformers, scikit-learn, ONNX, PyTorch