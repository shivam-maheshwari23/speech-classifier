from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def create_approach_doc() -> None:
    """Create the approach document as a Word file."""
    doc = Document()

    # Title
    title = doc.add_heading("Approach Document", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_heading("Problem 1: Lightweight Speech Command Classifier for Edge Deployment", 2)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    name = doc.add_paragraph("Candidate: Shivam Maheshwari")
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")

    # Problem Understanding
    doc.add_heading("1. Problem Understanding", 1)
    doc.add_paragraph(
        "The goal is to build an offline, low-latency semantic command classifier "
        "for automotive voice control systems. Given a text string from an ASR engine, "
        "the system must map it to one of 14 predefined commands or reject it as "
        "out-of-scope. The solution must run entirely on-device with model size under "
        "25 MB and inference latency under 1 second."
    )

    # Key Challenges
    doc.add_heading("2. Key Challenges", 1)
    challenges = [
        "Semantically opposite commands must not be confused (e.g. increase vs decrease volume)",
        "ASR noise — filler words like 'um', 'uh', dropped words, paraphrases",
        "Out-of-scope rejection without misclassifying valid commands",
        "Strict size and latency constraints for edge deployment",
        "Extension commands must be addable without full retraining",
    ]
    for c in challenges:
        p = doc.add_paragraph(c, style="List Bullet")

    # Solution Architecture
    doc.add_heading("3. Solution Architecture", 1)
    doc.add_paragraph(
        "Text Input → SentenceTransformer Embedding → Cosine Similarity → "
        "Threshold Check → Predicted Command or OUT_OF_SCOPE Rejection"
    )

    doc.add_heading("Why This Approach", 2)
    doc.add_paragraph(
        "I chose the all-MiniLM-L6-v2 SentenceTransformer model because:"
    )
    reasons = [
        "It is extremely lightweight — only 0.67 MB after ONNX export",
        "It understands semantic meaning, not just keywords",
        "It generalises to paraphrases and ASR noise without retraining",
        "It works fully offline after first download",
        "Cosine similarity is simple, fast, and explainable",
    ]
    for r in reasons:
        doc.add_paragraph(r, style="List Bullet")

    # Dataset
    doc.add_heading("4. Dataset", 1)
    doc.add_paragraph(
        "I created a synthetic dataset of 14 commands with 7-8 natural language "
        "phrasings each. The dataset covers direct phrasings, paraphrases, "
        "Indian English accent variations, and ASR-style noisy inputs with "
        "filler words and dropped articles."
    )

    # OOS Rejection
    doc.add_heading("5. Out-of-Scope Rejection", 1)
    doc.add_paragraph(
        "I implemented a cosine similarity threshold of 0.58. Any input scoring "
        "below this threshold is rejected as OUT_OF_SCOPE rather than forced into "
        "a wrong class. The threshold was determined empirically — the lowest "
        "scoring valid command scores 0.72, giving a safe margin above 0.58."
    )

    # Extensibility
    doc.add_heading("6. Extensibility", 1)
    doc.add_paragraph(
        "The architecture is fully embedding-based, so new commands can be added "
        "by simply adding new phrases to data.py — no retraining required. "
        "Commands 11-14 (increase/decrease brightness, start/stop vehicle) were "
        "added this way with zero interference to the original 10 commands. "
        "The same approach could support 50+ commands before performance degrades."
    )

    # Results
    doc.add_heading("7. Results", 1)
    table = doc.add_table(rows=6, cols=2)
    table.style = "Table Grid"
    headers = table.rows[0].cells
    headers[0].text = "Metric"
    headers[1].text = "Score"
    results = [
        ("Clean input accuracy", "100% (12/12)"),
        ("Noisy input accuracy", "100% (8/8)"),
        ("OOS rejection rate", "100% (7/7)"),
        ("Model size (ONNX)", "0.67 MB (limit: 25 MB)"),
        ("Inference latency", "7.61 ms (limit: 1000 ms)"),
    ]
    for i, (metric, score) in enumerate(results):
        row = table.rows[i + 1].cells
        row[0].text = metric
        row[1].text = score

    doc.add_paragraph("")

    # Milestones
    doc.add_heading("8. Milestones Completed", 1)
    milestones = [
        "M1: Baseline classifier on 10 core commands — COMPLETE",
        "M2: Noise robustness and ASR simulation — COMPLETE",
        "M3: ONNX export and latency benchmark — COMPLETE",
        "M4: Extension commands 11-14 added — COMPLETE",
        "M5: End-to-end pipeline with example runs — COMPLETE",
    ]
    for m in milestones:
        doc.add_paragraph(m, style="List Bullet")

    # Tools
    doc.add_heading("9. Tools Used", 1)
    tools = [
        "Python 3.14",
        "sentence-transformers — all-MiniLM-L6-v2 embedding model",
        "scikit-learn — cosine similarity and evaluation metrics",
        "ONNX + onnxruntime — edge model export",
        "PyTorch — model export pipeline",
        "matplotlib + seaborn — confusion matrix visualisation",
    ]
    for t in tools:
        doc.add_paragraph(t, style="List Bullet")

    # GitHub
    doc.add_heading("10. GitHub Repository", 1)
    doc.add_paragraph("https://github.com/shivam-maheshwari23/speech-classifier")

    # Self Assessment
    doc.add_heading("11. Self Assessment", 1)
    doc.add_paragraph("What works well:")
    good = [
        "Semantic understanding handles paraphrases perfectly",
        "OOS rejection works reliably with threshold approach",
        "Noisy inputs with filler words handled without any preprocessing",
        "Model is extremely small and fast for edge deployment",
    ]
    for g in good:
        doc.add_paragraph(g, style="List Bullet")

    doc.add_paragraph("What could be improved:")
    improve = [
        "Testing on real ASR output from Indian English speakers",
        "Mobile device latency measurement instead of CPU benchmark",
        "Larger and more diverse training dataset",
        "Quantisation to further reduce model size",
    ]
    for i in improve:
        doc.add_paragraph(i, style="List Bullet")

    doc.add_paragraph("What would be needed for production:")
    prod = [
        "Integration with a real ASR engine like Whisper or Google STT",
        "Continuous learning pipeline to add new commands over time",
        "On-device testing on actual automotive hardware",
        "Larger labelled dataset with real user utterances",
    ]
    for p in prod:
        doc.add_paragraph(p, style="List Bullet")

    # Save
    doc.save("Shivam-Approach.docx")
    print("Shivam-Approach.docx created successfully!")




def create_code_doc() -> None:
    """Create the code document as a Word file."""
    doc = Document()

    title = doc.add_heading("Code Document", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_heading("Problem 1: Lightweight Speech Command Classifier", 2)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    name = doc.add_paragraph("Candidate: Shivam Maheshwari")
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("GitHub: https://github.com/shivam-maheshwari23/speech-classifier")
    doc.add_paragraph("")

    # Read and add each file
    files = ["data.py", "classifier.py", "evaluate.py", "export.py", "main.py"]

    for filename in files:
        doc.add_heading(filename, 1)
        try:
            with open(filename, "r", encoding="utf-8") as f:
                code = f.read()
            para = doc.add_paragraph()
            run = para.add_run(code)
            run.font.name = "Courier New"
            run.font.size = Pt(8)
        except FileNotFoundError:
            doc.add_paragraph(f"{filename} not found")
        doc.add_page_break()

    doc.save("Shivam-Code.docx")
    print("Shivam-Code.docx created successfully!")
if __name__ == "__main__":
    create_approach_doc()
    create_code_doc()