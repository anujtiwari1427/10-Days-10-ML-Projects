import random
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
import tempfile
import os

from src.features import BLOOM_VERBS
from src.extractor import extract_questions_from_pdf

app = FastAPI(title="Adaptive Exam Question Classifier & Paper Generator")

# Load serialized pipeline on startup
MODEL_PATH = "models/difficulty_classifier.joblib"
classifier_pipeline = joblib.load(MODEL_PATH)

BLOOM_LABELS = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create"
}

def detect_bloom_level(text: str) -> str:
    words = set(text.lower().split())
    highest = 1
    for tier, verbs in BLOOM_VERBS.items():
        if words.intersection(verbs):
            highest = max(highest, tier)
    return BLOOM_LABELS[highest]

class PaperRequest(BaseModel):
    questions: List[str]
    total_questions: int = 10
    distribution: Dict[str, float] = {"Easy": 0.3, "Medium": 0.5, "Hard": 0.2}

@app.post("/generate-paper")
def generate_balanced_paper(req: PaperRequest):
    if abs(sum(req.distribution.values()) - 1.0) > 1e-5:
        raise HTTPException(status_code=400, detail="Distribution weights must sum to 1.0")

    if not req.questions:
        raise HTTPException(status_code=400, detail="Question bank cannot be empty")

    # Batch predict difficulty and confidence scores
    predictions = classifier_pipeline.predict(req.questions)
    probabilities = classifier_pipeline.predict_proba(req.questions)
    classes = classifier_pipeline.classes_

    tagged_bank = {"Easy": [], "Medium": [], "Hard": []}

    for idx, q_text in enumerate(req.questions):
        pred_label = predictions[idx]
        conf_score = float(max(probabilities[idx]))
        tagged_bank[pred_label].append({
            "question": q_text,
            "difficulty": pred_label,
            "confidence": round(conf_score, 3),
            "bloom_level": detect_bloom_level(q_text)
        })

    # Sample balanced paper according to target distribution
    selected_paper = []
    shortfalls = {}

    for diff_level, ratio in req.distribution.items():
        target_count = round(req.total_questions * ratio)
        available = tagged_bank.get(diff_level, [])
        if len(available) < target_count:
            shortfalls[diff_level] = f"Requested {target_count}, only {len(available)} available"
            selected_paper.extend(available)
        else:
            selected_paper.extend(random.sample(available, target_count))

    return {
        "total_selected": len(selected_paper),
        "shortfalls": shortfalls if shortfalls else None,
        "exam_paper": selected_paper
    }

@app.post("/parse-and-classify-pdf")
async def parse_and_classify_pdf(file: UploadFile = File(...)):
    """Upload a raw exam PDF, extract questions via PyMuPDF, and auto-tag difficulty + Bloom's."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        extracted = extract_questions_from_pdf(tmp_path)
        if not extracted:
            raise HTTPException(status_code=422, detail="No numbered questions detected in PDF.")

        texts = [item["text"] for item in extracted]
        preds = classifier_pipeline.predict(texts)

        for item, diff in zip(extracted, preds):
            item["difficulty"] = diff
            item["bloom_level"] = detect_bloom_level(item["text"])

        return {"count": len(extracted), "questions": extracted}
    finally:
        os.remove(tmp_path)