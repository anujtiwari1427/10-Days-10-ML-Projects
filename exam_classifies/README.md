# Adaptive Exam Question Difficulty & Bloom's Classifier

An end-to-end NLP and Machine Learning pipeline that automatically extracts exam questions from raw PDFs, classifies their difficulty level (**Easy**, **Medium**, **Hard**), identifies their cognitive tier based on **Bloom's Taxonomy**, and generates balanced exam papers via a REST API—without relying on slow or costly LLM APIs.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [System Architecture & Tech Stack](#system-architecture--tech-stack)
3. [Project Structure](#project-structure)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
   - [1. Extracting Questions from PDFs](#1-extracting-questions-from-pdfs)
   - [2. Training the Classifier](#2-training-the-classifier)
   - [3. Running the FastAPI Server](#3-running-the-fastapi-server)
6. [API Reference](#api-reference)
7. [Feature Engineering Details](#feature-engineering-details)
8. [Future Enhancements](#future-enhancements)

---

## Problem Statement

Automated test-prep engines and ed-tech platforms need to ingest thousands of textbook and past board exam questions and tag them by difficulty (**Easy**, **Medium**, **Hard**) and cognitive complexity (**Bloom's Taxonomy**). Using Large Language Models (LLMs) for bulk tagging introduces high latency, recurring API costs, and rate limits. 

This project solves the problem locally using deterministic PDF layout parsing, custom psycholinguistic feature engineering, and a lightweight `scikit-learn` ensemble model served over FastAPI.

---

## System Architecture & Tech Stack

* **PDF Layout & Text Extraction:** `PyMuPDF` (`fitz`) — extracts text blocks with bounding-box coordinates to strip running headers/footers and capture section metadata (`Section A`, `Section B`, etc.).
* **Linguistic & NLP Processing:** `NLTK` (`punkt`, `cmudict`) — computes sentence boundaries, tokenization, and syllable counts via the CMU Pronouncing Dictionary.
* **Feature Engineering & ML Pipeline:** `scikit-learn` (`FeatureUnion`, `TfidfVectorizer`, `RandomForestClassifier`, `BaseEstimator`, `TransformerMixin`).
* **Model Serialization:** `joblib` — packages the custom linguistic transformer, TF-IDF vectorizer, and trained classifier into a single portable artifact (`difficulty_classifier.joblib`).
* **API Serving:** `FastAPI` & `Uvicorn` — exposes interactive Swagger documentation (`/docs`) and JSON endpoints for PDF parsing and balanced test generation.

---

## Project Structure

```text
exam_classifies/
├── data/
│   ├── sample_exam.pdf              # Sample board/textbook exam PDF for testing
│   └── labeled_questions.csv        # Training dataset ('question_text', 'difficulty')
├── models/
│   └── difficulty_classifier.joblib # Serialized scikit-learn pipeline
├── src/
│   ├── __init__.py
│   ├── extractor.py                 # PyMuPDF question & section parser
│   ├── features.py                  # Custom NLTK & Bloom's Taxonomy transformer
│   ├── train.py                     # FeatureUnion + RandomForest training script
│   └── api.py                       # FastAPI server & paper generator endpoints
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Clone or Navigate to the Project Directory
```bash
cd exam_classifies
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pymupdf nltk scikit-learn pandas joblib fastapi uvicorn python-multipart
```

### 4. Download Required NLTK Corpora
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('cmudict')"
```

---

## Usage Guide

### 1. Extracting Questions from PDFs
Place any past board exam or textbook PDF inside `data/sample_exam.pdf` and test the extractor standalone:
```bash
python -m src.extractor
```
This outputs structured JSON objects containing the question number, section header, page number, and cleaned question text.

### 2. Training the Classifier
Ensure `data/labeled_questions.csv` contains two columns: `question_text` and `difficulty` (`Easy`, `Medium`, or `Hard`). Run the training module from the `exam_classifies` root directory:
```bash
python -m src.train
```
This script splits the data (80/20 stratified), fits the `FeatureUnion` + `RandomForestClassifier` pipeline, prints a classification report (Precision, Recall, F1-score), and saves the serialized pipeline to `models/difficulty_classifier.joblib`.

### 3. Running the FastAPI Server
Start the Uvicorn ASGI server:
```bash
uvicorn src.api:app --reload --port 8000
```
Open your browser and navigate to **`http://127.0.0.1:8000/docs`** to access the interactive Swagger UI.

---

## API Reference

### 1. `POST /parse-and-classify-pdf`
Uploads a raw exam PDF, extracts all numbered questions using PyMuPDF, and returns each question tagged with its predicted difficulty and Bloom's cognitive level.

* **Request:** `multipart/form-data` with a `.pdf` file upload.
* **Sample Response:**
```json
{
  "count": 2,
  "questions": [
    {
      "q_num": "1",
      "section": "SECTION A",
      "page": 1,
      "text": "State Ohm's law and write its mathematical formula.",
      "difficulty": "Easy",
      "bloom_level": "Remember"
    },
    {
      "q_num": "14",
      "section": "SECTION C",
      "page": 3,
      "text": "Derive the mirror formula 1/f = 1/v + 1/u for a spherical concave mirror.",
      "difficulty": "Hard",
      "bloom_level": "Analyze"
    }
  ]
}
```

### 2. `POST /generate-paper`
Accepts a raw pool of questions, classifies them in batch, and samples a balanced exam paper matching the requested difficulty ratio.

* **Sample Request Body:**
```json
{
  "questions": [
    "Define photosynthesis and name the pigment responsible for it.",
    "State Newton's first law of motion.",
    "Explain why the sky appears blue during the daytime.",
    "Calculate the equivalent resistance when two 6 ohm resistors are connected in parallel.",
    "Derive the mirror formula 1/f = 1/v + 1/u for a spherical concave mirror."
  ],
  "total_questions": 4,
  "distribution": {
    "Easy": 0.5,
    "Medium": 0.25,
    "Hard": 0.25
  }
}
```

* **Sample Response:**
```json
{
  "total_selected": 4,
  "shortfalls": null,
  "exam_paper": [
    {
      "question": "Define photosynthesis and name the pigment responsible for it.",
      "difficulty": "Easy",
      "confidence": 0.945,
      "bloom_level": "Remember"
    },
    {
      "question": "State Newton's first law of motion.",
      "difficulty": "Easy",
      "confidence": 0.91,
      "bloom_level": "Remember"
    },
    {
      "question": "Explain why the sky appears blue during the daytime.",
      "difficulty": "Medium",
      "confidence": 0.865,
      "bloom_level": "Understand"
    },
    {
      "question": "Derive the mirror formula 1/f = 1/v + 1/u for a spherical concave mirror.",
      "difficulty": "Hard",
      "confidence": 0.93,
      "bloom_level": "Analyze"
    }
  ]
}
```

---

## Feature Engineering Details

The `FeatureUnion` in `src/train.py` merges two distinct feature spaces before classification:

1. **Lexical N-Gram Features (`TfidfVectorizer`):**
   - Captures unigrams and bigrams (`ngram_range=(1, 2)`) with sublinear term-frequency scaling across up to 3,000 vocabulary terms.
2. **Psycholinguistic & Cognitive Features (`LinguisticBloomExtractor`):**
   - **Average Sentence Length:** Words per sentence (`num_words / num_sentences`).
   - **Syllable Density:** Average syllables per word using `cmudict` with a regex vowel-cluster fallback for domain-specific terms.
   - **Complex Word Ratio:** Proportion of polysyllabic words ($\ge 3$ syllables).
   - **Bloom's Taxonomy Action Verb Densities:** Normalized frequencies across all 6 cognitive tiers (*1: Remember, 2: Understand, 3: Apply, 4: Analyze, 5: Evaluate, 6: Create*), plus the normalized highest cognitive tier triggered.
   - **Structural Cues:** Binary indicator for multi-part sub-questions (`(i)`, `(ii)`, `(a)`) and count of mathematical/scientific symbols (`=`, `^`, `∫`, `∑`, `√`).

---

## Future Enhancements

* **Multi-Output Classification:** Upgrade `RandomForestClassifier` with `MultiOutputClassifier` to predict both Difficulty and Bloom's Taxonomy tier via supervised learning rather than rule-based verb matching.
* **Section-Aware Marks Allocation:** Automatically group generated exam papers into board-style sections (e.g., 1-mark MCQs, 3-mark short answers, 5-mark long answers).
* **OCR Fallback:** Integrate `Tesseract` or `RapidOCR` in `extractor.py` for scanned image-only exam PDFs.