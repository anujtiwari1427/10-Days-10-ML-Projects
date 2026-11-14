import pandas as pd
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.features import LinguisticBloomExtractor

def train_and_serialize(data_path: str, model_output_path: str):
    # Expected CSV columns: 'question_text', 'difficulty' (Easy, Medium, Hard)
    df = pd.read_csv(data_path)
    X = df["question_text"].fillna("")
    y = df["difficulty"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Combine lexical TF-IDF features with custom linguistic + Bloom's features
    combined_features = FeatureUnion([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            stop_words="english",
            sublinear_tf=True
        )),
        ("linguistic_bloom", LinguisticBloomExtractor())
    ])

    pipeline = Pipeline([
        ("features", combined_features),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate performance
    y_pred = pipeline.predict(X_test)
    print("Model Evaluation Report:\n")
    print(classification_report(y_test, y_pred))

    # Serialize the complete end-to-end pipeline
    joblib.dump(pipeline, model_output_path)
    print(f"Pipeline successfully saved to {model_output_path}")

if __name__ == "__main__":
    train_and_serialize("data/labeled_questions.csv", "models/difficulty_classifier.joblib")