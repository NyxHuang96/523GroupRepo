import json
import re
import jieba
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report

# Set up with print outputs for terminal
print("Loading SpaCy NER model (en_core_web_sm)...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model not downloaded")
    exit()

# --- Data Loading & Tokenization ---
def load_jsonl(filepath):
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            texts.append(data.get("text", ""))
            labels.append(data.get("label", "Unknown").capitalize())
    return texts, labels

def bilingual_tokenizer(text):
    """Jieba for Chinese, Regex for English."""
    if re.search(r'[\u4e00-\u9fff]', text):
        return list(jieba.cut(text))
    else:
        return re.findall(r'\b\w+\b', text.lower())

# --- The MTL Feature Extractor (Auxiliary Task) ---
class NERDensityExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts the density of specific Named Entities to serve as 
    structural/semantic features alongside TF-IDF.
    """
    def __init__(self):
        # The entities most diagnostic of Phish vs. Spam
        self.target_ents = ['ORG', 'MONEY', 'DATE', 'PERSON', 'GPE']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            # We only pass the first 1000 chars to SpaCy to speed up processing
            doc = nlp(text[:1000]) 
            counts = {ent: 0 for ent in self.target_ents}
            
            for ent in doc.ents:
                if ent.label_ in counts:
                    counts[ent.label_] += 1
                    
            # Normalize by length to get density (prevents long docs from dominating)
            total_tokens = len(doc) if len(doc) > 0 else 1
            densities = [counts[ent] / total_tokens for ent in self.target_ents]
            
            # Add an overall "Entity Density" score
            total_ents = sum(counts.values())
            densities.append(total_ents / total_tokens)
            
            features.append(densities)
        return np.array(features)

# --- Main Execution ---
def main():
    print("\nLoading Sprint 2 Transfer Learning Data...")
    train_path = "data/processed/transfer_train.jsonl"
    dev_path = "data/processed/validation.jsonl"
    
    X_train, y_train = load_jsonl(train_path)
    X_dev, y_dev = load_jsonl(dev_path)
    
    # Add the Gold switch to the evaluation data (Consistency with Sprint 2)
    X_dev = [f"{text} _source_gold_" for text in X_dev]
    
    print(f"Training on {len(X_train)} samples, Evaluating on {len(X_dev)} samples.")

    # --- MODEL 1: STANDARD BASELINE ---
    print("\n--- Training Standard Baseline (TF-IDF Only) ---")
    baseline_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=bilingual_tokenizer, ngram_range=(1, 2))),
        ('clf', LinearSVC(class_weight='balanced', random_state=42))
    ])
    baseline_pipeline.fit(X_train, y_train)
    y_pred_base = baseline_pipeline.predict(X_dev)

    # --- MODEL 2: MTL PIPELINE ---
    print("--- Training MTL Pipeline (TF-IDF + Silver NER Density) ---")
    print("(Extracting NER features... this will take ~10-20 seconds)")
    
    mtl_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(tokenizer=bilingual_tokenizer, ngram_range=(1, 2))),
            ('ner', NERDensityExtractor()) # The Auxiliary Task!
        ])),
        ('clf', LinearSVC(class_weight='balanced', random_state=42, max_iter=2000))
    ])
    mtl_pipeline.fit(X_train, y_train)
    y_pred_mtl = mtl_pipeline.predict(X_dev)

    # --- Reporting ---
    print("\n" + "="*50)
    print(" RESULTS: BASELINE VS. MULTI-TASK LEARNING")
    print("="*50)
    
    acc_base = accuracy_score(y_dev, y_pred_base)
    acc_mtl = accuracy_score(y_dev, y_pred_mtl)
    
    print(f"{'Metric':<20} | {'Baseline (TF-IDF)':<20} | {'MTL (TF-IDF + NER)':<20}")
    print("-" * 65)
    print(f"{'Overall Accuracy':<20} | {acc_base:.4f}               | {acc_mtl:.4f}")
    
    print("\n=== Detailed MTL Classification Report ===")
    print(classification_report(y_dev, y_pred_mtl, target_names=["Ham", "Phish", "Spam"]))

if __name__ == "__main__":
    main()