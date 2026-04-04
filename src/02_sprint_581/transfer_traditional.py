"""
File: transfer_traditional.py
Author: Marco Wang
Date: 2026-04-03
Course: COLX 581 — Sprint 2
Description:
    Transfer learning for the TF-IDF + LinearSVC baseline via pre-trained
    multilingual fastText embeddings (frozen).

    Architecture
    ────────────
    Each email is represented as:
        [ TF-IDF features (sparse) | mean fastText embedding (dense, 300-d) ]

    The two feature sets are concatenated into a single matrix and fed
    to a LinearSVC, identical to the Sprint-1 baseline.

    Motivation
    ──────────
    TF-IDF counts lexical co-occurrences but has no semantic awareness —
    "phishing" and "fraud" are as unrelated as "phishing" and "apple".
    Adding mean fastText embeddings injects a *frozen* semantic signal:
      • OOV robustness: fastText uses subword n-grams, so it can produce
        useful vectors for misspelled words common in spam/phish emails.
      • Multilingual coverage: the CC-100 fastText vectors cover 157 languages
        and align English and Chinese words in the same 300-d space, which
        matches the bilingual nature of our dataset.
      • Why freeze? LinearSVC is not a gradient-based model — there is no
        back-propagation. The embeddings serve as fixed feature extractors.

    Pre-trained vectors used
    ────────────────────────
    We use the fastText common-crawl vectors available at:
        https://fasttext.cc/docs/en/crawl-vectors.html
    The script expects .vec (text-format) files, which are much lighter
    than the binary .bin models and require no C extension.
    Download once and set FASTTEXT_VEC_PATH accordingly (or let the script
    auto-download from the official S3 mirror via urllib).

    Supported languages: English (cc.en.300.vec) and/or Chinese (cc.zh.300.vec)
    If only one file is available, the script will fall back gracefully.
"""

import json
import os
import re
import sys
import urllib.request
import warnings

import jieba
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 581
EMBEDDING_DIM = 300
MAX_VOCAB_LOAD = 200_000   # cap on fastText vocab to keep RAM reasonable

LABEL_NAMES = ["Ham", "Phish", "Spam"]
LABEL2ID    = {"Ham": 0, "Phish": 1, "Spam": 2}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models", "transfer_traditional")

# Paths where pre-trained .vec files are expected.
# Users can override via env-vars: FASTTEXT_EN_VEC, FASTTEXT_ZH_VEC
FASTTEXT_EN_VEC = os.environ.get(
    "FASTTEXT_EN_VEC",
    os.path.join(PROJECT_ROOT, "models", "fasttext", "cc.en.300.vec"),
)
FASTTEXT_ZH_VEC = os.environ.get(
    "FASTTEXT_ZH_VEC",
    os.path.join(PROJECT_ROOT, "models", "fasttext", "cc.zh.300.vec"),
)

# Official fastText S3 mirror (gzipped .vec files)
FASTTEXT_EN_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
FASTTEXT_ZH_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz"

np.random.seed(SEED)


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_split(records: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """
    Return (en_texts, combined_texts, labels).
    combined_texts = English + Chinese text concatenated (for embedding lookup).
    """
    en_texts       = [r.get("text", "") for r in records]
    combined_texts = [
        f"{r.get('text', '')} {r.get('text_zh', '')}".strip()
        for r in records
    ]
    labels = [r["label"] for r in records]
    return en_texts, combined_texts, labels


# ── fastText loader ───────────────────────────────────────────────────────────
def _maybe_download_vec(url: str, dest_path: str) -> None:
    """Download + gunzip a .vec.gz fastText file if it doesn't exist."""
    import gzip
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    gz_path = dest_path + ".gz"
    print(f"  Downloading {url} …")
    urllib.request.urlretrieve(url, gz_path)
    print(f"  Decompressing → {dest_path} …")
    with gzip.open(gz_path, "rb") as f_in, open(dest_path, "wb") as f_out:
        f_out.write(f_in.read())
    os.remove(gz_path)


def load_fasttext_vec(
    vec_path: str,
    max_words: int = MAX_VOCAB_LOAD,
    url: str | None = None,
) -> dict[str, np.ndarray] | None:
    """
    Load a fastText .vec text file into a {word: vector} dict.

    Returns None if the file is missing and no URL is provided.
    Only the first `max_words` entries are loaded to cap memory usage.
    """
    if not os.path.isfile(vec_path):
        if url is None:
            print(f"  [WARN] fastText file not found: {vec_path}  (skipping)")
            return None
        try:
            _maybe_download_vec(url, vec_path)
        except Exception as exc:
            print(f"  [WARN] Auto-download failed ({exc}). Skipping {vec_path}.")
            return None

    print(f"  Loading fastText vectors from {vec_path} (up to {max_words:,} words) …")
    embeddings: dict[str, np.ndarray] = {}
    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        # First line is "<vocab_size> <dim>"
        header = f.readline().strip().split()
        vocab_total, dim = int(header[0]), int(header[1])
        print(f"    Vocab: {vocab_total:,}  |  Dim: {dim}")
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.rstrip().split(" ")
            word  = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == dim:
                    embeddings[word] = vec
            except ValueError:
                continue
    print(f"    Loaded {len(embeddings):,} word vectors.")
    return embeddings


def build_embedding_lookup(
    en_vecs: dict | None,
    zh_vecs: dict | None,
) -> dict[str, np.ndarray]:
    """
    Merge English and Chinese fastText dicts into one lookup.
    Chinese entries do NOT collide with English since they use different scripts.
    """
    merged: dict[str, np.ndarray] = {}
    if en_vecs:
        merged.update(en_vecs)
    if zh_vecs:
        merged.update(zh_vecs)
    print(f"  Combined embedding lookup: {len(merged):,} entries.")
    return merged


# ── Sentence-level embeddings ─────────────────────────────────────────────────
def bilingual_tokenize(text: str) -> list[str]:
    """
    Tokenise bilingual email text:
      - Chinese segments → jieba
      - English / other  → regex word tokens (lowercased)
    """
    tokens: list[str] = []
    if re.search(r"[\u4e00-\u9fff]", text):
        for seg in jieba.cut(text):
            seg = seg.strip()
            if seg:
                tokens.append(seg)
    else:
        tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def mean_embed(texts: list[str], lookup: dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute mean fastText embedding for each text.

    Strategy: tokenise → look up each token → average found vectors.
    Unknown tokens contribute a zero vector (i.e. they are ignored in the mean).
    If no token has a vector, return a zero vector.

    Returns shape (n_texts, EMBEDDING_DIM).
    """
    dim = EMBEDDING_DIM
    matrix = np.zeros((len(texts), dim), dtype=np.float32)

    for i, text in enumerate(texts):
        tokens = bilingual_tokenize(text)
        vecs = [lookup[t] for t in tokens if t in lookup]
        if vecs:
            matrix[i] = np.mean(vecs, axis=0)

    return matrix


# ── Feature construction ───────────────────────────────────────────────────────
def build_tfidf_pipeline() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=bilingual_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2,
    )


def build_feature_matrix(
    tfidf: TfidfVectorizer,
    texts_for_tfidf: list[str],
    texts_for_embed: list[str],
    lookup: dict,
    fit: bool = False,
) -> object:
    """
    Concatenate TF-IDF sparse matrix and mean fastText dense matrix.

      X = [ TF-IDF | mean_embed ]   shape: (n, vocab + 300)
    """
    if fit:
        X_tfidf = tfidf.fit_transform(texts_for_tfidf)
    else:
        X_tfidf = tfidf.transform(texts_for_tfidf)

    X_emb = mean_embed(texts_for_embed, lookup)
    X_combined = hstack([X_tfidf, csr_matrix(X_emb)])
    return X_combined


# ── Training & evaluation ──────────────────────────────────────────────────────
def train_and_evaluate(
    tfidf: TfidfVectorizer,
    lookup: dict,
    train_tfidf_texts: list[str],
    train_embed_texts: list[str],
    train_labels: list[str],
    eval_tfidf_texts: list[str],
    eval_embed_texts: list[str],
    eval_labels: list[str],
    split_name: str = "Validation",
) -> tuple[LinearSVC, np.ndarray]:
    """
    Fit a LinearSVC on combined features and evaluate on the given split.

    Returns the fitted classifier and the predicted label ids.
    """
    print(f"\nBuilding training features …")
    X_train = build_feature_matrix(
        tfidf, train_tfidf_texts, train_embed_texts, lookup, fit=True
    )
    print(f"  Training matrix shape: {X_train.shape}")

    clf = LinearSVC(random_state=SEED, class_weight="balanced", max_iter=2000)
    clf.fit(X_train, train_labels)

    print(f"Building {split_name} features …")
    X_eval = build_feature_matrix(
        tfidf, eval_tfidf_texts, eval_embed_texts, lookup, fit=False
    )

    preds = clf.predict(X_eval)
    print_results(split_name, eval_labels, preds)

    return clf, preds


def print_results(split_name: str, true_labels: list[str], pred_labels) -> None:
    acc = accuracy_score(true_labels, pred_labels)
    f1  = f1_score(true_labels, pred_labels, average="macro", labels=LABEL_NAMES)

    print(f"\n{'─'*60}")
    print(f"  {split_name} Results")
    print(f"{'─'*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro-F1  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=LABEL_NAMES, digits=4))

    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true_labels, pred_labels, labels=LABEL_NAMES)
    header = "          " + "  ".join(f"{n:>6}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<6}  " + "  ".join(f"{v:>6}" for v in row))


# ── Ablation: TF-IDF only baseline (for comparison) ───────────────────────────
def run_tfidf_only_baseline(
    train_tfidf_texts, train_labels,
    eval_tfidf_texts, eval_labels,
    split_name: str = "Validation",
) -> None:
    """Run the pure TF-IDF + LinearSVC without any embeddings (Sprint-1 baseline)."""
    print(f"\n{'='*60}")
    print("  ABLATION: TF-IDF only (reproduced Sprint-1 baseline)")
    print(f"{'='*60}")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=bilingual_tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2,
        )),
        ("clf", LinearSVC(random_state=SEED, class_weight="balanced", max_iter=2000)),
    ])
    pipeline.fit(train_tfidf_texts, train_labels)
    preds = pipeline.predict(eval_tfidf_texts)
    print_results(f"{split_name} (TF-IDF only)", eval_labels, preds)


# ── Save predictions ───────────────────────────────────────────────────────────
def save_predictions(
    records: list[dict],
    pred_labels,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record, pred in zip(records, pred_labels):
            row = {
                "text":            record.get("text", ""),
                "true_label":      record["label"],
                "predicted_label": pred,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print(" Transfer Learning | Traditional Baseline")
    print(" TF-IDF + fastText Embeddings + LinearSVC")
    print("=" * 60)

    # ── Load data ──
    print("\nLoading data splits …")
    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_records   = load_jsonl(os.path.join(DATA_DIR, "validation.jsonl"))
    test_records  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    train_en, train_combined, train_labels = prepare_split(train_records)
    val_en,   val_combined,   val_labels   = prepare_split(val_records)
    test_en,  test_combined,  test_labels  = prepare_split(test_records)

    print(f"  Train: {len(train_records)}  |  Val: {len(val_records)}  |  Test: {len(test_records)}")

    # ── Ablation: Sprint-1 baseline (no embeddings) ──
    run_tfidf_only_baseline(
        train_combined, train_labels,
        val_combined, val_labels,
        split_name="Validation",
    )

    # ── Load fastText vectors ──
    print("\n" + "=" * 60)
    print("  Loading pre-trained fastText vectors (frozen)")
    print("=" * 60)
    print("  Rationale: fastText CC vectors are trained on 157-language CommonCrawl")
    print("  (en + zh both available). Subword n-grams handle OOV phishing vocabulary.")
    print("  Vectors are FROZEN — LinearSVC has no gradient flow to update them.")

    en_vecs = load_fasttext_vec(FASTTEXT_EN_VEC, url=FASTTEXT_EN_URL)
    zh_vecs = load_fasttext_vec(FASTTEXT_ZH_VEC, url=FASTTEXT_ZH_URL)

    if en_vecs is None and zh_vecs is None:
        print("\n[ERROR] No fastText vectors could be loaded. "
              "Please set FASTTEXT_EN_VEC / FASTTEXT_ZH_VEC env-vars to point at "
              "pre-downloaded .vec files, or allow the script to auto-download.")
        sys.exit(1)

    lookup = build_embedding_lookup(en_vecs, zh_vecs)

    # ── Train + evaluate with embeddings ──
    print("\n" + "=" * 60)
    print("  TRANSFER MODEL: TF-IDF + fastText (concatenated features)")
    print("=" * 60)

    tfidf = build_tfidf_pipeline()

    clf, val_preds = train_and_evaluate(
        tfidf,
        lookup,
        train_tfidf_texts=train_combined,
        train_embed_texts=train_combined,
        train_labels=train_labels,
        eval_tfidf_texts=val_combined,
        eval_embed_texts=val_combined,
        eval_labels=val_labels,
        split_name="Validation",
    )

    # ── Final test evaluation ──
    print("\n" + "=" * 60)
    print("  TEST SET RESULTS (Transfer Model)")
    print("=" * 60)
    X_test = build_feature_matrix(
        tfidf, test_combined, test_combined, lookup, fit=False
    )
    test_preds = clf.predict(X_test)
    print_results("Test", test_labels, test_preds)

    # ── Save predictions ──
    save_predictions(
        test_records,
        test_preds,
        out_path=os.path.join(MODEL_DIR, "test_predictions.jsonl"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
