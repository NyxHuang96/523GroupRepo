"""
File: transfer_neural.py
Author: Marco Wang
Date: 2026-04-03
Course: COLX 581 — Sprint 2
Description:
    Transfer learning for the DistilBERT neural baseline via two
    complementary strategies:

    Strategy 1 — Domain-adapted backbone (primary)
    ───────────────────────────────────────────────
    Swap distilbert-base-uncased for distilbert-base-multilingual-cased
    (mDistilBERT).  The original model was pre-trained on English
    Wikipedia + BooksCorpus only.  Our dataset contains bilingual English/
    Chinese emails (see text_zh field).  The multilingual model was
    pre-trained on 104 languages including both, giving it richer
    representations for non-English tokens and code-switched text.

    Motivation:
      • Same parameter count as DistilBERT — no extra compute cost.
      • Handles Chinese characters natively; the English tokeniser in
        distilbert-base-uncased maps Chinese text almost entirely to [UNK].
      • Phishing tactics (urgency, threats, rewards) operate across languages;
        a multilingual model can leverage cross-lingual transfer.

    Strategy 2 — Layer freezing experiment (secondary, optional flag)
    ─────────────────────────────────────────────────────────────────
    When FREEZE_LAYERS > 0, the first N transformer blocks and the word
    embedding layer are frozen before fine-tuning.  Only the upper
    blocks and the classification head receive gradient updates.

    Motivation:
      • Lower layers encode universal linguistic structure (syntax,
        subword semantics) that is already well-calibrated; freezing
        them prevents "catastrophic forgetting" of the pre-trained signal.
      • Reduces trainable parameters, which matters when the labelled
        dataset is small — regularisation through freezing.
      • Allows a direct comparison: does fine-tuning ALL layers actually
        help on our task, or does the frozen signal suffice?

    Usage
    ──────
    python transfer_neural.py                          # both strategies
    python transfer_neural.py --freeze 0               # no freezing
    python transfer_neural.py --freeze 3               # freeze first 3 blocks
    python transfer_neural.py --baseline-only          # reproduce Sprint-1 result
"""

import argparse
import json
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 581

# DistilBERT baseline (Sprint 1) — English only
BASELINE_MODEL    = "distilbert-base-uncased"

# Transfer model — multilingual DistilBERT (104 languages incl. EN + ZH)
TRANSFER_MODEL    = "distilbert-base-multilingual-cased"

MAX_LENGTH  = 256
BATCH_SIZE  = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS  = 10
PATIENCE    = 3        # early-stopping on validation macro-F1
FREEZE_LAYERS = 3      # default: freeze the bottom 3 of 6 transformer blocks

LABEL2ID    = {"Ham": 0, "Phish": 1, "Spam": 2}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["Ham", "Phish", "Spam"]
NUM_LABELS  = len(LABEL_NAMES)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Data helpers ───────────────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_split(
    records: list[dict],
    multilingual: bool = False,
) -> tuple[list[str], list[int]]:
    """
    Extract texts and integer labels from JSONL records.

    When multilingual=True the English and Chinese fields are concatenated,
    which gives the multilingual tokeniser its full signal.
    For the monolingual baseline, only the English `text` field is used.
    """
    if multilingual:
        texts = [
            f"{r.get('text', '')} {r.get('text_zh', '')}".strip()
            for r in records
        ]
    else:
        texts = [r.get("text", "") for r in records]

    labels = [LABEL2ID[r["label"]] for r in records]
    return texts, labels


class EmailDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ── Layer freezing ─────────────────────────────────────────────────────────────
def freeze_lower_layers(model, n_freeze: int) -> None:
    """
    Freeze the word embedding layer and (optionally) the first n_freeze
    transformer blocks of a DistilBERT-family model.

    Rationale
    ─────────
    Lower transformer layers encode general-purpose linguistic structure
    (token identity, positional patterns).  The pre-trained signal there
    is already strong and stable.  By freezing them we:
      (a) preserve the knowledge distilled / pre-trained into those layers,
      (b) focus gradient updates on the task-specific upper layers and head,
      (c) reduce effective parameter count — beneficial for our medium-sized
          labelled dataset.

    The same principle is used in gradual-unfreezing fine-tuning (Howard &
    Ruder, 2018, ULMFiT) and is a well-established practice in NLP transfer
    learning.
    """
    if n_freeze == 0:
        print("  Layer freezing: DISABLED (all parameters trainable)")
        return

    # Freeze embedding layer
    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False
    print(f"  Frozen: word embedding layer")

    # Freeze first n_freeze transformer blocks
    n_layers = len(model.distilbert.transformer.layer)
    for i, block in enumerate(model.distilbert.transformer.layer):
        if i < min(n_freeze, n_layers):
            for param in block.parameters():
                param.requires_grad = False
            print(f"  Frozen: transformer block {i}")

    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,} / {total:,}  "
          f"({100 * trainable / total:.1f}%)")


# ── Training & evaluation loops ────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


# ── Full fine-tuning run ───────────────────────────────────────────────────────
def run_experiment(
    model_name: str,
    experiment_label: str,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    test_records: list[dict],
    device: torch.device,
    freeze_n: int = 0,
    output_subdir: str = "transfer_neural",
) -> dict:
    """
    Fine-tune a HuggingFace sequence-classification model and return results.
    """
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_label}")
    print(f"  Backbone:   {model_name}")
    print(f"  Freeze layers: {freeze_n}")
    print(f"{'='*60}")

    output_dir = os.path.join(PROJECT_ROOT, "models", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Tokeniser & datasets ──
    print(f"\nLoading tokeniser from {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = EmailDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset   = EmailDataset(val_texts,   val_labels,   tokenizer, MAX_LENGTH)
    test_dataset  = EmailDataset(test_texts,  test_labels,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    # ── Model ──
    print(f"Loading model {model_name} …")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # ── Optional layer freezing ──
    if freeze_n > 0:
        print(f"\nApplying layer freezing (bottom {freeze_n} transformer blocks) …")
        freeze_lower_layers(model, freeze_n)

    # ── Class-weighted cross-entropy ──
    class_weights = compute_class_weights(train_labels, NUM_LABELS).to(device)
    print(f"\nClass weights: " +
          ", ".join(f"{ID2LABEL[i]}={w:.3f}" for i, w in enumerate(class_weights)))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser & scheduler ──
    #   Only pass parameters with requires_grad=True (respects frozen layers)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Training loop with early stopping ──
    print(f"\nTraining for up to {NUM_EPOCHS} epochs (patience={PATIENCE}) …\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Train F1':>8}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Val F1':>6}")
    print("-" * 72)

    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.4f}  {train_f1:>8.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>7.4f}  {val_f1:>6.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    # ── Reload best checkpoint & final test evaluation ──
    print(f"\nBest validation macro-F1: {best_val_f1:.4f}")
    print(f"Loading best model checkpoint from {output_dir} …")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)

    test_loss, test_acc, test_f1, test_preds, test_true = evaluate(
        model, test_loader, criterion, device
    )

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS — {experiment_label}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Macro-F1  : {test_f1:.4f}")
    print(f"  Loss      : {test_loss:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=LABEL_NAMES, digits=4))

    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(test_true, test_preds, labels=list(range(NUM_LABELS)))
    header = "          " + "  ".join(f"{n:>6}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<6}  " + "  ".join(f"{v:>6}" for v in row))

    # ── Save predictions ──
    pred_path = os.path.join(output_dir, "test_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(test_records):
            row = {
                "text":            record.get("text", ""),
                "true_label":      record["label"],
                "predicted_label": ID2LABEL[int(test_preds[i])],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved → {pred_path}")

    return {
        "experiment":  experiment_label,
        "model":       model_name,
        "freeze_n":    freeze_n,
        "best_val_f1": best_val_f1,
        "test_acc":    test_acc,
        "test_f1":     test_f1,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neural transfer learning: multilingual DistilBERT with optional layer freezing"
    )
    parser.add_argument(
        "--freeze", type=int, default=FREEZE_LAYERS,
        help=f"Number of transformer blocks to freeze from the bottom (default: {FREEZE_LAYERS}). "
             "Set to 0 to disable freezing."
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Run only the Sprint-1 DistilBERT baseline (monolingual, no freezing)."
    )
    parser.add_argument(
        "--transfer-only", action="store_true",
        help="Skip the Sprint-1 baseline and run only the transfer model."
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    set_seed(SEED)

    print("=" * 60)
    print("  Transfer Learning | Neural Baseline")
    print("  Multilingual DistilBERT + Layer Freezing Experiment")
    print("=" * 60)

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # ── Load data ──
    print("\nLoading data splits …")
    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_records   = load_jsonl(os.path.join(DATA_DIR, "validation.jsonl"))
    test_records  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    print(f"  Train: {len(train_records)}  |  Val: {len(val_records)}  |  Test: {len(test_records)}")

    # Monolingual text (for Sprint-1 baseline reproduction)
    train_texts_en, train_labels = prepare_split(train_records, multilingual=False)
    val_texts_en,   val_labels   = prepare_split(val_records,   multilingual=False)
    test_texts_en,  test_labels  = prepare_split(test_records,  multilingual=False)

    # Bilingual text (EN + ZH concatenated) for multilingual model
    train_texts_bi, _ = prepare_split(train_records, multilingual=True)
    val_texts_bi,   _ = prepare_split(val_records,   multilingual=True)
    test_texts_bi,  _ = prepare_split(test_records,  multilingual=True)

    results = []

    # ── Experiment 1: Reproduce Sprint-1 baseline (optional) ──
    if not args.transfer_only:
        print("\n" + "─" * 60)
        print("  STRATEGY 0: Sprint-1 baseline reproduction")
        print("  (distilbert-base-uncased, English only, no freezing)")
        print("─" * 60)
        r0 = run_experiment(
            model_name="distilbert-base-uncased",
            experiment_label="Sprint-1 Baseline (monolingual DistilBERT)",
            train_texts=train_texts_en,
            train_labels=train_labels,
            val_texts=val_texts_en,
            val_labels=val_labels,
            test_texts=test_texts_en,
            test_labels=test_labels,
            test_records=test_records,
            device=device,
            freeze_n=0,
            output_subdir="transfer_neural_baseline_repro",
        )
        results.append(r0)

    if args.baseline_only:
        print("\nBaseline-only mode — done.")
        return

    # ── Experiment 2: Multilingual DistilBERT — all layers trainable ──
    print("\n" + "─" * 60)
    print("  STRATEGY 1: Multilingual DistilBERT (full fine-tuning)")
    print("  Rationale: EN-only DistilBERT maps Chinese to [UNK];")
    print("             mDistilBERT has 104-language pre-training incl. ZH.")
    print("─" * 60)
    r1 = run_experiment(
        model_name=TRANSFER_MODEL,
        experiment_label="mDistilBERT – full fine-tuning (bilingual input)",
        train_texts=train_texts_bi,
        train_labels=train_labels,
        val_texts=val_texts_bi,
        val_labels=val_labels,
        test_texts=test_texts_bi,
        test_labels=test_labels,
        test_records=test_records,
        device=device,
        freeze_n=0,
        output_subdir="transfer_neural_full",
    )
    results.append(r1)

    # ── Experiment 3: Multilingual DistilBERT — frozen lower layers ──
    print("\n" + "─" * 60)
    print(f"  STRATEGY 2: Multilingual DistilBERT + frozen bottom {args.freeze} layers")
    print("  Rationale: Frozen lower layers retain general cross-lingual")
    print("             representations; reduces catastrophic forgetting.")
    print("─" * 60)
    r2 = run_experiment(
        model_name=TRANSFER_MODEL,
        experiment_label=f"mDistilBERT – frozen {args.freeze} layer(s) (bilingual input)",
        train_texts=train_texts_bi,
        train_labels=train_labels,
        val_texts=val_texts_bi,
        val_labels=val_labels,
        test_texts=test_texts_bi,
        test_labels=test_labels,
        test_records=test_records,
        device=device,
        freeze_n=args.freeze,
        output_subdir=f"transfer_neural_frozen{args.freeze}",
    )
    results.append(r2)

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  {'Experiment':<45}  {'Val F1':>7}  {'Test Acc':>8}  {'Test F1':>7}")
    print("  " + "─" * 70)
    for r in results:
        print(f"  {r['experiment']:<45}  "
              f"{r['best_val_f1']:>7.4f}  "
              f"{r['test_acc']:>8.4f}  "
              f"{r['test_f1']:>7.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
