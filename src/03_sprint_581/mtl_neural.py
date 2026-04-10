"""
File: mtl_neural.py
Author: Tianhao Cao
Date: 2026-04-08
Course: COLX 523 — Sprint 03 (581)
Description:
    Multi-task DistilBERT for the Ham/Phish/Spam email task.

    Architecture:
        DistilBERT encoder
          ├── classification head  → 3 classes (Ham/Phish/Spam)        [primary]
          └── token-classification head → BIO NER tags (silver)        [auxiliary]

    Loss:  L = L_cls + LAMBDA_NER * L_ner
    Silver NER labels are produced on the fly with spaCy en_core_web_sm.
    See documentation/03_sprint_581/README.md for the justification of
    NER as the auxiliary task.
"""

import json
import os
import random
import warnings
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from silver_ner import (
    align_bio_to_wordpieces,
    NUM_NER_LABELS,
    PAD_NER_ID,
)

warnings.filterwarnings("ignore")

# ----- Config -----
SEED = 581
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
PATIENCE = 3
LAMBDA_NER = 0.3  # weight of the auxiliary NER loss

LABEL2ID = {"Ham": 0, "Phish": 1, "Spam": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----- Data -----

def load_jsonl(filepath: str) -> list[dict]:
    out = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class MTLEmailDataset(Dataset):
    """Tokenises emails and produces (input_ids, attention_mask, cls_label, ner_labels)."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        offset_mapping = enc["offset_mapping"].tolist()
        special_tokens_mask = enc["special_tokens_mask"].tolist()

        # Pre-compute silver NER labels (one BIO id per wordpiece).
        ner_label_lists = []
        for text, offsets, specials in zip(texts, offset_mapping, special_tokens_mask):
            ner_label_lists.append(
                align_bio_to_wordpieces(text, offsets, specials)
            )
        self.ner_labels = torch.tensor(ner_label_lists, dtype=torch.long)
        self.cls_labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.cls_labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "cls_labels": self.cls_labels[idx],
            "ner_labels": self.ner_labels[idx],
        }


def prepare_split(records):
    texts = [r["text"] for r in records]
    labels = [LABEL2ID[r["label"]] for r in records]
    return texts, labels


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    w = 1.0 / counts
    return torch.tensor(w / w.sum() * num_classes, dtype=torch.float32)


# ----- Model -----

class DistilBertMTL(nn.Module):
    """Shared DistilBERT encoder with two heads: sequence-cls and token-NER."""

    def __init__(self, model_name: str, num_cls_labels: int, num_ner_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden, num_cls_labels)
        self.ner_head = nn.Linear(hidden, num_ner_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state          # (B, T, H)
        pooled = hidden[:, 0]                   # [CLS]
        cls_logits = self.cls_head(self.dropout(pooled))         # (B, C)
        ner_logits = self.ner_head(self.dropout(hidden))         # (B, T, N)
        return cls_logits, ner_logits


# ----- Train / Eval -----

def train_one_epoch(model, loader, optimizer, scheduler, cls_criterion, ner_criterion, device):
    model.train()
    total = 0.0
    n = 0
    preds_all, labels_all = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cls_labels = batch["cls_labels"].to(device)
        ner_labels = batch["ner_labels"].to(device)

        optimizer.zero_grad()
        cls_logits, ner_logits = model(input_ids, attention_mask)
        loss_cls = cls_criterion(cls_logits, cls_labels)
        loss_ner = ner_criterion(
            ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1)
        )
        loss = loss_cls + LAMBDA_NER * loss_ner
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        bs = cls_labels.size(0)
        total += loss.item() * bs
        n += bs
        preds_all.extend(cls_logits.argmax(-1).cpu().numpy())
        labels_all.extend(cls_labels.cpu().numpy())
    return total / n, accuracy_score(labels_all, preds_all), f1_score(labels_all, preds_all, average="macro")


@torch.no_grad()
def evaluate(model, loader, cls_criterion, device):
    model.eval()
    total = 0.0
    n = 0
    preds_all, labels_all = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cls_labels = batch["cls_labels"].to(device)

        cls_logits, _ = model(input_ids, attention_mask)
        loss = cls_criterion(cls_logits, cls_labels)
        bs = cls_labels.size(0)
        total += loss.item() * bs
        n += bs
        preds_all.extend(cls_logits.argmax(-1).cpu().numpy())
        labels_all.extend(cls_labels.cpu().numpy())
    return (
        total / n,
        accuracy_score(labels_all, preds_all),
        f1_score(labels_all, preds_all, average="macro"),
        np.array(preds_all),
        np.array(labels_all),
    )


def main():
    set_seed(SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "data", "processed"))
    output_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "models", "mtl_neural"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 64)
    print("DistilBERT MTL — Email Classification + Silver NER (auxiliary)")
    print("=" * 64)

    train_records = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_records = load_jsonl(os.path.join(data_dir, "validation.jsonl"))
    test_records = load_jsonl(os.path.join(data_dir, "test.jsonl"))
    print(f"\nSizes  →  Train: {len(train_records)}  |  Val: {len(val_records)}  |  Test: {len(test_records)}")

    train_texts, train_labels = prepare_split(train_records)
    val_texts, val_labels = prepare_split(val_records)
    test_texts, test_labels = prepare_split(test_records)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    print("Building datasets (running spaCy for silver NER, this may take a moment) …")
    train_ds = MTLEmailDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds = MTLEmailDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_ds = MTLEmailDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}  |  λ_NER = {LAMBDA_NER}")

    model = DistilBertMTL(MODEL_NAME, NUM_LABELS, NUM_NER_LABELS).to(device)

    cls_weights = compute_class_weights(train_labels, NUM_LABELS).to(device)
    print("Class weights:  " + ", ".join(f"{ID2LABEL[i]}: {w:.3f}" for i, w in enumerate(cls_weights)))
    cls_criterion = nn.CrossEntropyLoss(weight=cls_weights)
    ner_criterion = nn.CrossEntropyLoss(ignore_index=PAD_NER_ID)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    print(f"\nTraining for up to {NUM_EPOCHS} epochs (patience={PATIENCE}) …\n")
    print(f"{'Epoch':>5}  {'Train L':>8}  {'Train Acc':>9}  {'Train F1':>8}  "
          f"{'Val L':>7}  {'Val Acc':>7}  {'Val F1':>6}")
    print("-" * 64)

    best_f1 = -1.0
    patience = 0
    best_path = os.path.join(output_dir, "best.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, cls_criterion, ner_criterion, device
        )
        v_loss, v_acc, v_f1, _, _ = evaluate(model, val_loader, cls_criterion, device)
        print(f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>9.4f}  {tr_f1:>8.4f}  "
              f"{v_loss:>7.4f}  {v_acc:>7.4f}  {v_f1:>6.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest validation macro-F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_f1, preds, true = evaluate(model, test_loader, cls_criterion, device)

    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    print("\n" + "=" * 64)
    print("TEST SET RESULTS")
    print("=" * 64)
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  Macro-F1 : {test_f1:.4f}")
    print(f"  Loss     : {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(true, preds, target_names=label_names, digits=4))
    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true, preds, labels=list(range(NUM_LABELS)))
    print("          " + "  ".join(f"{n:>6}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:<6}  " + "  ".join(f"{v:>6}" for v in row))

    pred_path = os.path.join(output_dir, "test_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(test_records):
            f.write(json.dumps({
                "text": rec["text"],
                "true_label": rec["label"],
                "predicted_label": ID2LABEL[int(preds[i])],
            }, ensure_ascii=False) + "\n")
    print(f"\nTest predictions saved to {pred_path}")
    print("Done.")


if __name__ == "__main__":
    main()
