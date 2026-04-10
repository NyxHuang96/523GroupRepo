# `silver_ner.py` — Silver NER Annotation Utility

- **Author:** Tianhao Cao
- **Date:** 2026-04-08
- **Course:** COLX 523 — Sprint 03 (581)
- **Disclaimer:** This documentation is assisted by Claude Sonnet-4.6

## Purpose

`silver_ner.py` is a shared helper module used by `mtl_neural.py`. It does
two things:

1. **Runs spaCy** on raw email text to extract named-entity spans.
2. **Aligns** those character-level spans onto HuggingFace WordPiece tokens
   so the result can be fed directly to a DistilBERT token-classification
   head as integer BIO label ids.

The auxiliary NER task is justified by the entity profile of phishing/spam
emails: phish systematically impersonates `ORG`s (banks, PayPal, Microsoft,
IRS), references `MONEY` amounts, and manufactures urgency through
`DATE`/`TIME` mentions. Forcing the encoder to predict entity spans pushes
representations toward the lexical anchors that are diagnostic of phishing,
rather than toward surface topical content.

## BIO label space

```python
ENTITY_TYPES = ["PERSON", "ORG", "GPE", "LOC", "MONEY",
                "DATE", "TIME", "PRODUCT", "CARDINAL", "PERCENT"]
NER_LABELS = ["O", "B-PERSON", "I-PERSON", "B-ORG", "I-ORG", ...]   # 21 labels
PAD_NER_ID = -100   # ignored by torch.nn.CrossEntropyLoss
```

Only ten entity types are kept (chosen for relevance to phish/spam); other
spaCy types collapse to `O`. Each kept type is expanded into B- and I-
variants, giving `1 + 10 × 2 = 21` labels. `PAD_NER_ID = -100` is the
PyTorch convention for "ignore this position when computing loss" — it is
used for `[CLS]`, `[SEP]`, and `[PAD]` wordpieces.
