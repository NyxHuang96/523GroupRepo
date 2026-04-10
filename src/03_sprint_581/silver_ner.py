"""
File: silver_ner.py
Author: Tianhao Cao
Date: 2026-04-08
Course: COLX 523 — Sprint 03 (581)
Description:
    Shared utility for producing silver NER annotations with spaCy and
    aligning them with HuggingFace WordPiece tokens as BIO tag ids.

    The auxiliary NER task is justified in documentation/03_sprint_581/README.md
    — phishing/spam emails have a characteristic ORG/MONEY/DATE/PERSON profile
    that the encoder is encouraged to attend to under the joint objective.
"""

from __future__ import annotations
import functools
from typing import List, Tuple

# spaCy entity labels we keep. Anything else collapses into "O".
ENTITY_TYPES = [
    "PERSON", "ORG", "GPE", "LOC", "MONEY", "DATE",
    "TIME", "PRODUCT", "CARDINAL", "PERCENT",
]

# Build BIO label space: O + B-X / I-X for each entity type.
NER_LABELS: List[str] = ["O"]
for et in ENTITY_TYPES:
    NER_LABELS.append(f"B-{et}")
    NER_LABELS.append(f"I-{et}")

NER_LABEL2ID = {lab: i for i, lab in enumerate(NER_LABELS)}
NER_ID2LABEL = {i: lab for lab, i in NER_LABEL2ID.items()}
NUM_NER_LABELS = len(NER_LABELS)
PAD_NER_ID = -100  # ignored by CrossEntropyLoss


@functools.lru_cache(maxsize=1)
def _get_spacy():
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "parser"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run:  python -m spacy download en_core_web_sm"
        ) from exc
    return nlp


def char_bio_for_text(text: str) -> List[Tuple[int, int, str]]:
    """Run spaCy and return [(start_char, end_char, BIO_label)] for each entity token.

    Each *entity* contributes one B- token and possibly several I- tokens
    (we use spaCy's word-tokens). Non-entity spans are not returned —
    they are filled in with "O" by `align_bio_to_wordpieces`.
    """
    nlp = _get_spacy()
    doc = nlp(text[:5000])  # cap absurdly long emails for spaCy speed
    spans: List[Tuple[int, int, str]] = []
    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPES:
            continue
        for i, tok in enumerate(ent):
            tag = ("B-" if i == 0 else "I-") + ent.label_
            spans.append((tok.idx, tok.idx + len(tok.text), tag))
    return spans


def align_bio_to_wordpieces(
    text: str,
    offset_mapping: List[Tuple[int, int]],
    special_tokens_mask: List[int],
) -> List[int]:
    """Convert spaCy char-level entity spans into per-wordpiece BIO label ids.

    - Special tokens ([CLS], [SEP], [PAD]) get PAD_NER_ID so they're ignored
      by the loss.
    - Wordpieces that fall inside an entity span receive that entity's BIO id;
      continuation pieces of a B- token become I-.
    - All other wordpieces are "O".
    """
    spans = char_bio_for_text(text)
    # Build a fast char -> label lookup over only entity spans.
    # For each wordpiece, find the first span that overlaps it.
    label_ids: List[int] = []
    for (start, end), is_special in zip(offset_mapping, special_tokens_mask):
        if is_special or (start == 0 and end == 0):
            label_ids.append(PAD_NER_ID)
            continue
        chosen = "O"
        for s, e, tag in spans:
            if start < e and end > s:  # overlap
                # If the wordpiece doesn't start at the entity's start char,
                # demote a B- to I-.
                if tag.startswith("B-") and start > s:
                    chosen = "I-" + tag[2:]
                else:
                    chosen = tag
                break
        label_ids.append(NER_LABEL2ID[chosen])
    return label_ids
