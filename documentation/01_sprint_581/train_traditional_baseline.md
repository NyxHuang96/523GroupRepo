---
title: "Evaluation of Traditional ML Baseline (TF-IDF + SVM)"
author: Darwin Zhang
date: "2026-03-28"
---

## Overview
This document outlines the evaluation of our traditional Machine Learning baseline. To establish a benchmark for our future neural networks, for the traditional method, a Linear Support Vector Machine (SVM) using a TF-IDF vectorizer was trained. The model features a custom bilingual tokenizer (Jieba for Chinese, Regex for English) to capture 1-gram and 2-gram linguistic patterns.

The model was trained on **105 Gold-annotated samples** and evaluated against a strictly isolated validation (dev) set of **12 Gold-annotated samples**. 

---

## 1. Quantitative Results (The Numbers)

Despite being trained on a very small, the baseline performed well, achieving an **83% Overall Accuracy**.

**Classification Report Summary:**
* **Ham (Legitimate):** 100% Precision / 100% Recall
* **Phish (Malicious):** 71% Precision / 100% Recall
* **Spam (Commercial Junk):** 100% Precision / 50% Recall

**Confusion Matrix:**
| True Label \ Predicted | Predicted Ham | Predicted Phish | Predicted Spam |
| :--- | :---: | :---: | :---: |
| **True Ham** | 3 | 0 | 0 |
| **True Phish** | 0 | 5 | 0 |
| **True Spam** | 0 | **2** | 2 |

---

## 2. What Do These Metrics Actually Mean?

To make sense of the numbers:

* **Perfect Safety (Ham):** The model perfectly identified all legitimate messages. It did not flag any safe communication as dangerous, which means a "zero false-alarm" rate for normal users.
* **Highly Paranoid about Phishing:** The model successfully caught **100% of the actual Phishing attempts**. From a security standpoint, this is a massive win. However, it cast its net a bit too wide.
* **The Blind Spot (Spam vs. Phish):** The only errors the model made were taking 2 actual **Spam** messages and over-escalating them, incorrectly classifying them as **Phish**. 

---

## 3. Qualitative Error Analysis

Why did the model confuse Spam for Phishing? This may be an error on our model and what we can look to improve on.

With TF-IDF, we are using a word counter that may be looking into the vocabulary, and not the semantics that humans look for. 

**The Overlapping Vocabulary Problem:**
Both Spam (aggressive marketing) and Phishing (malicious theft) rely on the same linguistic toolbox: 
* *Urgency:* "Act now," "Limited time," "Urgent."
* *Action:* "Click here," "Visit this link," "Claim your prize."

Because our baseline model lacks deep contextual awareness, when it sees a Spam message packed with urgency and links, it heavily weights those features and cautiously guesses "Phish" to be safe. It cannot differentiate between the intent to *sell a counterfeit watch* (Spam) and the intent to *steal a bank password* (Phish).

## Conclusion & Next Steps
This traditional baseline is a strong success. It helps show that that our data is high-quality and clean enough for a model to perfectly isolate legitimate communication and catch all major threats. 

However, the Spam/Phish confusion confirms the core motivation of our project: **fraud is nuanced**. To solve this confusion, our next sprint will look into more neural methods, but we also have a neural baseline to train off of. Unlike this baseline, neural models can read the *context* surrounding the words, allowing them to differentiate between aggressive marketing and actual deceptive social engineering.