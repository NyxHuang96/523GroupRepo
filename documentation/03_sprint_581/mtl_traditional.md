---
title: "Traditional Multi-Task Learning (Classification + Silver NER Density)"
author: Darwin Zhang
date: "2026-04-10"
---

### Overview

This document describes the implementation of `mtl_traditional.py`, which carries off of the our existing Support Vector Machine (SVM) pipeline, with a Multi-Task Learning (MTL) objective. While neural models achieve MTL through shared hidden layers and multiple "heads", this traditional implementation utilizes Feature-Augmented Joint Learning. 

We force the model to account for both vocabulary (TF-IDF) and structural intent (Named Entity density) simultaneously to break the semantic barrier of pure vocabulary which we have identified in the previous sprint.

### Architecture

Unlike the Neural MTL which uses a DistilBERT backbone, our traditional MTL model meshes the features together:

1. **Primary Task (Vocabulary):** Standard TF-IDF vectorization capturing unigrams and bigrams across the bilingual (English and Chinese) corpus.
2. **Auxiliary Task (Silver NER Density):** A custom transformer that utilizes the SpaCy `en_core_web_sm` model to calculate the density of specific entities (ORG, MONEY, DATE, PERSON, GPE) within each document.

By concatenating these two feature sets, the SVM's decision boundary should be mathematically influenced by both what the sender says (words) and who they claim to be (entities).

### Why NER as the Auxiliary Task?

To remain consistent with the neural MTL implementation, we selected Named Entity Recognition (NER) as our secondary signal. Phishing and Spam are not just random collections of words; and we are targetting the fingerprints of them. 

* **Phishing:** Systematically impersonates specific ORG entities (Banks, Government agencies) and utilizes DATE markers to manufacture false urgency.
* **Spam:** Focuses heavily on PRODUCT and MONEY but lacks the institutional impersonation found in Phish.
* **Ham:** Features a diffuse distribution of PERSON and GPE entities within a conversational context.

### Results and Comparison

We evaluated the performance of our traditional MTL model against our Sprint 2 baseline (which included the 1,104 Transfer Learning samples). The results on our 12 sample validation set are detailed below.

#### Metric Comparison: Baseline vs. MTL
| Metric | Baseline (TF-IDF Only) | MTL (TF-IDF + NER) |
| :--- | :---: | :---: |
| **Overall Accuracy** | 0.9167 | 0.8333 |
| **Ham F1 Score** | 1.00 | 1.00 |
| **Phish F1 Score** | 0.83 | 0.83 |
| **Spam F1 Score** | 0.80 | 0.67 |

#### Comparative Performance (Traditional vs. Neural)
| Configuration | Val Accuracy | Val Macro F1 |
| :--- | :---: | :---: |
| **Sprint 3 Neural MTL (DistilBERT + NER)** | 0.7500 | 0.7231 |
| **Sprint 3 Traditional MTL (SVM + NER)** | 0.8333 | 0.8300 |
| **Sprint 2 Traditional Baseline (Transfer)** | **0.9167** | **0.9000** |

### Discussion

The results of this experiment provide a fascinating look at the differences between Traditional and Neural Multi-Task Learning.

#### The "Feature Noise" Effect
In the traditional SVM, adding the MTL objective (NER density) actually caused a decrease in performance compared to the baseline. While the baseline reached 91.6% accuracy, the MTL model dropped to 83.3%. This is likely due to adding noise within our low sample size. In a traditional model, concatenating dense numerical features (NER percentages) with thousands of sparse text features (TF-IDF) may confuse the linear boundary on very small datasets. Therefore, the model struggled to weigh the new entity information without losing the signal from the vocabulary.

#### Traditional vs. Neural Success
Interestingly, the Traditional model (both baseline and MTL) still maintains a higher overall accuracy on our specific validation set than the Neural MTL model. This suggests that for our low resource data (12 to 16 evaluation samples), the high bias of the SVM is more robust than the deep representations of DistilBERT. However, as noted in Tianhao's neural report, the MTL objective helped the neural model achieve a much better validation curve (+0.15 improvement over its own baseline). 

#### The Spam Problem
Both models continue to struggle with the Spam class. In the traditional MTL model, Spam recall dropped to 0.50. This confirms that even with Named Entity data, the understanding of semantics still remain a significant challenge. The model can see "MONEY" and "DATE" tags, but it still cannot fully distinguish if those entities are being used for a legitimate sale (Spam) or a malicious theft (Phish).

### Conclusion

This sprint successfully demonstrated that while Multi-Task Learning is a powerful tool for Neural Networks to learn underlying representations, it must be handled carefully in Traditional ML to avoid introducing more noise. The traditional baseline remains our strongest individual model for this specific dataset, but the Neural MTL has proven to be the more "teachable" architecture for secondary linguistic tasks (and likely would perform better if we scale the data).