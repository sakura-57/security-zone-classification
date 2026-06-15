# security-zone-classification

> Predicting security-zone tags for unlabeled network objects.

A machine-learning approach to assigning security-zone tags to network objects
(hosts, services, firewall rules) when only a subset of objects are labeled. The
project combines a **rule co-occurrence signal** with a **supervised classifier**
to propagate zone labels across a sparsely-labeled network graph.

---

## Table of Contents

- [Problem](#problem)
- [Approach](#approach)
- [Why these choices](#why-these-choices)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Key Decisions & Learnings](#key-decisions--learnings)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Problem

In a firewall / network-security configuration, objects are grouped into
**security zones** (e.g. internal, DMZ, external). Many objects carry an
explicit zone tag, but a large share are **unlabeled** — they appear in rules
and policies without an assigned zone.

The goal is to **predict the most likely security zone** for each unlabeled
object, so the configuration can be analyzed, audited, and cleaned up
automatically instead of by hand.

This is a **semi-supervised, multi-class classification** problem on tabular
and relational (graph-structured) data.

---

## Approach

The pipeline has two complementary signals:

1. **Rule co-occurrence voting** — Objects that frequently appear together in
   the same firewall rules tend to belong to related zones. For each unlabeled
   object, neighboring labeled objects "vote" for their zone, weighted by
   co-occurrence frequency. This produces a propagated label signal directly
   from the rule structure.

2. **Supervised classifier (Random Forest)** — A Random Forest model is trained
   on labeled objects using their tabular features (and the co-occurrence
   signal as an engineered feature) to predict the zone for unlabeled objects.

The two signals are combined to produce the final zone prediction for each
unlabeled object.

```
raw network objects + rules
        │
        ▼
  feature engineering ──► rule co-occurrence matrix
        │                        │
        │                        ▼
        │                 co-occurrence voting
        ▼                        │
  Random Forest  ◄───────────────┘
        │
        ▼
  predicted security-zone tags
```

---

## Why these choices

These decisions came out of exploratory analysis of the data, not defaults:

- **Co-occurrence voting over label propagation.** A standard graph label-
  propagation approach assumes **homophily** — that connected nodes share
  labels. Measured homophily on this graph was very low (~0.10), meaning
  connected objects often belong to *different* zones. Label propagation
  therefore performs poorly here, while direct co-occurrence voting (treating
  shared-rule membership as the signal) is more robust.

- **Random Forest over a linear / single-tree model.** The labeled subset is
  affected by **selection bias** (labeled objects are not a random sample of
  all objects). Random Forest's bagging and feature subsampling make it more
  resilient to that bias and to the mixed, partly-correlated feature set than
  a single decision tree or a linear model.

---

## Project Structure

> Update this to match your actual files once the code is committed.

```
security-zone-classification/
├── data/                 # input data (gitignored if sensitive)
│   ├── raw/              # original network objects + rules
│   └── processed/        # engineered features
├── src/
│   ├── features.py       # feature engineering + co-occurrence matrix
│   ├── voting.py         # rule co-occurrence voting
│   ├── model.py          # Random Forest training / inference
│   └── pipeline.py       # end-to-end run
├── notebooks/
│   └── eda.ipynb         # exploratory analysis (homophily, class balance)
├── outputs/              # predictions + reports
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
# clone
git clone https://github.com/sakura-57/security-zone-classification.git
cd security-zone-classification

# create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, `pandas`, `numpy`, `scikit-learn`
(add `networkx` if you build an explicit graph). Pin versions in
`requirements.txt`.

---

## Usage

> Adjust to your actual entry point.

```bash
# run the full pipeline: features → voting → model → predictions
python src/pipeline.py --input data/raw/objects.csv --output outputs/predictions.csv
```

Or step by step:

```bash
python src/features.py    # build features + co-occurrence matrix
python src/voting.py      # generate co-occurrence vote signal
python src/model.py       # train Random Forest and predict
```

Output is a table of objects with their **predicted security zone** and a
**confidence / vote score**.

---

## Data

Each network object has tabular attributes plus its membership in firewall
rules. A subset carry a ground-truth `zone` label; the rest are unlabeled and
are the prediction target.

> Note whether the dataset is included, synthetic, or private. If it can't be
> shared, describe the schema (columns, label values) so others can adapt the
> code to their own data.

---

## Results

> Fill in with your actual evaluation once finalized.

| Metric | Value |
| --- | --- |
| Accuracy (held-out labeled) | _e.g. 0.XX_ |
| Macro F1 | _e.g. 0.XX_ |
| Per-zone F1 | _e.g. internal 0.XX / DMZ 0.XX / external 0.XX_ |

Evaluation was done on a held-out portion of the **labeled** objects, with
attention to per-zone performance given class imbalance.

---

## Key Decisions & Learnings

- **Measure homophily before reaching for graph methods.** Low homophily
  (~0.10) ruled out label propagation and pointed to co-occurrence voting.
- **Account for selection bias** in how labeled objects were sampled — it shaped
  both the model choice (Random Forest) and how results were interpreted.
- **Combine a structural signal (rule co-occurrence) with a feature-based
  classifier** rather than relying on either alone.

---

## Limitations & Future Work

- Co-occurrence voting depends on objects sharing rules; isolated objects get a
  weak signal.
- Selection bias in labels limits how far held-out metrics generalize to the
  truly-unlabeled population.
- Possible extensions: calibrated probabilities, graph neural networks if
  homophily improves with richer edges, active learning to label the most
  informative objects first.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.
