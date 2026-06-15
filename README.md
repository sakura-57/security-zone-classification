# security-zone-classification

> Predicting security-zone tags for unlabeled network objects.

A semi-supervised approach to assigning **security-zone tags** to network objects
when only a subset of objects are labeled. Given a set of firewall rules
(directed `src → dst` connections) and a partial set of zone labels, the project
predicts the most likely zone for every unlabeled object using a
**rule co-occurrence voting** signal.

The project is documented end-to-end: the main solution, the evaluation, and a
separate set of prototyped improvements.

---

## Table of Contents

- [Problem](#problem)
- [Data](#data)
- [Approach](#approach)
- [Why these choices](#why-these-choices)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Key Learnings](#key-learnings)
- [License](#license)

---

## Problem

In a firewall / network-security configuration, objects (hosts, services, groups)
are organized into **security zones**. Some objects carry an explicit zone tag,
but most do not — they appear in rules without an assigned zone.

The goal is to **predict the zone tag for each unlabeled object** so the
configuration can be audited and cleaned up automatically instead of by hand.

This is a **semi-supervised, multi-class classification** problem on
relational (graph-structured) data.

---

## Data

| Item | Value |
| --- | --- |
| Total objects | ~10,000 (3,000 labeled / 7,000 unlabeled) |
| Firewall rules | 500 unique rule IDs, expanded into ~113K directed `src → dst` edges |
| Zone tags | 9 classes |

The zones follow a **tiered security architecture**:
`Users → T0_DMZ → T1_DMZ → T2_Application → T3_Database`, plus
`Common_services`, `Management`, `Guests`, and `c_TER_CH_SRV`.

Two key properties shaped the modeling:

- **Low homophily / mostly inter-zone traffic.** ~89% of connections cross
  zones; only ~11% are intra-zone. Connected objects usually belong to
  *different* zones.
- **Severe selection bias.** Labeled objects have a much lower mean degree
  (~14.9) than unlabeled ones (~25.9). Only 9 of 578 hub nodes are labeled —
  the model trains on "easy" low-degree nodes but must predict "hard"
  high-degree ones.

**Input files** (in `data/`):
- `rules.csv` — `rule_id, src_obj_id, dst_obj_id`
- `tags.csv` — `obj_id, tag` (the 3,000 labeled objects)

---

## Approach

The **main solution** (`solution.py`) uses a single signal:

**Rule co-occurrence voting.** For each unlabeled object, find labeled objects
that appear on the *same side* of the *same rules* (co-sources or
co-destinations) and predict its tag by majority vote. Confidence is the share
of votes going to the winning tag.

```
data/rules.csv  +  data/tags.csv
        │
        ▼
   clean (drop self-loops & duplicate edges)
        │
        ▼
   index objects → rules they appear in (as src / as dst)
        │
        ▼
   co-occurrence majority vote
        │
        ├──► leave-one-out evaluation  ──► metrics.csv
        ▼
   predicted zone tags             ──► predictions.csv
```

---

## Why these choices

These decisions came out of EDA, not defaults:

- **Co-occurrence voting over label propagation.** Standard graph
  label-propagation assumes **homophily** (connected nodes share labels).
  Measured homophily here was very low — connected objects mostly belong to
  *different* zones — so propagation performs poorly. Direct co-occurrence
  voting (treating shared-rule membership as the signal) is more robust.

- **Co-occurrence voting over a supervised classifier (for the main approach).**
  A Random Forest on graph features was considered but **set aside**: the
  feature distributions of labeled and unlabeled objects don't overlap well
  enough (selection bias), so a model trained on labeled nodes wouldn't transfer
  reliably to the unlabeled population.

- **Leave-one-out over k-fold for evaluation.** The rarest classes have only
  8–10 examples, which would break stratified k-fold splits.

---

## Project Structure

```
security-zone-classification/
├── data/
│   ├── rules.csv         # firewall rules: rule_id, src_obj_id, dst_obj_id
│   └── tags.csv          # labeled objects: obj_id, tag
├── solution.py           # main pipeline: load → EDA → voting → eval → predict
├── improvements.py       # prototyped next steps (not part of the main solution)
├── predictions.csv       # predicted tags for the 7,000 unlabeled objects
├── metrics.csv           # leave-one-out evaluation metrics
├── writeup.txt           # short write-up: EDA, method, evaluation, discussion
├── LICENSE               # MIT
└── README.md
```

---

## Installation

```bash
git clone https://github.com/sakura-57/security-zone-classification.git
cd security-zone-classification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install pandas numpy scikit-learn matplotlib seaborn networkx
```

**Requirements:** Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `matplotlib`,
`seaborn`, `networkx`.
*(No `requirements.txt` is committed yet — consider adding one to pin versions.)*

---

## Usage

Run the main solution (writes `predictions.csv` and `metrics.csv`):

```bash
python solution.py
```

Run the improvement prototypes (depends on `predictions.csv`
produced by `solution.py`):

```bash
python improvements.py
```

---

## Results

Evaluated with **leave-one-out** over the 3,000 labeled objects
(values from `metrics.csv`):

| Metric | Value |
| --- | --- |
| Accuracy | 0.890 |
| Precision (macro) | 0.507 |
| Recall (macro) | 0.474 |
| F1 (macro) | 0.484 |
| F1 (weighted) | 0.870 |

**Reading the numbers honestly:** high accuracy / weighted-F1 are driven by the
large, well-connected classes. The much lower **macro** scores reflect poor
performance on rare classes (e.g. `Guests` ≈ 10, `c_TER_CH_SRV` ≈ 8 examples).
Because evaluation runs on low-degree labeled nodes, these metrics likely
**overestimate** real performance on the high-degree unlabeled objects that are
the actual prediction target.

`predictions.csv` contains predicted tags for all 7,000 unlabeled objects.

---

## Future Work

Prototyped in `improvements.py` (not part of the main solution):

1. **Weighted voting** — weight votes by inverse class frequency so rare classes
   aren't drowned out by common ones.
2. **Connection-profile features** — for each object, build in/out tag
   distributions, degree features, and unknown-neighbor fractions, capturing
   *how* an object communicates rather than just who it shares a rule with.
3. **Random Forest on those profiles** — with `class_weight='balanced'` to handle
   imbalance, once the labeled/unlabeled feature gap is addressed.
4. **Zero-traffic / forbidden-pair validation** — flag predictions that imply
   tag pairs which never co-occur in the labeled data (e.g. `c_TER_CH_SRV → Users`)
   as likely errors.
5. **Reduce selection bias** — label a sample of high-degree hub nodes so the
   model can learn from hard cases, not only easy ones.

---

## Key Learnings

- **Measure homophily before reaching for graph methods.** Low homophily ruled
  out label propagation and pointed to co-occurrence voting.
- **Account for selection bias** in how labels were sampled — it shaped both the
  method choice and how the metrics should be interpreted.
- **Macro vs weighted metrics tell different stories** under class imbalance;
  report both and explain the gap.

---

## License

MIT — see [LICENSE](LICENSE).
