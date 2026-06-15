"""
Next Steps: Improvements to Co-occurrence Voting
=================================================
This file shows what I would add with more time.
Not part of the submitted solution — for interview discussion only.
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import networkx as nx

# Load data
rules = pd.read_csv('data/rules.csv')
tags = pd.read_csv('data/tags.csv')
tag_dict = tags.set_index('obj_id')['tag'].to_dict()
tag_dist = tags['tag'].value_counts()

rules = rules[rules['src_obj_id'] != rules['dst_obj_id']]

all_objs = set(rules['src_obj_id']) | set(rules['dst_obj_id'])
labeled_objs = set(tags['obj_id'])
unlabeled_objs = all_objs - labeled_objs

G = nx.DiGraph()
G.add_edges_from(zip(rules['src_obj_id'], rules['dst_obj_id']))

all_tags = sorted(tags['tag'].unique())

# ============================================================================
# IMPROVEMENT 1: Weighted Voting (fixes rare class problem)
# ============================================================================
print("=== IMPROVEMENT 1: Weighted Voting ===")

class_weights = {tag: 1.0 / count for tag, count in tag_dist.items()}

rule_to_srcs = rules.groupby('rule_id')['src_obj_id'].apply(set).to_dict()
rule_to_dsts = rules.groupby('rule_id')['dst_obj_id'].apply(set).to_dict()

obj_src_rules = defaultdict(set)
obj_dst_rules = defaultdict(set)
for _, row in rules.iterrows():
    obj_src_rules[row['src_obj_id']].add(row['rule_id'])
    obj_dst_rules[row['dst_obj_id']].add(row['rule_id'])

def predict_weighted(obj, lookup):
    """Weighted majority vote — rare classes count more."""
    votes = []
    for rid in obj_src_rules.get(obj, set()):
        for co in rule_to_srcs[rid]:
            if co != obj and co in lookup:
                votes.append(lookup[co])
    for rid in obj_dst_rules.get(obj, set()):
        for co in rule_to_dsts[rid]:
            if co != obj and co in lookup:
                votes.append(lookup[co])
    if not votes:
        return None, 0.0
    # Weighted counting
    weighted = defaultdict(float)
    for v in votes:
        weighted[v] += class_weights[v]
    winner = max(weighted, key=weighted.get)
    confidence = weighted[winner] / sum(weighted.values())
    return winner, confidence

# Evaluate weighted voting with leave-one-out
y_true_w, y_pred_w = [], []
for obj, true_tag in tag_dict.items():
    temp = {k: v for k, v in tag_dict.items() if k != obj}
    pred, _ = predict_weighted(obj, temp)
    if pred:
        y_true_w.append(true_tag)
        y_pred_w.append(pred)

print(f"Accuracy:   {accuracy_score(y_true_w, y_pred_w):.4f}")
print(f"F1 (macro): {f1_score(y_true_w, y_pred_w, average='macro'):.4f}")
print(f"\n{classification_report(y_true_w, y_pred_w, zero_division=0)}")

# ============================================================================
# IMPROVEMENT 2: Connection Profile Features
# ============================================================================
print("=== IMPROVEMENT 2: Connection Profile Features ===")

def build_profiles(G, tag_dict, all_tags):
    """Build connection profile for each object."""
    profiles = {}
    for obj in G.nodes():
        profile = {}
        # Outgoing: what tags does this object connect TO
        out_neighbors = list(G.successors(obj))
        out_tags = [tag_dict.get(n, 'unknown') for n in out_neighbors]
        out_counts = Counter(out_tags)
        out_total = max(len(out_tags), 1)
        for tag in all_tags:
            profile[f'out_{tag}'] = out_counts.get(tag, 0) / out_total

        # Incoming: what tags connect TO this object
        in_neighbors = list(G.predecessors(obj))
        in_tags = [tag_dict.get(n, 'unknown') for n in in_neighbors]
        in_counts = Counter(in_tags)
        in_total = max(len(in_tags), 1)
        for tag in all_tags:
            profile[f'in_{tag}'] = in_counts.get(tag, 0) / in_total

        # Degree features
        profile['in_degree'] = G.in_degree(obj)
        profile['out_degree'] = G.out_degree(obj)
        profile['total_degree'] = profile['in_degree'] + profile['out_degree']
        profile['degree_ratio'] = profile['in_degree'] / max(profile['out_degree'], 1)

        # Fraction of neighbors that are unknown
        profile['frac_unknown_out'] = out_counts.get('unknown', 0) / out_total
        profile['frac_unknown_in'] = in_counts.get('unknown', 0) / in_total

        profiles[obj] = profile
    return pd.DataFrame.from_dict(profiles, orient='index')

profile_df = build_profiles(G, tag_dict, all_tags)
print(f"Profile features: {profile_df.shape[1]} columns for {profile_df.shape[0]} objects")
print(f"Features: {list(profile_df.columns)}")

# ============================================================================
# IMPROVEMENT 3: Random Forest on Profile Features
# ============================================================================
print("\n=== IMPROVEMENT 3: Random Forest on Profiles ===")

feature_cols = list(profile_df.columns)

# Split labeled and unlabeled
labeled_profiles = profile_df.loc[profile_df.index.isin(labeled_objs)]
unlabeled_profiles = profile_df.loc[profile_df.index.isin(unlabeled_objs)]

X_train = labeled_profiles[feature_cols].values
y_train = np.array([tag_dict[obj] for obj in labeled_profiles.index])

X_unlabeled = unlabeled_profiles[feature_cols].values

# Train with class_weight='balanced' to handle imbalance
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Leave-one-out evaluation
from sklearn.model_selection import cross_val_predict, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_rf = cross_val_predict(rf, X_train, y_train, cv=skf)

print(f"Accuracy:   {accuracy_score(y_train, y_pred_rf):.4f}")
print(f"F1 (macro): {f1_score(y_train, y_pred_rf, average='macro'):.4f}")
print(f"\n{classification_report(y_train, y_pred_rf, zero_division=0)}")

# Feature importances
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Top 10 features:")
print(importances.head(10))

# ============================================================================
# IMPROVEMENT 4: Zero-Traffic Validation
# ============================================================================
print("\n=== IMPROVEMENT 4: Zero-Traffic Validation ===")

# Build flow matrix from labeled data
rules_tagged = rules.copy()
rules_tagged['src_tag'] = rules_tagged['src_obj_id'].map(tag_dict)
rules_tagged['dst_tag'] = rules_tagged['dst_obj_id'].map(tag_dict)
both_tagged = rules_tagged.dropna(subset=['src_tag', 'dst_tag'])
flow = both_tagged.groupby(['src_tag', 'dst_tag']).size().unstack(fill_value=0)

# Find forbidden pairs (zero traffic)
forbidden_pairs = set()
for src_tag in flow.index:
    for dst_tag in flow.columns:
        if flow.loc[src_tag, dst_tag] == 0:
            forbidden_pairs.add((src_tag, dst_tag))

print(f"Forbidden tag pairs (zero traffic): {len(forbidden_pairs)}")
for pair in sorted(forbidden_pairs):
    print(f"  {pair[0]} -> {pair[1]}")

# Check predictions against forbidden pairs
def validate_prediction(obj, predicted_tag, tag_dict, G, forbidden_pairs):
    """Check if predicted tag creates any forbidden connections."""
    violations = []
    # Check outgoing
    for neighbor in G.successors(obj):
        n_tag = tag_dict.get(neighbor)
        if n_tag and (predicted_tag, n_tag) in forbidden_pairs:
            violations.append(f"{predicted_tag} -> {n_tag}")
    # Check incoming
    for neighbor in G.predecessors(obj):
        n_tag = tag_dict.get(neighbor)
        if n_tag and (n_tag, predicted_tag) in forbidden_pairs:
            violations.append(f"{n_tag} -> {predicted_tag}")
    return violations

# Example: validate a few predictions
print("\nValidation example on first 100 predictions:")
pred_df = pd.read_csv('predictions.csv')
violation_count = 0
for _, row in pred_df.head(100).iterrows():
    v = validate_prediction(row['obj_id'], row['predicted_tag'], tag_dict, G, forbidden_pairs)
    if v:
        violation_count += 1
print(f"Predictions with violations: {violation_count}/100")

print("\nDONE")