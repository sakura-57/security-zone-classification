"""
# Tufin Home Assignment | Security Zone Tag Prediction
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import (classification_report, accuracy_score,  
                            f1_score, confusion_matrix,
                            precision_score, recall_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('output', exist_ok=True)

# ============================================================================
# 1. LOAD & CLEAN
# ============================================================================
rules = pd.read_csv('data/rules.csv')
tags = pd.read_csv('data/tags.csv')
tag_dict = tags.set_index('obj_id')['tag'].to_dict()

# Clean: remove self-loops and duplicate connections
rules = rules[rules['src_obj_id'] != rules['dst_obj_id']]
rules = rules.drop_duplicates(subset=['src_obj_id', 'dst_obj_id'])

all_objs = set(rules['src_obj_id']) | set(rules['dst_obj_id'])
labeled_objs = set(tags['obj_id'])
unlabeled_objs = all_objs - labeled_objs

print(f"Objects: {len(all_objs)} | Labeled: {len(labeled_objs)} | Unlabeled: {len(unlabeled_objs)}")
print(f"Rules: {len(rules)} | Unique rule IDs: {rules['rule_id'].nunique()}")

# ============================================================================
# 2. EDA
# ============================================================================
print("\n=== EDA ===")

# Tag distribution
print("\nTag distribution:")
tag_dist = tags['tag'].value_counts()
print(tag_dist.to_string())

# Flow matrix - what talks to what?
rules['src_tag'] = rules['src_obj_id'].map(tag_dict)
rules['dst_tag'] = rules['dst_obj_id'].map(tag_dict)
# Let's focus on rules where both sides are tagged 
# & remove where at least one side is missing a tag
both_tagged = rules.dropna(subset=['src_tag', 'dst_tag'])
flow = both_tagged.groupby(['src_tag', 'dst_tag']).size().unstack(fill_value=0)
print("\nFlow Matrix:")
print(flow)

intra = sum(flow.loc[t, t] for t in flow.index if t in flow.columns)
print(f"\nIntra-zone traffic: {intra}/{len(both_tagged)} ({intra/len(both_tagged)*100:.1f}%)")
print(f"Inter-zone traffic: {len(both_tagged)-intra}/{len(both_tagged)} ({(len(both_tagged)-intra)/len(both_tagged)*100:.1f}%)")

# Degree comparison | labeled vs unlabeled
import networkx as nx
G = nx.DiGraph()
G.add_edges_from(zip(rules['src_obj_id'], rules['dst_obj_id']))
degrees = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}
lab_deg = [degrees[n] for n in labeled_objs if n in degrees]
unlab_deg = [degrees[n] for n in unlabeled_objs if n in degrees]
print(f"\nLabeled mean degree: {np.mean(lab_deg):.1f}")
print(f"Unlabeled mean degree: {np.mean(unlab_deg):.1f}")
# top 5% most-connected nodes qualify as "hubs"
threshold = np.percentile(list(degrees.values()), 95)
hubs = {n for n, d in degrees.items() if d >= threshold}
print(f"Hub nodes (degree>={threshold:.0f}): {len(hubs)} | Labeled hubs: {len(hubs & labeled_objs)}")
# ============================================================================
# 3. METHOD: Rule Co-occurrence Voting
# ============================================================================
print("\n=== METHOD: Rule Co-occurrence Voting ===")

# Idea: An object is likely to have the same tag as other objects 
# that appear together with it in the same rules.

# Precompute: which objects appear together on each side of each rule
rule_to_srcs = rules.groupby('rule_id')['src_obj_id'].apply(set).to_dict()
rule_to_dsts = rules.groupby('rule_id')['dst_obj_id'].apply(set).to_dict()

obj_src_rules = defaultdict(set)
obj_dst_rules = defaultdict(set)
for _, row in rules.iterrows():
    obj_src_rules[row['src_obj_id']].add(row['rule_id'])
    obj_dst_rules[row['dst_obj_id']].add(row['rule_id'])

def predict(obj, lookup):
    """Majority vote from labeled co-members in shared rules."""
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
    winner, count = Counter(votes).most_common(1)[0]
    return winner, count / len(votes)

# ============================================================================
# 4. EVALUATION: Leave-one-out
# ============================================================================
print("\n=== EVALUATION (Leave-one-out) ===")

y_true, y_pred = [], []
for obj, true_tag in tag_dict.items():
    temp = {k: v for k, v in tag_dict.items() if k != obj}
    pred, _ = predict(obj, temp)
    if pred:
        y_true.append(true_tag)
        y_pred.append(pred)

print(f"\nAccuracy:     {accuracy_score(y_true, y_pred):.4f}")
print(f"F1 (macro):   {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"F1 (weighted):{f1_score(y_true, y_pred, average='weighted'):.4f}")
print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

# ============================================================================
# 5. PREDICT UNLABELED & SAVE
# ============================================================================
print("=== PREDICTIONS ===")

results = []
for obj in sorted(unlabeled_objs):
    pred, conf = predict(obj, tag_dict)
    if pred is None:
        pred, conf = tag_dist.index[0], 0.0
    results.append({'obj_id': obj, 'predicted_tag': pred, 'confidence': conf})

pred_df = pd.DataFrame(results)
pred_df[['obj_id', 'predicted_tag']].to_csv('predictions.csv', index=False)
print(f"Saved predictions.csv ({len(pred_df)} rows)")
print(f"\n{pred_df['predicted_tag'].value_counts().to_string()}")

# metrics.csv
acc = accuracy_score(y_true, y_pred)
pd.DataFrame([
    {'metric': 'accuracy', 'value': acc},
    {'metric': 'precision_per_tag', 'value': precision_score(y_true, y_pred, average='macro', zero_division=0)},
    {'metric': 'recall_per_tag', 'value': recall_score(y_true, y_pred, average='macro', zero_division=0)},
    {'metric': 'f1_per_tag', 'value': f1_score(y_true, y_pred, average='macro')},
    {'metric': 'f1_weighted', 'value': f1_score(y_true, y_pred, average='weighted')},
]).to_csv('metrics.csv', index=False)
print("Saved metrics.csv")