EMBER_BASE_URL?=
LIMIT?=50000
OPS_Q?=0.2
IMPACT?=5000000

.PHONY: dirs download download_full extract extract_full labels train eval register plots logs ci all

dirs:
	mkdir -p data/ember data/ember2024 artifacts notebooks logs mappings src .github/workflows

download:
	python src/download_ember.py --outdir data/ember --datasets train test --limit $(LIMIT) --force --base-url $(EMBER_BASE_URL)

extract:
	python src/feature_extractor.py --input data/ember/train/ember_features.jsonl --out data/ember/train_pe.npz --limit $(LIMIT) --force

download_full:
	python - <<END
import os
import thrember
os.makedirs("data/ember2024", exist_ok=True)
thrember.download_dataset("data/ember2024")
END

extract_full:
	python src/feature_extractor.py --input data/ember2024/train/ember_features.jsonl --out data/ember/train_pe.npz --limit $(LIMIT) --force

labels:
	python - <<END
import pandas as pd, numpy as np, json
features = pd.read_json('data/ember2024/train/ember_features.jsonl', lines=True)
with open('mappings/attack_mapping.json') as f:
    attack_mapping = json.load(f)
ransomware_families = [fam for fam, techniques in attack_mapping.items() if 'T1486' in techniques]
features['is_ransomware'] = features['family'].apply(lambda f: 1 if f in ransomware_families else 0)
X = features.drop(columns=['family','is_ransomware'])
y = features['is_ransomware']
np.savez('data/ember/train_pe_labels.npz', X=X.values, y=y.values)
END

train:
	python src/train_lightgbm.py --features data/ember/train_pe.npz --labels data/ember/train_pe_labels.npz --mapping mappings/family_mapping.json --outdir artifacts --bag-seeds 42 43 44 --calibration isotonic --robust-loss focal

eval:
	python src/evaluate.py --predictions artifacts/predictions.npz --ops-quantile $(OPS_Q) --out metrics_report.json --time-ordered-split --metrics auroc pr_auc brier ece lift confusion --fn_pref_weight 10

register:
	python src/policy_writer.py --predictions artifacts/predictions.npz --features data/ember/train_pe.npz --labels data/ember/train_pe_labels.npz --mapping mappings/family_mapping.json --attack_mapping mappings/attack_mapping.json --d3fend_graph mappings/d3fend_graph.json --impact $(IMPACT) --out artifacts/cyber_risk_advisor.json --csv artifacts/cyber_risk_advisor.csv --policy-json artifacts/policy.json --risk_buckets High Medium Low --attach_controls

plots:
	python - <<END
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve

ns = np.load('artifacts/predictions.npz', allow_pickle=True)
y_true = ns['val_labels']
y_probs = ns['val_probs']

# AUROC
fpr, tpr, _ = roc_curve(y_true, y_probs)
auroc = auc(fpr, tpr)
print(f"AUROC: {auroc:.4f}")

# PR-AUC
precision, recall, _ = precision_recall_curve(y_true, y_probs)
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.4f}")

# Brier Score
brier = brier_score_loss(y_true, y_probs)
print(f"Brier Score: {brier:.4f}")

# ECE
prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
ece = np.abs(prob_true - prob_pred).mean()
print(f"ECE: {ece:.4f}")

# Lift@10%
k = int(len(y_probs) * 0.1)
top_k_idx = np.argsort(y_probs)[-k:]
lift = y_true[top_k_idx].mean() / y_true.mean()
print(f"Lift@10%: {lift:.4f}")

# Confusion Matrix at ops threshold
ops_threshold = 0.738
y_pred = (y_probs >= ops_threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix (TN, FP; FN, TP):")
print(cm)

# Save plots
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'AUROC={auroc:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Ransomware AUROC'); plt.legend()
plt.savefig('artifacts/auroc_curve.png')
plt.close()

plt.figure(figsize=(6,6))
plt.plot(recall, precision, label=f'PR-AUC={pr_auc:.3f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Ransomware PR Curve'); plt.legend()
plt.savefig('artifacts/pr_curve.png')
plt.close()
END

logs:
	python src/log_ingest.py --logs logs/ --out data/normalized_logs.npz --clean --merge

ci:
	@echo "GitHub Actions CI configured in .github/workflows/lint.yml"

all: dirs download_full extract_full labels train eval register plots logs
	@echo "AICRA build complete → Train → Evaluate → Optimize Threshold → Generate Register"