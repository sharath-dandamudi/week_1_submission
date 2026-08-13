import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import PartialDependenceDisplay
import joblib

np.random.seed(42)

model = joblib.load("models/churn/model.pkl")
encoders = joblib.load("models/churn/encoders.pkl")
feature_cols = joblib.load("models/churn/feature_cols.pkl")
monotonicity = joblib.load("models/churn/monotonicity.pkl")

train_pool = pd.read_csv("/tmp/_churn_train_ref.csv")
for col, mapping in encoders.items():
    train_pool[col + "_enc"] = train_pool[col].map(mapping)

X = train_pool[feature_cols]
y = train_pool["churn_flag"]
X_dev, X_val, y_dev, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

holdout = pd.read_csv("sample_data/credit_card_churn_synthetic_v3_holdout_scoring_input.csv")
# holdout scoring input has no churn_flag (blind) -- pull true labels from the
# original v3 holdout file (not shipped with the app) purely to regenerate
# this reference plot, matching the same customers.
holdout_labeled = pd.read_csv("/home/claude/final_package/01_datasets/credit_card_churn_synthetic_v3_holdout.csv")
for col, mapping in encoders.items():
    holdout_labeled[col + "_enc"] = holdout_labeled[col].map(mapping)
X_holdout = holdout_labeled[feature_cols]
y_holdout = holdout_labeled["churn_flag"]

dev_probs = model.predict_proba(X_dev)[:, 1]
val_probs = model.predict_proba(X_val)[:, 1]
holdout_probs = model.predict_proba(X_holdout)[:, 1]

dev_auc = roc_auc_score(y_dev, dev_probs)
val_auc = roc_auc_score(y_val, val_probs)
holdout_auc = roc_auc_score(y_holdout, holdout_probs)
print(f"Dev AUC: {dev_auc:.4f}  Val AUC: {val_auc:.4f}  Holdout AUC: {holdout_auc:.4f}")

BLUE_LIGHT = "#4E9FE0"
BLUE_MID = "#00509E"
BLUE_DARK = "#0B3D66"

# ROC curve
fig, ax = plt.subplots(figsize=(7, 6))
for probs, labels, name, color in [
    (dev_probs, y_dev, f"Dev (AUC={dev_auc:.3f})", BLUE_LIGHT),
    (val_probs, y_val, f"Validation (AUC={val_auc:.3f})", BLUE_MID),
    (holdout_probs, y_holdout, f"Holdout (AUC={holdout_auc:.3f})", BLUE_DARK),
]:
    fpr, tpr, _ = roc_curve(labels, probs)
    ax.plot(fpr, tpr, label=name, linewidth=2.2, color=color)
ax.plot([0, 1], [0, 1], linestyle="--", color="#B0B8C0", alpha=0.8)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve: dev vs validation vs holdout", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150)
plt.close()

# Cumulative gain
def cumulative_gain(y_true, y_prob, n_bins=10):
    order = np.argsort(-y_prob)
    y_sorted = np.array(y_true)[order]
    n = len(y_sorted)
    total_positives = y_sorted.sum()
    pct_points = np.linspace(0, 1, n_bins + 1)
    gains = [0]
    for p in pct_points[1:]:
        k = int(np.ceil(p * n))
        gains.append(y_sorted[:k].sum() / total_positives)
    return pct_points * 100, np.array(gains) * 100

fig, ax = plt.subplots(figsize=(7, 6))
for probs, labels, name, color in [
    (dev_probs, y_dev.values, "Dev", BLUE_LIGHT),
    (val_probs, y_val.values, "Validation", BLUE_MID),
    (holdout_probs, y_holdout.values, "Holdout", BLUE_DARK),
]:
    pct, gain = cumulative_gain(labels, probs)
    ax.plot(pct, gain, marker="o", markersize=3, linewidth=2.2, label=name, color=color)
ax.plot([0, 100], [0, 100], linestyle="--", color="#B0B8C0", alpha=0.8, label="Random")
ax.set_xlabel("% of customers targeted (ranked by predicted risk)")
ax.set_ylabel("% of actual churners captured")
ax.set_title("Cumulative gain: dev vs validation vs holdout", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("reports/cumulative_gain.png", dpi=150)
plt.close()

# Partial dependence for top monotonic features
mono_features_by_importance = (
    pd.Series(model.feature_importances_, index=feature_cols)
    .loc[[f for f in feature_cols if monotonicity[f]["monotonic"]]]
    .sort_values(ascending=False)
)
top_features = list(mono_features_by_importance.index[:9])

friendly = {
    "total_trans_amt_last_3m": "spend_last_3_months", "total_trans_ct_last_3m": "transactions_last_3_months",
    "trans_amt_trend_ratio": "spend_trend", "avg_utilization_ratio": "credit_utilization",
    "months_inactive_12mo": "months_inactive", "mobile_app_logins_monthly": "app_logins_monthly",
    "digital_engagement_score": "digital_engagement", "autopay_enrolled": "autopay_enrolled",
    "contacts_last_12mo": "service_contacts", "complaint_flag_last_12mo": "complaint_filed",
    "rewards_redemption_rate": "rewards_redemption", "num_products_held": "products_held",
}
top_labels = [friendly.get(f, f) for f in top_features]

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
pdp_display = PartialDependenceDisplay.from_estimator(
    model, X_dev, top_features, kind="average", ax=axes.ravel()[:len(top_features)], n_cols=3
)
for ax, label in zip(pdp_display.axes_.ravel(), top_labels):
    if ax is not None:
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Predicted churn risk", fontsize=9)
        for line in ax.get_lines():
            line.set_color(BLUE_MID)
        ax.spines[['top', 'right']].set_visible(False)
fig.suptitle("Partial dependence — top monotonic features vs predicted churn risk", fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig("reports/partial_dependence.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: reports/roc_curve.png, reports/cumulative_gain.png, reports/partial_dependence.png")
