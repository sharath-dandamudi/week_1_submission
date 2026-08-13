"""
Core inference and scoring logic for the banker dashboard.

Deliberately kept independent of Streamlit so it can be tested and run as
plain Python (see test_logic.py). app.py imports these functions and only
handles UI/session-state wiring.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHURN_MODEL_DIR = os.path.join(BASE_DIR, "models", "churn")
LIFE_MOMENTS_MODEL_DIR = os.path.join(BASE_DIR, "models", "life_moments")
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data")

CHURN_SAMPLE_PATH = os.path.join(SAMPLE_DATA_DIR, "credit_card_churn_synthetic_v3_holdout_scoring_input.csv")
LIFE_MOMENTS_SAMPLE_PATH = os.path.join(SAMPLE_DATA_DIR, "life_moments_synthetic_holdout_scoring_input.csv")

# Risk tier thresholds are fixed, absolute probability cutoffs (not derived
# from whatever batch gets uploaded), so "Elevated" means the same thing
# every time the app is used.
RISK_TIER_LOW_MAX = 0.15
RISK_TIER_WATCH_MAX = 0.35

# Static demographic attributes excluded from the driver candidate pool
# regardless of monotonicity -- a banker has no lever to act on these.
NON_ACTIONABLE = {"age", "income_bracket_enc", "tenure_months"}

FRIENDLY_NAMES = {
    "age": "age", "tenure_months": "tenure_months",
    "income_bracket_enc": "income_bracket", "card_category_enc": "card_category",
    "num_products_held": "num_products_held", "credit_limit": "credit_limit",
    "total_trans_amt_last_3m": "spend_last_3_months",
    "total_trans_ct_last_3m": "transaction_count_last_3_months",
    "trans_amt_trend_ratio": "spend_trend_vs_prior_period",
    "avg_utilization_ratio": "credit_utilization_ratio",
    "months_inactive_12mo": "months_inactive_last_12mo",
    "mobile_app_logins_monthly": "mobile_app_logins_monthly",
    "digital_engagement_score": "digital_engagement_score",
    "autopay_enrolled": "autopay_enrolled",
    "contacts_last_12mo": "service_contacts_last_12mo",
    "complaint_flag_last_12mo": "complaint_filed_last_12mo",
    "rewards_redemption_rate": "rewards_redemption_rate"
}

DRIVER_PHRASES = {
    "spend_last_3_months": {
        "decreasing": lambda v: f"card spend has been lighter than usual lately, around ${v:,.0f} over the last three months",
        "increasing": lambda v: f"card spend has been unusually high, around ${v:,.0f} over the last three months",
    },
    "transaction_count_last_3_months": {
        "decreasing": lambda v: f"they've been using the card less often, about {v:,.0f} purchases over the last three months",
        "increasing": lambda v: f"they've been using the card more than usual, about {v:,.0f} purchases over the last three months",
    },
    "spend_trend_vs_prior_period": {
        "decreasing": lambda v: "their spending has been trending down compared to before",
        "increasing": lambda v: "their spending has been trending up compared to before",
    },
    "credit_utilization_ratio": {
        "increasing": lambda v: f"they're carrying a higher balance relative to their limit, around {v:.0%}",
        "decreasing": lambda v: f"they're carrying a notably low balance relative to their limit, around {v:.0%}",
    },
    "months_inactive_last_12mo": {
        "increasing": lambda v: f"the account has been quieter than usual, with little activity over about {v:.0f} of the last 12 months",
        "decreasing": lambda v: "the account has stayed consistently active",
    },
    "mobile_app_logins_monthly": {
        "decreasing": lambda v: f"they haven't been opening the app much, only around {v:.0f} times a month",
        "increasing": lambda v: f"they've been opening the app often, around {v:.0f} times a month",
    },
    "digital_engagement_score": {
        "decreasing": lambda v: "they haven't been as engaged with digital banking lately",
        "increasing": lambda v: "their digital engagement looks a little inconsistent lately",
    },
    "autopay_enrolled": {
        "decreasing": lambda v: "they haven't set up autopay",
        "increasing": lambda v: "their autopay status is worth a quick check",
    },
    "service_contacts_last_12mo": {
        "increasing": lambda v: f"they've contacted support {v:.0f} times this year",
        "decreasing": lambda v: "their support contact history is worth a quick check",
    },
    "complaint_filed_last_12mo": {
        "increasing": lambda v: "they filed a complaint earlier this year",
        "decreasing": lambda v: "their complaint history is worth a quick check",
    },
    "rewards_redemption_rate": {
        "decreasing": lambda v: f"they're not making much use of their rewards, only around {v:.0%}",
        "increasing": lambda v: f"they're making unusually heavy use of their rewards, around {v:.0%}",
    },
    "num_products_held": {
        "decreasing": lambda v: f"they currently just have {v:.0f} product(s) with us",
        "increasing": lambda v: "their product mix with us is worth a quick check",
    },
    "credit_limit": {
        "decreasing": lambda v: f"their credit limit is on the lower end, around ${v:,.0f}",
        "increasing": lambda v: f"their credit limit is on the higher end, around ${v:,.0f}",
    },
    "card_category": {
        "decreasing": lambda v: "their card tier is worth a quick look",
        "increasing": lambda v: "their card tier is worth a quick look",
    },
}

# ---------------------------------------------------------------------------
# Driver theme classification -- groups the 17 features into 3 buckets so
# both the short "insight tag" (Tab 2/3) and the full combined motive
# narrative (Tab 4) can reason about *what kind* of risk this is, not just
# which single feature ranked highest.
# ---------------------------------------------------------------------------
ENGAGEMENT_DECLINE_FEATURES = {
    "spend_last_3_months", "transaction_count_last_3_months", "spend_trend_vs_prior_period",
    "mobile_app_logins_monthly", "digital_engagement_score", "autopay_enrolled",
    "rewards_redemption_rate", "num_products_held", "months_inactive_last_12mo",
}
FINANCIAL_STRESS_FEATURES = {"credit_utilization_ratio"}
SERVICE_FRICTION_FEATURES = {"service_contacts_last_12mo", "complaint_filed_last_12mo"}

THEME_TAG_LABELS = {
    "engagement_decline": "Reduced engagement",
    "financial_stress": "Financial stress",
    "service_friction": "Service friction",
    "other": "General risk factors",
}

# Short, plain-language framing of what each of the 7 personas is likely
# focused on right now -- used to give the combined motive its life context.
PERSONA_CONTEXT = {
    "first_home_buyer": "in the process of buying their first home",
    "new_parent": "recently welcomed a new baby into their family",
    "career_changer": "going through a career change or return to study",
    "milestone_celebrator": "planning or celebrating a major life milestone",
    "home_mover": "in the middle of relocating to a new home",
    "frequent_traveler": "a frequent traveler",
    "retirement_prep": "approaching or actively planning for retirement",
}

# Short label used in "though that's likely separate from X" style sentences,
# where the full PERSONA_CONTEXT phrase wouldn't read grammatically.
PERSONA_SHORT_LABEL = {
    "first_home_buyer": "buying a first home",
    "new_parent": "having a new baby",
    "career_changer": "a career change",
    "milestone_celebrator": "an upcoming milestone",
    "home_mover": "a move",
    "frequent_traveler": "travel",
    "retirement_prep": "retirement planning",
}

# Which driver themes a given persona plausibly explains. service_friction is
# deliberately never included for any persona -- a complaint pattern is a
# real relationship problem regardless of what else is going on in someone's
# life, and softening it with life-stage context would be misleading.
PERSONA_EXPLAINS_THEME = {
    "first_home_buyer": {"engagement_decline", "financial_stress"},
    "new_parent": {"engagement_decline", "financial_stress"},
    "career_changer": {"engagement_decline", "financial_stress"},
    "milestone_celebrator": {"engagement_decline", "financial_stress"},
    "home_mover": {"engagement_decline", "financial_stress"},
    "frequent_traveler": {"engagement_decline"},
    "retirement_prep": {"engagement_decline"},
}

# Persona-specific, theme-specific action hooks -- this is what makes the
# recommended action genuinely marry the churn-risk theme with the life
# moment, rather than depending on risk tier and theme alone. 7 personas x
# 4 themes = 28 distinct hooks, each composed with a tier-based urgency
# prefix ("Worth a call to..." / "Prioritize a call to...").
ACTION_HOOKS = {
    "first_home_buyer": {
        "engagement_decline": "remind them of value they're not currently using, and check whether moving costs have temporarily pulled focus away from day-to-day banking",
        "financial_stress": "see if a temporary credit line increase or flexible payment plan would help smooth the costs of closing on their home",
        "service_friction": "confirm any issue from their mortgage or account setup process was fully resolved",
        "other": "check in and understand what's changed as they move through the home-buying process",
    },
    "new_parent": {
        "engagement_decline": "check in personally and remind them of value they're not currently using, many new parents simply have less time for banking right now",
        "financial_stress": "offer a temporary credit line increase or flexible payment option to help with new-baby costs",
        "service_friction": "confirm any recent service issue was fully resolved, sleep-deprived new parents have little patience for unresolved problems",
        "other": "check in and see how things are going now that they have a new addition to the family",
    },
    "career_changer": {
        "engagement_decline": "check in on how the transition is going, and highlight lower-fee or flexible products suited to a temporary income change",
        "financial_stress": "review their credit line and discuss flexible repayment options while their income is in transition",
        "service_friction": "confirm their recent service issue was fully resolved, don't let a bad experience compound an already stressful transition",
        "other": "check in on how their career change or return to study is going",
    },
    "milestone_celebrator": {
        "engagement_decline": "highlight short-term financing or rewards options suited to a big event, funds may simply be tied up in celebration costs",
        "financial_stress": "discuss short-term financing options for their upcoming celebration rather than letting utilization climb unchecked",
        "service_friction": "confirm any recent service issue was fully resolved before their big event",
        "other": "check in and see how planning for their upcoming milestone is going",
    },
    "home_mover": {
        "engagement_decline": "highlight moving-related offers, engagement often dips temporarily during a relocation",
        "financial_stress": "see if a short-term credit line increase would help cover moving costs without straining their utilization",
        "service_friction": "confirm any issue related to their move or account update was fully resolved",
        "other": "check in and see how the move is going",
    },
    "frequent_traveler": {
        "engagement_decline": "confirm this is genuinely just travel and not disengagement, and highlight travel perks like fee waivers or lounge access",
        "financial_stress": "review their credit line given their travel spend pattern, and flag any foreign transaction fee concerns",
        "service_friction": "confirm their recent service issue was fully resolved, frequent travelers have low tolerance for friction",
        "other": "check in and see if their travel plans have changed recently",
    },
    "retirement_prep": {
        "engagement_decline": "check in personally, since a drop this pronounced goes beyond the usual retirement slow-down, and confirm nothing else is going on",
        "financial_stress": "review their credit line, unusual for this life stage, so worth understanding what's driving it",
        "service_friction": "confirm their recent service issue was fully resolved, and use the call to discuss retirement-focused products",
        "other": "check in on how their retirement planning is going",
    },
}

# Short, human-sounding callback labels used when a secondary driver (from a
# different theme than the dominant one) gets an addendum appended to the
# recommended action.
SECONDARY_ADDRESS_LABELS = {
    "engagement_decline": "their recent drop in activity",
    "financial_stress": "their credit utilization",
    "service_friction": "their recent contact with support",
    "other": "the other factor noted above",
}

# Representative raw-feature columns shown in the Tab 1 data preview -- a
# genuine, uninterpreted subset of what the model actually sees, not a
# pre-summarized label that would give away what the model later predicts.
CHURN_PREVIEW_COLS = ["total_trans_amt_last_3m", "avg_utilization_ratio", "months_inactive_12mo", "digital_engagement_score"]
LIFE_MOMENTS_PREVIEW_COLS = ["travel_airfare_spend", "jewelry_spend", "baby_retail_spend", "mortgage_conveyancing_fees"]

# Human-readable labels for the Load Customers preview -- shown instead of
# raw column names.
PREVIEW_COL_LABELS = {
    "total_trans_amt_last_3m": "Spend (3 Months)",
    "avg_utilization_ratio": "Credit Utilization",
    "months_inactive_12mo": "Months Inactive",
    "digital_engagement_score": "Digital Engagement",
    "travel_airfare_spend": "Travel & Airfare Spend",
    "jewelry_spend": "Jewelry Spend",
    "baby_retail_spend": "Baby Retail Spend",
    "mortgage_conveyancing_fees": "Mortgage Fees",
}


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def load_churn_artifacts():
    return {
        "model": joblib.load(os.path.join(CHURN_MODEL_DIR, "model.pkl")),
        "encoders": joblib.load(os.path.join(CHURN_MODEL_DIR, "encoders.pkl")),
        "feature_cols": joblib.load(os.path.join(CHURN_MODEL_DIR, "feature_cols.pkl")),
        "baseline_values": joblib.load(os.path.join(CHURN_MODEL_DIR, "dev_baseline_values.pkl")),
        "monotonicity": joblib.load(os.path.join(CHURN_MODEL_DIR, "monotonicity.pkl")),
        "decile_cutpoints": joblib.load(os.path.join(CHURN_MODEL_DIR, "decile_cutpoints.pkl")),
    }


def load_life_moments_artifacts():
    return {
        "model": joblib.load(os.path.join(LIFE_MOMENTS_MODEL_DIR, "model.pkl")),
        "feature_cols": joblib.load(os.path.join(LIFE_MOMENTS_MODEL_DIR, "feature_cols.pkl")),
        "persona_descriptions": joblib.load(os.path.join(LIFE_MOMENTS_MODEL_DIR, "persona_descriptions.pkl")),
    }


def load_sample_churn_data():
    return pd.read_csv(CHURN_SAMPLE_PATH)


def load_sample_life_moments_data():
    return pd.read_csv(LIFE_MOMENTS_SAMPLE_PATH)


# ---------------------------------------------------------------------------
# Schema validation -- so a bad upload gives a clear message instead of a
# raw traceback
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Optional-field handling -- name/city/state are enrichments this app adds
# for readability, not part of the original model feature schema. Uploaded
# files that lack them should still work, falling back to the customer ID.
# ---------------------------------------------------------------------------
def _ensure_display_fields(df):
    df = df.copy()
    if "customer_name" not in df.columns:
        df["customer_name"] = df["customer_id"]
    if "city" not in df.columns:
        df["city"] = "—"
    if "state" not in df.columns:
        df["state"] = "—"
    return df


def validate_churn_schema(df, encoders):
    required_raw_cols = {"customer_id", "income_bracket", "card_category"}
    required_raw_cols |= {
        c for c in [
            "age", "tenure_months", "num_products_held", "credit_limit",
            "total_trans_amt_last_3m", "total_trans_ct_last_3m",
            "trans_amt_trend_ratio", "avg_utilization_ratio",
            "months_inactive_12mo", "mobile_app_logins_monthly",
            "digital_engagement_score", "autopay_enrolled",
            "contacts_last_12mo", "complaint_flag_last_12mo",
            "rewards_redemption_rate"
        ]
    }
    missing = sorted(required_raw_cols - set(df.columns))
    errors = []
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
        return errors

    for col, mapping in encoders.items():
        bad_values = sorted(set(df[col].dropna().unique()) - set(mapping.keys()))
        if bad_values:
            errors.append(
                f"Column '{col}' has unrecognized value(s): {', '.join(map(str, bad_values))} "
                f"(expected one of: {', '.join(mapping.keys())})"
            )
    if "customer_id" not in df.columns:
        errors.append("Missing required column: customer_id")
    return errors


def validate_life_moments_schema(df, feature_cols):
    required = set(feature_cols) | {"customer_id"}
    missing = sorted(required - set(df.columns))
    errors = []
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
    return errors


# ---------------------------------------------------------------------------
# Churn scoring
# ---------------------------------------------------------------------------
def assign_risk_tier(p):
    if p < RISK_TIER_LOW_MAX:
        return "Low risk"
    elif p < RISK_TIER_WATCH_MAX:
        return "Watch"
    else:
        return "Elevated"


def assign_risk_decile(probs, cutpoints):
    """Bucket against FIXED cutpoints from the training reference population,
    not a fresh qcut on this batch -- so decile 1 means the same absolute
    risk level regardless of what's uploaded."""
    bins = [-np.inf] + list(cutpoints) + [np.inf]
    labels = list(range(10, 0, -1))
    return np.array(pd.cut(probs, bins=bins, labels=labels)).astype(int)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _phrase_for_driver(driver):
    direction = driver["relationship_to_risk"].split(" ")[0]
    template = DRIVER_PHRASES.get(driver["feature"], {}).get(direction)
    if template:
        return template(driver["customer_value"])
    return f"{driver['feature'].replace('_', ' ')} is a contributing factor"


def _join_fragments(fragments):
    if len(fragments) == 1:
        return fragments[0]
    return ", ".join(fragments[:-1]) + f", and {fragments[-1]}"


def _get_driver_theme(friendly_feature_name):
    if friendly_feature_name in ENGAGEMENT_DECLINE_FEATURES:
        return "engagement_decline"
    if friendly_feature_name in FINANCIAL_STRESS_FEATURES:
        return "financial_stress"
    if friendly_feature_name in SERVICE_FRICTION_FEATURES:
        return "service_friction"
    return "other"


def score_churn(df, artifacts):
    model = artifacts["model"]
    encoders = artifacts["encoders"]
    feature_cols = artifacts["feature_cols"]
    baseline_values = artifacts["baseline_values"]
    monotonicity = artifacts["monotonicity"]
    decile_cutpoints = artifacts["decile_cutpoints"]

    df = _ensure_display_fields(df).reset_index(drop=True)
    for col, mapping in encoders.items():
        df[col + "_enc"] = df[col].map(mapping)

    X = df[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    base_logit = _logit(probs)
    contributions = pd.DataFrame(index=X.index, columns=feature_cols, dtype=float)
    for feat in feature_cols:
        X_ablated = X.copy()
        X_ablated[feat] = baseline_values[feat]
        ablated_logit = _logit(model.predict_proba(X_ablated)[:, 1])
        contributions[feat] = base_logit - ablated_logit

    def get_top_drivers(row_idx, top_n=3):
        row = contributions.loc[row_idx]
        candidates = row.drop(labels=NON_ACTIONABLE, errors="ignore")
        candidates = candidates[candidates > 0]
        mono_candidates = candidates[[f for f in candidates.index if monotonicity[f]["monotonic"]]]
        mono_candidates = mono_candidates.sort_values(ascending=False)
        chosen = list(mono_candidates.index[:top_n])
        if len(chosen) < top_n:
            remaining = candidates.drop(labels=chosen, errors="ignore").sort_values(ascending=False)
            chosen += list(remaining.index[: top_n - len(chosen)])
        drivers = []
        for feat in chosen:
            drivers.append({
                "feature": FRIENDLY_NAMES[feat],
                "customer_value": round(float(X.loc[row_idx, feat]), 3),
                "contribution_to_churn_risk": round(float(row[feat]), 3),
                "relationship_to_risk": monotonicity[feat]["direction"] + (
                    "" if monotonicity[feat]["monotonic"] else " (weakly monotonic)"
                )
            })
        return drivers

    top_drivers_list = [get_top_drivers(i) for i in X.index]
    risk_tier = [assign_risk_tier(p) for p in probs]
    risk_decile = assign_risk_decile(probs, decile_cutpoints)

    # Insight tag: the theme of the single dominant driver, computable from
    # churn data alone (no persona needed yet) -- shown in Tab 2/3 as a
    # quick scannable tag, ahead of the full narrative in Tab 4.
    insight_tag = []
    for drivers in top_drivers_list:
        if not drivers:
            insight_tag.append("No notable drivers")
        else:
            theme = _get_driver_theme(drivers[0]["feature"])
            insight_tag.append(THEME_TAG_LABELS[theme])

    out = df.copy()
    out["churn_probability"] = probs.round(4)
    out["churn_prediction"] = preds
    out["risk_tier"] = risk_tier
    out["risk_decile"] = risk_decile
    out["top_drivers"] = top_drivers_list
    out["top_drivers_json"] = [json.dumps(d) for d in top_drivers_list]
    out["insight_tag"] = insight_tag
    if "tenure_months" in out.columns:
        out["tenure_years"] = (out["tenure_months"] / 12).round(1)
    return out


# ---------------------------------------------------------------------------
# Life moments scoring
# ---------------------------------------------------------------------------
def score_life_moments(df, artifacts):
    model = artifacts["model"]
    feature_cols = artifacts["feature_cols"]
    persona_descriptions = artifacts["persona_descriptions"]

    df = _ensure_display_fields(df).reset_index(drop=True)
    X = df[feature_cols]
    preds = model.predict(X)
    probs = model.predict_proba(X)
    confidence = probs.max(axis=1)

    out = df.copy()
    out["predicted_life_moment"] = preds
    out["prediction_confidence"] = confidence.round(3)
    out["persona_description"] = out["predicted_life_moment"].map(persona_descriptions)
    return out


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Combined motive + recommended action -- the core "why might they leave,
# and what should I do" narrative. Combines the churn driver evidence with
# the life-moment persona context. Deliberately deterministic (no live LLM
# call) so it's free and reproducible; see generate_talking_point_llm below
# for an optional per-customer AI upgrade.
#
# Design: the persona provides plausible life context for engagement decline
# or financial stress (a new parent has less banking time; someone buying a
# first home has one-off costs). Service friction is never softened by
# persona -- a complaint pattern is a real relationship problem regardless
# of what else is going on in someone's life.
# ---------------------------------------------------------------------------
def generate_motive_and_action(risk_tier, drivers, persona):
    themes = [_get_driver_theme(d["feature"]) for d in drivers]
    dominant_theme = themes[0] if themes else "other"
    persona_context = PERSONA_CONTEXT.get(persona, "")
    persona_short = PERSONA_SHORT_LABEL.get(persona, persona)
    explains = dominant_theme in PERSONA_EXPLAINS_THEME.get(persona, set())

    same_theme_drivers = [d for d, t in zip(drivers, themes) if t == dominant_theme]
    diff_theme_drivers = [d for d, t in zip(drivers, themes) if t != dominant_theme]
    joined_same = _join_fragments([_phrase_for_driver(d) for d in same_theme_drivers]) if same_theme_drivers else ""
    joined_all = _join_fragments([_phrase_for_driver(d) for d in drivers]) if drivers else ""

    if risk_tier == "Low risk":
        if not drivers:
            motive = "Low risk overall, with no notable risk signals."
        elif persona_context:
            motive = f"Low risk overall. They're {persona_context}, and {joined_all}. Nothing here needs attention."
        else:
            motive = f"Low risk overall. The only soft spots are {joined_all}. Nothing here needs attention."
        action = "No action needed beyond routine relationship maintenance."
        return motive, action

    # Motive: if the persona plausibly explains the dominant theme, lead with
    # that life context. Only add a "that said" caveat when there's a real
    # secondary driver from a genuinely different theme -- never a generic,
    # customer-independent hedge.
    if explains and dominant_theme != "service_friction" and joined_same:
        motive = f"This customer is {persona_context}, which may partly explain why {joined_same}."
        if diff_theme_drivers:
            secondary_phrase = _phrase_for_driver(diff_theme_drivers[0])
            motive += f" That said, {secondary_phrase}, though that's likely separate from {persona_short}."
    else:
        motive = joined_all[0].upper() + joined_all[1:] + "." if joined_all else "No notable risk drivers identified."
        if persona_context:
            if explains:
                motive += f" They're also {persona_context}, which may explain some of this, but the pattern above is worth its own conversation."
            else:
                motive += f" They're also {persona_context}, though that doesn't obviously explain the pattern above."

    # Action: persona-specific hook for the dominant theme, composed with
    # tier urgency, plus a short addendum if a genuinely different secondary
    # theme is present -- this is what makes the action reflect the full
    # picture, not just the single top driver.
    urgency = "Prioritize a call to" if risk_tier == "Elevated" else "Worth a call to"
    hook = ACTION_HOOKS.get(persona, {}).get(dominant_theme, "check in and understand what's changed")
    action = f"{urgency} {hook}."
    if diff_theme_drivers:
        secondary_theme = _get_driver_theme(diff_theme_drivers[0]["feature"])
        action += f" Also worth addressing {SECONDARY_ADDRESS_LABELS[secondary_theme]} while you're at it."

    return motive, action


def combine_outputs(churn_scored, life_moments_scored):
    churn_ids = set(churn_scored["customer_id"])
    lm_ids = set(life_moments_scored["customer_id"])
    matched = churn_ids & lm_ids

    combined = churn_scored.merge(
        life_moments_scored[["customer_id", "predicted_life_moment", "prediction_confidence", "persona_description"]],
        on="customer_id", how="inner"
    )

    motives, actions = [], []
    for _, row in combined.iterrows():
        motive, action = generate_motive_and_action(row["risk_tier"], row["top_drivers"], row["predicted_life_moment"])
        motives.append(motive)
        actions.append(action)
    combined["motive_text"] = motives
    combined["recommended_action"] = actions

    match_stats = {
        "churn_customers": len(churn_ids),
        "life_moments_customers": len(lm_ids),
        "matched": len(matched),
    }
    return combined, match_stats


# ---------------------------------------------------------------------------
# Chart insights -- one short, computed sentence per Customer Profile
# Builder chart, so the page reads as insight rather than a raw report.
# ---------------------------------------------------------------------------
def get_risk_breakdown_insight(combined):
    n = len(combined)
    if n == 0:
        return "No customers to summarize yet."
    counts = combined["risk_tier"].value_counts()
    low_pct = counts.get("Low risk", 0) / n * 100
    watch_pct = counts.get("Watch", 0) / n * 100
    elevated_pct = counts.get("Elevated", 0) / n * 100
    return f"{low_pct:.0f}% of customers are Low Risk, {watch_pct:.0f}% Watch, and {elevated_pct:.0f}% Elevated."


def get_life_moment_distribution_insight(combined):
    if len(combined) == 0:
        return "No customers to summarize yet."
    counts = combined["predicted_life_moment"].value_counts()
    top_persona = counts.index[0]
    top_label = top_persona.replace("_", " ").title()
    return f"{top_label} is the most common life stage, at {counts.iloc[0]:,} customers."


def get_risk_by_life_moment_insight(combined):
    if len(combined) == 0:
        return "No customers to summarize yet."
    personas = combined["predicted_life_moment"].unique()
    best_persona, best_share = None, -1
    for p in personas:
        subset = combined[combined["predicted_life_moment"] == p]
        share = (subset["risk_tier"] == "Elevated").mean()
        if share > best_share:
            best_persona, best_share = p, share
    label = best_persona.replace("_", " ").title()
    return f"{label} shows the highest concentration of Elevated risk ({best_share:.0%}), worth prioritizing outreach here."


# ---------------------------------------------------------------------------
# Optional LLM upgrade for a single customer (used on-demand in the
# dashboard detail view, not run automatically for the whole batch)
# ---------------------------------------------------------------------------
LLM_SYSTEM_PROMPT = """You are a customer retention assistant that translates a \
structured churn-risk model output into a short, plain-language briefing \
for a bank relationship manager.

Rules:
- Use ONLY the drivers provided in the JSON. Do not introduce any other \
reasons, additional features, or details about the customer's identity or \
history beyond what is given.
- Write 2-3 sentences: (1) a plain-language summary of why this customer is \
at this risk level, grounded in the specific driver values given, and (2) \
for Watch or Elevated tiers, one concrete, specific next step the banker \
could take. For Low risk, state plainly that no action is needed.
- Keep it professional and free of jargon -- never mention "SHAP", \
"ablation", "log-odds", "monotonic", "the model", or any modeling \
internals.
- Do not open with the customer ID or restate the risk tier label verbatim.
"""


def generate_talking_point_llm(api_key, risk_tier, probability, drivers, persona=None, model="claude-haiku-4-5-20251001"):
    """Requires the `anthropic` package and a valid API key. Raises on
    failure -- callers should catch and fall back to the template version."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    payload = {
        "risk_tier": risk_tier,
        "churn_probability": probability,
        "top_drivers": drivers,
    }
    if persona:
        payload["life_moment_persona"] = persona
    response = client.messages.create(
        model=model,
        max_tokens=200,
        system=LLM_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


# ---------------------------------------------------------------------------
# Display formatting -- pure functions, no Streamlit dependency, so they can
# be unit tested directly (see test_logic.py). app.py imports these rather
# than duplicating them, and uses html_table() in place of st.dataframe for
# every reference table, since Streamlit's native data tables render onto an
# HTML canvas (glide-data-grid) that no CSS -- including font rules -- can
# reach. Building tables as plain HTML keeps font, comma formatting, decimal
# precision, and color-coded badges fully under our control.
# ---------------------------------------------------------------------------
def fmt_currency(v):
    return f"${v:,.0f}"


def fmt_int(v):
    return f"{int(round(v)):,}"


def fmt_decimal2(v):
    return f"{v:.2f}"


def fmt_decimal1(v):
    return f"{v:.1f}"


def fmt_percent0(v):
    return f"{v:.0%}"


def fmt_title(v):
    return str(v).replace("_", " ").title()


def fmt_text(v):
    return str(v)


def html_table(df, specs):
    """specs: list of (label, render_fn) where render_fn(row) -> cell HTML."""
    thead = "".join(f"<th>{label}</th>" for label, _ in specs)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{fn(row)}</td>" for _, fn in specs)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f'<table class="htmltable"><thead><tr>{thead}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )


DRIVER_VALUE_TYPE = {
    "spend_last_3_months": "currency", "credit_limit": "currency",
    "credit_utilization_ratio": "percent", "rewards_redemption_rate": "percent",
    "transaction_count_last_3_months": "int", "months_inactive_last_12mo": "int",
    "mobile_app_logins_monthly": "int", "service_contacts_last_12mo": "int",
    "num_products_held": "int",
    "autopay_enrolled": "bool", "complaint_filed_last_12mo": "bool",
    "digital_engagement_score": "decimal1", "spend_trend_vs_prior_period": "decimal2",
}


def fmt_driver_value(feature, value):
    t = DRIVER_VALUE_TYPE.get(feature, "text")
    if t == "bool":
        return "Yes" if value >= 0.5 else "No"
    if t == "currency":
        return fmt_currency(value)
    if t == "percent":
        return fmt_percent0(value)
    if t == "int":
        return fmt_int(value)
    if t == "decimal1":
        return fmt_decimal1(value)
    if t == "decimal2":
        return fmt_decimal2(value)
    return fmt_text(value)


def risk_badge(tier):
    cls = {"Low risk": "badge-low", "Watch": "badge-watch", "Elevated": "badge-elevated"}.get(tier, "badge-low")
    return f'<span class="badge {cls}">{tier}</span>'
