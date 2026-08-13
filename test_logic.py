"""
Plain-Python tests for logic.py -- no Streamlit required. Run with:
    python3 test_logic.py
"""

import sys
import pandas as pd
import logic


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        global failures
        failures += 1


failures = 0

print("=" * 60)
print("Loading artifacts")
print("=" * 60)
churn_artifacts = logic.load_churn_artifacts()
lm_artifacts = logic.load_life_moments_artifacts()
check("churn artifacts loaded", all(k in churn_artifacts for k in
      ["model", "encoders", "feature_cols", "baseline_values", "monotonicity", "decile_cutpoints"]))
check("life moments artifacts loaded", all(k in lm_artifacts for k in
      ["model", "feature_cols", "persona_descriptions"]))
check("decile cutpoints has 9 boundaries", len(churn_artifacts["decile_cutpoints"]) == 9)

print("\n" + "=" * 60)
print("Loading sample data")
print("=" * 60)
churn_df = logic.load_sample_churn_data()
lm_df = logic.load_sample_life_moments_data()
check("churn sample has 999 rows", len(churn_df) == 999)
check("life moments sample has 999 rows", len(lm_df) == 999)
check("churn sample has no churn_flag column (blind scoring input)", "churn_flag" not in churn_df.columns)

print("\n" + "=" * 60)
print("Schema validation")
print("=" * 60)
errors = logic.validate_churn_schema(churn_df, churn_artifacts["encoders"])
check("valid churn data passes validation with no errors", len(errors) == 0)

bad_df = churn_df.drop(columns=["avg_utilization_ratio"])
errors = logic.validate_churn_schema(bad_df, churn_artifacts["encoders"])
check("missing column is caught", len(errors) > 0 and "avg_utilization_ratio" in errors[0])

bad_category_df = churn_df.copy()
bad_category_df.loc[0, "card_category"] = "Diamond"
errors = logic.validate_churn_schema(bad_category_df, churn_artifacts["encoders"])
check("unrecognized category value is caught", any("Diamond" in e for e in errors))

lm_errors = logic.validate_life_moments_schema(lm_df, lm_artifacts["feature_cols"])
check("valid life moments data passes validation", len(lm_errors) == 0)

print("\n" + "=" * 60)
print("Churn scoring")
print("=" * 60)
churn_scored = logic.score_churn(churn_df, churn_artifacts)
check("churn_probability in valid range", churn_scored["churn_probability"].between(0, 1).all())
check("risk_tier has only expected values", set(churn_scored["risk_tier"].unique()) <= {"Low risk", "Watch", "Elevated"})
check("risk_decile in range 1-10", churn_scored["risk_decile"].between(1, 10).all())
import json as json_module
check("top_drivers_json is valid JSON for every row",
      all(isinstance(json_module.loads(j), list) for j in churn_scored["top_drivers_json"]))
check("every row has an insight_tag",
      churn_scored["insight_tag"].str.len().gt(0).all())
check("insight_tag values are from the expected set",
      set(churn_scored["insight_tag"].unique()) <= set(logic.THEME_TAG_LABELS.values()) | {"No notable drivers"})
check("tenure_years computed correctly",
      (churn_scored["tenure_years"] == (churn_scored["tenure_months"] / 12).round(1)).all())
check("customer_name column present (from sample data)",
      "customer_name" in churn_scored.columns and churn_scored["customer_name"].str.len().gt(0).all())
check("decile distribution is reasonably even (min bucket >= 5% of rows)",
      churn_scored["risk_decile"].value_counts().min() >= 0.05 * len(churn_scored))

print("\nSample scored row:")
print(churn_scored[["customer_id", "churn_probability", "risk_tier", "risk_decile"]].head(3).to_string(index=False))
print("\nSample insight tag:")
print(churn_scored.iloc[0]["insight_tag"])

print("\nRisk tier distribution:")
print(churn_scored["risk_tier"].value_counts())

print("\n" + "=" * 60)
print("Life moments scoring")
print("=" * 60)
lm_scored = logic.score_life_moments(lm_df, lm_artifacts)
check("predicted_life_moment has 7 expected personas",
      set(lm_scored["predicted_life_moment"].unique()) == {
          "first_home_buyer", "new_parent", "career_changer", "milestone_celebrator",
          "home_mover", "frequent_traveler", "retirement_prep"
      })
check("prediction_confidence in valid range", lm_scored["prediction_confidence"].between(0, 1).all())
check("persona_description is non-empty for every row", lm_scored["persona_description"].str.len().gt(0).all())

print("\n" + "=" * 60)
print("Combine outputs")
print("=" * 60)
combined, match_stats = logic.combine_outputs(churn_scored, lm_scored)
print("Match stats:", match_stats)
check("all customers matched (same source population)", match_stats["matched"] == 999)
check("combined has both churn and persona columns",
      "churn_probability" in combined.columns and "predicted_life_moment" in combined.columns)
check("combined has motive_text and recommended_action for every row",
      combined["motive_text"].str.len().gt(0).all() and combined["recommended_action"].str.len().gt(0).all())
check("Low risk customers get the routine-maintenance action",
      (combined.loc[combined["risk_tier"] == "Low risk", "recommended_action"]
       == "No action needed beyond routine relationship maintenance.").all())

# Simulate a mismatched join
churn_subset = churn_scored.head(500)
combined2, match_stats2 = logic.combine_outputs(churn_subset, lm_scored)
check("mismatched batch sizes reported correctly", match_stats2["matched"] == 500)

print("\n" + "=" * 60)
print("Combined motive generation -- service friction is never softened by persona")
print("=" * 60)
# A customer whose #1 driver is service friction should never get the
# "which may partly explain" softened framing, regardless of persona.
service_friction_drivers = [{
    "feature": "service_contacts_last_12mo", "customer_value": 7.0,
    "contribution_to_churn_risk": 1.3, "relationship_to_risk": "increasing"
}]
for persona in logic.PERSONA_CONTEXT:
    motive, action = logic.generate_motive_and_action("Watch", service_friction_drivers, persona)
    check(f"service friction not softened for persona={persona}",
          "which may partly explain" not in motive)

print("\nSample real combined record:")
sample_row = combined[combined["risk_tier"] == "Elevated"].iloc[0]
print(f"{sample_row['customer_name']} ({sample_row['customer_id']}) | {sample_row['risk_tier']} | {sample_row['predicted_life_moment']}")
print("Motive:", sample_row["motive_text"])
print("Action:", sample_row["recommended_action"])

import re as re_module
bad_caps = combined[combined["motive_text"].str.contains(r"\bis A[a-z]|\bis [A-Z][a-z]+ [a-z]", regex=True, na=False)]
check("no mid-sentence capitalization artifacts in motive text", len(bad_caps) == 0)

print("\n" + "=" * 60)
print("Hyper-personalized actions -- same theme+tier, different persona")
print("=" * 60)
service_friction_drivers = [{
    "feature": "service_contacts_last_12mo", "customer_value": 7.0,
    "contribution_to_churn_risk": 1.3, "relationship_to_risk": "increasing"
}]
actions_by_persona = {}
for persona in logic.PERSONA_CONTEXT:
    _, action = logic.generate_motive_and_action("Watch", service_friction_drivers, persona)
    actions_by_persona[persona] = action
check("recommended actions genuinely differ across personas (same theme+tier)",
      len(set(actions_by_persona.values())) > 1)
print(f"{len(set(actions_by_persona.values()))} / {len(actions_by_persona)} distinct actions across personas")

check("old static 'bigger than life stage' caveat sentence is gone",
      not combined["motive_text"].str.contains("bigger than their current life stage", na=False).any())

print("\n" + "=" * 60)
print("Secondary-driver addendum -- only appears when a genuinely different theme is present")
print("=" * 60)
single_theme_drivers = [
    {"feature": "digital_engagement_score", "customer_value": 1.0, "contribution_to_churn_risk": 0.5, "relationship_to_risk": "decreasing"},
    {"feature": "mobile_app_logins_monthly", "customer_value": 1.0, "contribution_to_churn_risk": 0.4, "relationship_to_risk": "decreasing"},
]
motive_single, action_single = logic.generate_motive_and_action("Elevated", single_theme_drivers, "frequent_traveler")
check("no 'also worth addressing' addendum when all drivers share one theme",
      "Also worth addressing" not in action_single)
check("no 'that said' caveat when all drivers share one theme",
      "That said" not in motive_single)

mixed_drivers = [
    {"feature": "digital_engagement_score", "customer_value": 1.0, "contribution_to_churn_risk": 0.5, "relationship_to_risk": "decreasing"},
    {"feature": "credit_utilization_ratio", "customer_value": 0.8, "contribution_to_churn_risk": 0.4, "relationship_to_risk": "increasing"},
]
motive_mixed, action_mixed = logic.generate_motive_and_action("Elevated", mixed_drivers, "frequent_traveler")
check("'also worth addressing' addendum appears when a genuinely different theme is present",
      "Also worth addressing" in action_mixed)
check("'that said' caveat appears when a genuinely different theme is present",
      "That said" in motive_mixed)

print("\n" + "=" * 60)
print("Chart insight helpers")
print("=" * 60)
insight1 = logic.get_risk_breakdown_insight(combined)
insight2 = logic.get_life_moment_distribution_insight(combined)
insight3 = logic.get_risk_by_life_moment_insight(combined)
print("Risk breakdown:", insight1)
print("Life moment distribution:", insight2)
print("Risk by life moment:", insight3)
check("risk breakdown insight mentions all three tiers", all(t in insight1 for t in ["Low Risk", "Watch", "Elevated"]))
check("life moment distribution insight is non-empty", len(insight2) > 0)
check("risk by life moment insight is non-empty", len(insight3) > 0)

print("\n" + "=" * 60)
print("Display formatting helpers")
print("=" * 60)
check("fmt_currency adds comma separators", logic.fmt_currency(1234567) == "$1,234,567")
check("fmt_int adds comma separators", logic.fmt_int(1234567) == "1,234,567")
check("fmt_decimal2 caps to 2 places", logic.fmt_decimal2(0.6789) == "0.68")
check("fmt_decimal1 caps to 1 place", logic.fmt_decimal1(3.789) == "3.8")
check("fmt_percent0 formats as whole percent", logic.fmt_percent0(0.647) == "65%")
check("fmt_title converts snake_case", logic.fmt_title("frequent_traveler") == "Frequent Traveler")
check("fmt_int does not add a decimal to whole numbers", "." not in logic.fmt_int(7))
check("fmt_driver_value formats currency-type features with $ and commas",
      logic.fmt_driver_value("spend_last_3_months", 456789) == "$456,789")
check("fmt_driver_value formats percent-type features",
      logic.fmt_driver_value("credit_utilization_ratio", 0.647) == "65%")
check("fmt_driver_value formats int-type features with commas, no decimal",
      logic.fmt_driver_value("service_contacts_last_12mo", 7) == "7")
check("fmt_driver_value formats bool-type features as Yes/No",
      logic.fmt_driver_value("autopay_enrolled", 0) == "No" and logic.fmt_driver_value("autopay_enrolled", 1) == "Yes")

print("\n" + "=" * 60)
print("html_table renders correctly")
print("=" * 60)
sample_df = combined.head(5)
specs = [
    ("Customer", lambda r: logic.fmt_text(r["customer_name"])),
    ("Churn Risk", lambda r: logic.fmt_decimal2(r["churn_probability"])),
    ("Risk Level", lambda r: logic.risk_badge(r["risk_tier"])),
]
table_html = logic.html_table(sample_df, specs)
check("html_table produces a <table> tag", table_html.startswith("<table"))
check("html_table includes all column headers", all(f"<th>{label}</th>" in table_html for label, _ in specs))
check("html_table produces exactly 5 data rows", table_html.count("<tr>") == 6)  # 1 header + 5 data rows
check("html_table risk badges render with correct CSS classes", "badge-elevated" in table_html or "badge-watch" in table_html or "badge-low" in table_html)

print("\n" + "=" * 60)
print("Optional-field fallback (uploaded data without name/city/state)")
print("=" * 60)
stripped_df = churn_df.drop(columns=["customer_name", "city", "state"], errors="ignore")
stripped_scored = logic.score_churn(stripped_df, churn_artifacts)
check("customer_name falls back to customer_id when missing",
      (stripped_scored["customer_name"] == stripped_scored["customer_id"]).all())
check("city/state fall back to placeholder when missing",
      (stripped_scored["city"] == "—").all() and (stripped_scored["state"] == "—").all())

print("\n" + "=" * 60)
if failures == 0:
    print("ALL CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
    sys.exit(1)
