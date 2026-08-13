"""
Simulates every data-transformation and table-rendering step app.py
performs, using logic.py directly. Doesn't test Streamlit rendering itself
(can't, no install here), but catches every KeyError/formatting bug that
would otherwise only surface at runtime in the browser.
"""
import logic
import pandas as pd

def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        global failures
        failures += 1

failures = 0

churn_artifacts = logic.load_churn_artifacts()
lm_artifacts = logic.load_life_moments_artifacts()

churn_raw = logic.load_sample_churn_data()
lm_raw = logic.load_sample_life_moments_data()
churn_scored = logic.score_churn(churn_raw, churn_artifacts)
lm_scored = logic.score_life_moments(lm_raw, lm_artifacts)
combined, match_stats = logic.combine_outputs(churn_scored, lm_scored)

# --- TAB 1: Load Customers preview tables ---
print("=" * 60, "\nTab 1: Load Customers previews\n" + "=" * 60)
preview_df = logic._ensure_display_fields(churn_raw).head(5)
preview_specs = [("Customer", lambda r: logic.fmt_text(r["customer_name"])),
                  ("ID", lambda r: logic.fmt_text(r["customer_id"]))]
for c in logic.CHURN_PREVIEW_COLS:
    label = logic.PREVIEW_COL_LABELS.get(c, c)
    if c == "total_trans_amt_last_3m":
        fn = lambda r, c=c: logic.fmt_currency(r[c])
    elif c == "avg_utilization_ratio":
        fn = lambda r, c=c: logic.fmt_percent0(r[c])
    elif c == "months_inactive_12mo":
        fn = lambda r, c=c: logic.fmt_int(r[c])
    else:
        fn = lambda r, c=c: logic.fmt_decimal1(r[c])
    preview_specs.append((label, fn))
churn_preview_html = logic.html_table(preview_df, preview_specs)
check("Tab1 churn preview table renders", churn_preview_html.startswith("<table"))
check("Tab1 churn preview uses friendly labels, not raw column names",
      "Spend (3 Months)" in churn_preview_html and "total_trans_amt_last_3m" not in churn_preview_html)

lm_preview_df = logic._ensure_display_fields(lm_raw).head(5)
lm_preview_specs = [("Customer", lambda r: logic.fmt_text(r["customer_name"])),
                     ("ID", lambda r: logic.fmt_text(r["customer_id"]))]
for c in logic.LIFE_MOMENTS_PREVIEW_COLS:
    label = logic.PREVIEW_COL_LABELS.get(c, c)
    lm_preview_specs.append((label, lambda r, c=c: logic.fmt_currency(r[c])))
lm_preview_html = logic.html_table(lm_preview_df, lm_preview_specs)
check("Tab1 life moments preview table renders", lm_preview_html.startswith("<table"))
check("Tab1 life moments preview uses friendly labels",
      "Travel & Airfare Spend" in lm_preview_html)

# --- TAB 2: Generate Predictions tables ---
print("=" * 60, "\nTab 2: Generate Predictions tables\n" + "=" * 60)
churn_specs = [
    ("Customer", lambda r: logic.fmt_text(r["customer_name"])),
    ("ID", lambda r: logic.fmt_text(r["customer_id"])),
    ("Churn Risk", lambda r: logic.fmt_decimal2(r["churn_probability"])),
    ("Risk Level", lambda r: logic.risk_badge(r["risk_tier"])),
    ("Risk Rank", lambda r: f"{logic.fmt_int(r['risk_decile'])} / 10"),
    ("Customer Insight", lambda r: logic.fmt_text(r["insight_tag"])),
]
tab2_churn_html = logic.html_table(churn_scored.head(10), churn_specs)
check("Tab2 churn table renders", tab2_churn_html.startswith("<table"))
check("Tab2 churn table has 10 data rows", tab2_churn_html.count("<tr>") == 11)
check("Tab2 churn risk shown as 2-decimal value, not raw 4-decimal", "0.6800" not in tab2_churn_html)

lm_specs = [
    ("Customer", lambda r: logic.fmt_text(r["customer_name"])),
    ("ID", lambda r: logic.fmt_text(r["customer_id"])),
    ("Life Stage", lambda r: logic.fmt_title(r["predicted_life_moment"])),
    ("Confidence", lambda r: logic.fmt_decimal2(r["prediction_confidence"])),
]
tab2_lm_html = logic.html_table(lm_scored.head(10), lm_specs)
check("Tab2 life moments table renders", tab2_lm_html.startswith("<table"))
check("Tab2 life stage shown in title case", "Frequent Traveler" in tab2_lm_html or "First Home Buyer" in tab2_lm_html)

avg_risk_pct = f"{churn_scored['churn_probability'].mean():.0%}"
check("Avg churn risk stat is a percentage string", avg_risk_pct.endswith("%"))

# --- TAB 3: Customer Profile Builder table + chart insights ---
print("=" * 60, "\nTab 3: Customer Profile Builder\n" + "=" * 60)
glance_specs = [
    ("Customer", lambda r: logic.fmt_text(r["customer_name"])),
    ("ID", lambda r: logic.fmt_text(r["customer_id"])),
    ("Churn Risk", lambda r: logic.fmt_decimal2(r["churn_probability"])),
    ("Risk Level", lambda r: logic.risk_badge(r["risk_tier"])),
    ("Risk Rank", lambda r: f"{logic.fmt_int(r['risk_decile'])} / 10"),
    ("Life Moment", lambda r: logic.fmt_title(r["predicted_life_moment"])),
    ("Customer Insight", lambda r: logic.fmt_text(r["insight_tag"])),
]
tab3_html = logic.html_table(combined.head(10), glance_specs)
check("Tab3 at-a-glance table renders", tab3_html.startswith("<table"))

insight1 = logic.get_risk_breakdown_insight(combined)
insight2 = logic.get_life_moment_distribution_insight(combined)
insight3 = logic.get_risk_by_life_moment_insight(combined)
check("all three Tab3 chart insights are non-empty strings", all(len(i) > 0 for i in [insight1, insight2, insight3]))

# --- TAB 4: Customer Action Center ---
print("=" * 60, "\nTab 4: Customer Action Center\n" + "=" * 60)
combined_sorted = combined.sort_values("churn_probability", ascending=False).reset_index(drop=True)
option_map = {f"{row.customer_name} — {row.risk_tier} ({row.customer_id})": row.customer_id for row in combined_sorted.itertuples()}
check("option_map built for every customer, no label collisions", len(option_map) == len(combined_sorted))

selected_id = combined_sorted.iloc[0]["customer_id"]
row = combined[combined["customer_id"] == selected_id].iloc[0]

profile_html = (
    f"Age: {logic.fmt_int(row['age'])}, Tenure: {row['tenure_years']:.1f} years, "
    f"Products: {logic.fmt_int(row['num_products_held'])}"
)
check("profile strip fields format without error", "Age:" in profile_html)

drivers = row["top_drivers"]
if drivers:
    driver_df = pd.DataFrame(drivers)
    driver_specs = [
        ("Feature", lambda r: logic.fmt_title(r["feature"])),
        ("Customer value", lambda r: logic.fmt_driver_value(r["feature"], r["customer_value"])),
        ("Relationship to risk", lambda r: logic.fmt_title(r["relationship_to_risk"])),
    ]
    driver_html = logic.html_table(driver_df, driver_specs)
    check("Tab4 driver detail table renders", driver_html.startswith("<table"))

for tier_filter, expect_all in [("Elevated", False), (None, True)]:
    queue = combined if expect_all else combined[combined["risk_tier"] == "Elevated"]
    queue_specs = [
        ("Customer", lambda r: logic.fmt_text(r["customer_name"])),
        ("ID", lambda r: logic.fmt_text(r["customer_id"])),
        ("Churn Risk", lambda r: logic.fmt_decimal2(r["churn_probability"])),
        ("Risk Level", lambda r: logic.risk_badge(r["risk_tier"])),
    ]
    queue_html = logic.html_table(queue.head(50), queue_specs)
    check(f"Tab4 queue table renders (show_all={expect_all})", queue_html.startswith("<table"))
    if not expect_all:
        check("Elevated-only queue excludes Watch/Low risk badges",
              "badge-watch" not in queue_html and "badge-low" not in queue_html)

# --- Uploaded-data-without-enrichment fallback ---
print("=" * 60, "\nUpload fallback (no name/city/state)\n" + "=" * 60)
stripped = churn_raw.drop(columns=["customer_name", "city", "state"])
stripped_scored = logic.score_churn(stripped, churn_artifacts)
check("scoring works on data without name/city/state", "customer_name" in stripped_scored.columns)
stripped_html = logic.html_table(stripped_scored.head(3), churn_specs)
check("html_table renders fine even with fallback customer_name (= customer_id)",
      stripped_html.startswith("<table"))

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else str(failures) + ' CHECK(S) FAILED'}")
