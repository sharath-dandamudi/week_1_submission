"""
Fair Dinkum Bank — Customer Retention Dashboard
=============================================
Streamlit demo app. Business logic lives in logic.py (tested independently,
see test_logic.py) -- this file only handles UI and session state.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logic

st.set_page_config(
    page_title="Fair Dinkum Bank — Customer Retention Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Theming. IMPORTANT: font-family is applied to .stApp only (no universal
# `*` selector, no `!important`). An earlier version used
# `.stApp, .stApp * { font-family: ... !important; }`, which also overrode
# the icon fonts Streamlit uses for expander arrows and the select dropdown
# chevron, causing them to render as distorted boxes. Letting the rule
# cascade naturally fixes that without touching real text.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    font-family: Arial, Helvetica, 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #EEF3F9;
    border-right: 1px solid #DCE6F0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00509E !important;
    border-bottom-color: #00509E !important;
}
[data-testid="stTabs"] button p {
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: #10243D !important;
}
button[kind="primary"] {
    background-color: #00509E !important;
    border-color: #00509E !important;
}
button[kind="primary"]:hover {
    background-color: #003868 !important;
    border-color: #003868 !important;
}
/* Dropdown visibility -- explicit white background and a visible blue-tinted
   border so it clearly reads as a dropdown rather than blending into the page. */
[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF;
    border: 1px solid #9DBEDC;
    border-radius: 8px;
}
.topbanner {
    background: #003868;
    color: #FFFFFF;
    padding: 18px 26px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.topbanner .brandrow { display: flex; align-items: center; gap: 14px; }
.topbanner .brandmark {
    min-width: 38px; height: 38px; border-radius: 9px; padding: 0 8px;
    background: #1E88E5;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px; color: #FFFFFF; letter-spacing: 0.5px;
    flex-shrink: 0;
}
.topbanner h1 { font-size: 18px; font-weight: 600; margin: 0; color: #FFFFFF; }
.topbanner .subtitle { font-size: 12.5px; color: #AFCBEA; margin-top: 2px; }
.protobadge {
    font-size: 11px; font-weight: 500; color: #003868;
    background: #FFD873; padding: 5px 13px; border-radius: 20px;
    white-space: nowrap;
}
.section-header { font-size: 18px; font-weight: 600; margin: 4px 0 10px; color: #10243D; }
.subsection-header { font-size: 14.5px; font-weight: 600; margin: 16px 0 8px; color: #10243D; }
.badge { display:inline-block; padding:4px 12px; border-radius:14px; font-size:12.5px; font-weight:600; white-space:nowrap; }
.badge-low { background:#D6EAFB; color:#0B4A82; }
.badge-watch { background:#4E9FE0; color:#FFFFFF; }
.badge-elevated { background:#0B3D66; color:#FFFFFF; }
.profile-strip { display:flex; flex-wrap:wrap; gap:26px; padding:6px 0 2px; }
.profile-item .k { font-size:11px; color:#8CA3BA; margin-bottom:2px; }
.profile-item .v { font-size:15px; font-weight:600; color:#10243D; }
.motive-card { border:1px solid #DCE6F0; border-left:4px solid #00509E; border-radius:12px; padding:18px 20px; background:#FFFFFF; }
.motive-text { font-size:14.5px; line-height:1.7; color:#10243D; margin:0 0 12px; }
.action-box { font-size:14px; line-height:1.6; background:#EEF3F9; border-radius:8px; padding:12px 14px; color:#10243D; }
.action-box b { color:#003868; }
.caveat-box { font-size:12.5px; color:#56708C; background:#EEF3F9; border-radius:8px; padding:12px 16px; font-style:italic; line-height:1.6; }
.chart-insight { font-size:12.5px; color:#56708C; margin: -6px 0 4px; line-height:1.5; }
.statcard-row { display:flex; gap:12px; margin-bottom:6px; }
.statcard { flex:1; border-radius:10px; padding:14px 16px; }
.statcard.neutral { background:#EEF3F9; border:1px solid #DCE6F0; }
.statcard.elevated { background:#0B3D66; }
.statcard.watch { background:#4E9FE0; }
.statcard.low { background:#D6EAFB; }
.statcard .label { font-size:11.5px; margin-bottom:4px; }
.statcard .value { font-size:22px; font-weight:700; }
.statcard.neutral .label, .statcard.neutral .value { color:#10243D; }
.statcard.elevated .label, .statcard.elevated .value { color:#FFFFFF; }
.statcard.watch .label, .statcard.watch .value { color:#FFFFFF; }
.statcard.low .label, .statcard.low .value { color:#0B4A82; }
table.htmltable { width:100%; border-collapse:collapse; font-size:13px; }
table.htmltable th {
    text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.03em;
    color:#8CA3BA; font-weight:600; padding:8px 10px; border-bottom:1px solid #DCE6F0;
}
table.htmltable td { padding:9px 10px; border-bottom:1px solid #EEF3F9; color:#10243D; }
table.htmltable tr:last-child td { border-bottom:none; }
</style>
""", unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def subsection_header(text):
    st.markdown(f'<div class="subsection-header">{text}</div>', unsafe_allow_html=True)


# Formatting helpers and html_table() now live in logic.py (pure functions,
# no Streamlit dependency, so they're directly unit tested there).
risk_badge = logic.risk_badge
html_table = logic.html_table
fmt_currency = logic.fmt_currency
fmt_int = logic.fmt_int
fmt_decimal2 = logic.fmt_decimal2
fmt_decimal1 = logic.fmt_decimal1
fmt_percent0 = logic.fmt_percent0
fmt_title = logic.fmt_title
fmt_text = logic.fmt_text
fmt_driver_value = logic.fmt_driver_value


# ---------------------------------------------------------------------------
# Cached artifact loading (models are static -- load once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_churn_artifacts():
    return logic.load_churn_artifacts()


@st.cache_resource
def get_lm_artifacts():
    return logic.load_life_moments_artifacts()


churn_artifacts = get_churn_artifacts()
lm_artifacts = get_lm_artifacts()


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in {
    "churn_raw_df": None,
    "lm_raw_df": None,
    "churn_scored_df": None,
    "lm_scored_df": None,
    "combined_df": None,
    "match_stats": None,
    "api_key": "",
    "notes": {},
    "feedback": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Fair Dinkum Bank")
    st.caption("Predicting who's at risk, and why, before the call")
    st.divider()

    st.subheader("Pipeline status")
    st.write(("✅" if st.session_state.churn_raw_df is not None else "⬜") + " Customers loaded")
    st.write(("✅" if st.session_state.churn_scored_df is not None else "⬜") + " Predictions generated")
    st.write(("✅" if st.session_state.combined_df is not None else "⬜") + " Profiles combined")

    st.divider()
    st.subheader("Optional: AI-generated insight")
    st.caption(
        "By default, insights are generated with a free, deterministic model — "
        "no API key needed. Add a key here to enable an AI-generated version "
        "per customer in the Customer Action Center."
    )
    st.session_state.api_key = st.text_input(
        "Anthropic API key", type="password", value=st.session_state.api_key
    )


st.markdown("""
<div class="topbanner">
  <div class="brandrow">
    <div class="brandmark">FDB</div>
    <div>
      <h1>Fair Dinkum Bank — Customer Retention Dashboard</h1>
      <div class="subtitle">Predicting who's at risk, and why, before the call</div>
    </div>
  </div>
  <div class="protobadge">Prototype &middot; synthetic data</div>
</div>
""", unsafe_allow_html=True)

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Load Customers",
    "Generate Predictions",
    "Customer Profile Builder",
    "Customer Action Center",
])


# ---------------------------------------------------------------------------
# TAB 0 — Overview
# ---------------------------------------------------------------------------
with tab0:
    st.markdown(
        '<div class="caveat-box">Fair Dinkum Bank is a fictitious company, and all data used in this '
        'prototype is synthetic. The business problem itself is real and product agnostic. '
        'Fragmented, reactive retention outreach with limited context for the frontline team is '
        'a pain point most consumer banks and subscription businesses face.</div>',
        unsafe_allow_html=True
    )

    subsection_header("Situation")
    st.markdown("Fair Dinkum Bank is seeing significant credit card churn, running at **roughly 20%**.")

    subsection_header("Problem")
    st.markdown(
        "Retention marketing today is **fragmented and reactive**. Leads often reach bankers "
        "**too late to retain** the customer, and with little context on why they're at risk, "
        "calls end up **feeling like cold calling** rather than a relationship conversation."
    )

    subsection_header("What this solution does")
    st.markdown("""
- **Predicts** each customer's likelihood to churn
- **Explains why**, in plain language, not a black box score
- Identifies **what's going on in a customer's life** right now, like a move or a new job
- That context shapes how retention should be approached
- Combines both into a **single, ready-to-use briefing** for the banker
- The conversation starts from understanding, not a cold call
- Blends traditional **machine learning** (binary classification for churn risk, segmentation for life stage) with **generative AI** to turn predictions into plain-language insight
- Looking ahead, this can evolve into a full **agentic workflow** (e.g. retrieval-augmented generation, more autonomous next-best-action logic)
- A natural next step once those concepts are covered later in the bootcamp
""")

    subsection_header("How we will measure success")
    st.markdown("""
- Higher **banker action rates** on the leads they're given
- Higher actual **customer retention** among flagged at-risk customers
- **Fewer false positives** over time
- Powered by banker feedback on lead quality, a built-in feedback loop
""")


# ---------------------------------------------------------------------------
# TAB 1 — Load Customers
# ---------------------------------------------------------------------------
with tab1:
    section_header("Get started")
    st.caption(
        "We've already set up the prediction engine behind the scenes. All you need to do here "
        "is choose which customers to look at. Sample data is loaded automatically so you can "
        "start exploring right away."
    )

    col1, col2 = st.columns(2)

    with col1:
        subsection_header("Credit card activity")
        churn_source = st.radio(
            "Data source", ["Use sample data", "Upload my own CSV"],
            key="churn_source", horizontal=True
        )
        if churn_source == "Use sample data":
            df = logic.load_sample_churn_data()
            errors = logic.validate_churn_schema(df, churn_artifacts["encoders"])
            if errors:
                st.error("Sample data failed validation:\n\n" + "\n".join(errors))
            else:
                st.session_state.churn_raw_df = df
                st.success(f"Loaded {len(df):,} customers from sample data.")
        else:
            uploaded = st.file_uploader("Churn customer CSV", type=["csv"], key="churn_upload")
            if uploaded is not None:
                df = pd.read_csv(uploaded)
                errors = logic.validate_churn_schema(df, churn_artifacts["encoders"])
                if errors:
                    st.error("Upload failed validation:\n\n" + "\n".join(f"- {e}" for e in errors))
                    st.session_state.churn_raw_df = None
                else:
                    st.session_state.churn_raw_df = df
                    st.success(f"Loaded {len(df):,} customers.")

        if st.session_state.churn_raw_df is not None:
            preview_df = logic._ensure_display_fields(st.session_state.churn_raw_df).head(5)
            preview_specs = [
                ("Customer", lambda r: fmt_text(r["customer_name"])),
                ("ID", lambda r: fmt_text(r["customer_id"])),
            ]
            for c in logic.CHURN_PREVIEW_COLS:
                if c in preview_df.columns:
                    label = logic.PREVIEW_COL_LABELS.get(c, c)
                    if c == "total_trans_amt_last_3m":
                        fn = lambda r, c=c: fmt_currency(r[c])
                    elif c == "avg_utilization_ratio":
                        fn = lambda r, c=c: fmt_percent0(r[c])
                    elif c == "months_inactive_12mo":
                        fn = lambda r, c=c: fmt_int(r[c])
                    else:
                        fn = lambda r, c=c: fmt_decimal1(r[c])
                    preview_specs.append((label, fn))
            st.markdown(html_table(preview_df, preview_specs), unsafe_allow_html=True)
            n_more = len(logic._ensure_display_fields(st.session_state.churn_raw_df).columns) - 2 - len(logic.CHURN_PREVIEW_COLS)
            st.caption(f"+{n_more} more columns used by the model, not shown here.")

    with col2:
        subsection_header("Lifestyle signals")
        lm_source = st.radio(
            "Data source", ["Use sample data", "Upload my own CSV"],
            key="lm_source", horizontal=True
        )
        if lm_source == "Use sample data":
            df = logic.load_sample_life_moments_data()
            errors = logic.validate_life_moments_schema(df, lm_artifacts["feature_cols"])
            if errors:
                st.error("Sample data failed validation:\n\n" + "\n".join(errors))
            else:
                st.session_state.lm_raw_df = df
                st.success(f"Loaded {len(df):,} customers from sample data.")
        else:
            uploaded = st.file_uploader("Life moments customer CSV", type=["csv"], key="lm_upload")
            if uploaded is not None:
                df = pd.read_csv(uploaded)
                errors = logic.validate_life_moments_schema(df, lm_artifacts["feature_cols"])
                if errors:
                    st.error("Upload failed validation:\n\n" + "\n".join(f"- {e}" for e in errors))
                    st.session_state.lm_raw_df = None
                else:
                    st.session_state.lm_raw_df = df
                    st.success(f"Loaded {len(df):,} customers.")

        if st.session_state.lm_raw_df is not None:
            preview_df = logic._ensure_display_fields(st.session_state.lm_raw_df).head(5)
            preview_specs = [
                ("Customer", lambda r: fmt_text(r["customer_name"])),
                ("ID", lambda r: fmt_text(r["customer_id"])),
            ]
            for c in logic.LIFE_MOMENTS_PREVIEW_COLS:
                if c in preview_df.columns:
                    label = logic.PREVIEW_COL_LABELS.get(c, c)
                    preview_specs.append((label, lambda r, c=c: fmt_currency(r[c])))
            st.markdown(html_table(preview_df, preview_specs), unsafe_allow_html=True)
            n_more = len(logic._ensure_display_fields(st.session_state.lm_raw_df).columns) - 2 - len(logic.LIFE_MOMENTS_PREVIEW_COLS)
            st.caption(f"+{n_more} more columns used by the model, not shown here.")


# ---------------------------------------------------------------------------
# TAB 2 — Generate Predictions
# ---------------------------------------------------------------------------
with tab2:
    section_header("Generate predictions")

    if st.session_state.churn_raw_df is None or st.session_state.lm_raw_df is None:
        st.warning("Complete the Load Customers tab first — both datasets need to be loaded.")
    else:
        st.caption(
            "We'll generate two sets of predictions: how likely each customer is to churn, "
            "and what life stage they're in right now, such as buying a home or planning retirement."
        )
        if st.button("Generate predictions", type="primary"):
            with st.spinner("Scoring churn risk..."):
                st.session_state.churn_scored_df = logic.score_churn(
                    st.session_state.churn_raw_df, churn_artifacts
                )
            with st.spinner("Identifying life stage..."):
                st.session_state.lm_scored_df = logic.score_life_moments(
                    st.session_state.lm_raw_df, lm_artifacts
                )
            st.session_state.combined_df = None
            st.session_state.match_stats = None
            st.success("Predictions generated.")

        if st.session_state.churn_scored_df is not None:
            scored = st.session_state.churn_scored_df
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers scored", f"{len(scored):,}")
            c2.metric("Avg. churn risk", f"{scored['churn_probability'].mean():.0%}")
            c3.metric("Elevated", f"{(scored['risk_tier'] == 'Elevated').sum():,}")
            c4.metric("Low risk", f"{(scored['risk_tier'] == 'Low risk').sum():,}")

            subsection_header("Who's at risk")
            churn_specs = [
                ("Customer", lambda r: fmt_text(r["customer_name"])),
                ("ID", lambda r: fmt_text(r["customer_id"])),
                ("Churn Risk", lambda r: fmt_decimal2(r["churn_probability"])),
                ("Risk Level", lambda r: risk_badge(r["risk_tier"])),
                ("Risk Rank", lambda r: f"{fmt_int(r['risk_decile'])} / 10"),
                ("Customer Insight", lambda r: fmt_text(r["insight_tag"])),
            ]
            st.markdown(html_table(scored.head(10), churn_specs), unsafe_allow_html=True)

        if st.session_state.lm_scored_df is not None:
            lm_scored = st.session_state.lm_scored_df
            subsection_header("What's going on with them")
            lm_specs = [
                ("Customer", lambda r: fmt_text(r["customer_name"])),
                ("ID", lambda r: fmt_text(r["customer_id"])),
                ("Life Stage", lambda r: fmt_title(r["predicted_life_moment"])),
                ("Confidence", lambda r: fmt_decimal2(r["prediction_confidence"])),
            ]
            st.markdown(html_table(lm_scored.head(10), lm_specs), unsafe_allow_html=True)

        if st.session_state.churn_scored_df is not None:
            subsection_header("How risk level and risk rank are determined")
            st.caption(
                "Risk Level compares each customer's churn probability to the bank's overall "
                "average churn rate (about 20%). Customers well below average are Low Risk, "
                "those near or modestly above it are Watch, and those significantly above it "
                "are Elevated."
            )
            st.caption(
                "Risk Rank ranks each customer from 1 to 10 relative to the full customer base, "
                "where 1 is the riskiest 10% and 10 is the safest 10%. It's meant for "
                "prioritizing outreach when you can only act on a limited number of leads at a time."
            )


# ---------------------------------------------------------------------------
# TAB 3 — Customer Profile Builder
# ---------------------------------------------------------------------------
with tab3:
    section_header("Customer profile builder")

    if st.session_state.churn_scored_df is None or st.session_state.lm_scored_df is None:
        st.warning("Complete the Generate Predictions tab first — both models need to be scored.")
    else:
        if st.button("Combine churn + life moments profiles", type="primary"):
            combined, match_stats = logic.combine_outputs(
                st.session_state.churn_scored_df, st.session_state.lm_scored_df
            )
            st.session_state.combined_df = combined
            st.session_state.match_stats = match_stats

        if st.session_state.combined_df is not None:
            combined = st.session_state.combined_df
            stats = st.session_state.match_stats

            c1, c2, c3 = st.columns(3)
            c1.metric("Churn customers", f"{stats['churn_customers']:,}")
            c2.metric("Life moments customers", f"{stats['life_moments_customers']:,}")
            c3.metric("Matched", f"{stats['matched']:,}")
            if stats["matched"] < min(stats["churn_customers"], stats["life_moments_customers"]):
                st.warning(
                    f"Only {stats['matched']:,} of the uploaded customers appear in both "
                    "datasets. The views below include matched customers only."
                )

            BLUE_LOW, BLUE_WATCH, BLUE_ELEVATED = "#D6EAFB", "#4E9FE0", "#0B3D66"

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                subsection_header("Customer risk breakdown")
                tier_counts = combined["risk_tier"].value_counts()
                order = ["Low risk", "Watch", "Elevated"]
                values = [tier_counts.get(t, 0) for t in order]
                fig = go.Figure(data=[go.Pie(
                    labels=order, values=values, hole=0.55,
                    marker=dict(colors=[BLUE_LOW, BLUE_WATCH, BLUE_ELEVATED]),
                    textinfo="label+percent",
                )])
                fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f'<div class="chart-insight">{logic.get_risk_breakdown_insight(combined)}</div>', unsafe_allow_html=True)

            with chart_col2:
                subsection_header("Life moment distribution")
                persona_counts = combined["predicted_life_moment"].value_counts().sort_values()
                labels = [p.replace("_", " ").title() for p in persona_counts.index]
                fig = go.Figure(data=[go.Bar(
                    x=persona_counts.values, y=labels, orientation="h",
                    marker=dict(color=BLUE_WATCH),
                )])
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300,
                                   xaxis_title="Customers", yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f'<div class="chart-insight">{logic.get_life_moment_distribution_insight(combined)}</div>', unsafe_allow_html=True)

            subsection_header("Risk by life moment")
            st.caption("Which life moments correlate with higher churn risk, at a glance.")
            personas = sorted(combined["predicted_life_moment"].unique())
            persona_labels = [p.replace("_", " ").title() for p in personas]
            fig = go.Figure()
            for tier, color in [("Low risk", BLUE_LOW), ("Watch", BLUE_WATCH), ("Elevated", BLUE_ELEVATED)]:
                shares = []
                for p in personas:
                    subset = combined[combined["predicted_life_moment"] == p]
                    shares.append((subset["risk_tier"] == tier).mean() * 100 if len(subset) else 0)
                fig.add_trace(go.Bar(
                    y=persona_labels, x=shares, name=tier, orientation="h",
                    marker=dict(color=color),
                ))
            fig.update_layout(
                barmode="stack", height=300, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Share of customers (%)", legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f'<div class="chart-insight">{logic.get_risk_by_life_moment_insight(combined)}</div>', unsafe_allow_html=True)

            subsection_header("At-a-glance customer view")
            glance_specs = [
                ("Customer", lambda r: fmt_text(r["customer_name"])),
                ("ID", lambda r: fmt_text(r["customer_id"])),
                ("Churn Risk", lambda r: fmt_decimal2(r["churn_probability"])),
                ("Risk Level", lambda r: risk_badge(r["risk_tier"])),
                ("Risk Rank", lambda r: f"{fmt_int(r['risk_decile'])} / 10"),
                ("Life Moment", lambda r: fmt_title(r["predicted_life_moment"])),
                ("Customer Insight", lambda r: fmt_text(r["insight_tag"])),
            ]
            st.markdown(html_table(combined.head(10), glance_specs), unsafe_allow_html=True)

            with st.expander("Model performance"):
                st.caption(
                    "Dev AUC 0.894 · Validation AUC 0.847 · Holdout AUC 0.874 — "
                    "the holdout score matching or beating validation confirms the model "
                    "generalizes rather than overfitting to the training data."
                )
                perf_col1, perf_col2 = st.columns(2)
                with perf_col1:
                    st.image("reports/roc_curve.png", use_container_width=True)
                with perf_col2:
                    st.image("reports/cumulative_gain.png", use_container_width=True)
                st.image("reports/partial_dependence.png", use_container_width=True)
                st.caption(
                    "Partial dependence plots show the shape and direction of each feature's "
                    "relationship with predicted churn risk — 13 of 17 features are monotonic "
                    "(a single, consistent direction of effect), which is what makes the driver "
                    "explanations shown elsewhere in this app defensible."
                )


# ---------------------------------------------------------------------------
# TAB 4 — Customer Action Center
# ---------------------------------------------------------------------------
with tab4:
    section_header("Customer action center")

    if st.session_state.combined_df is None:
        st.warning("Complete the Customer Profile Builder tab first.")
    else:
        combined = st.session_state.combined_df.sort_values("churn_probability", ascending=False).reset_index(drop=True)

        n_total = len(combined)
        n_elevated = (combined["risk_tier"] == "Elevated").sum()
        n_watch = (combined["risk_tier"] == "Watch").sum()
        n_low = (combined["risk_tier"] == "Low risk").sum()
        st.markdown(f"""
<div class="statcard-row">
  <div class="statcard neutral"><div class="label">Total customers</div><div class="value">{n_total:,}</div></div>
  <div class="statcard elevated"><div class="label">Elevated</div><div class="value">{n_elevated:,}</div></div>
  <div class="statcard watch"><div class="label">Watch</div><div class="value">{n_watch:,}</div></div>
  <div class="statcard low"><div class="label">Low risk</div><div class="value">{n_low:,}</div></div>
</div>
""", unsafe_allow_html=True)

        subsection_header("Find your customers")
        option_map = {
            f"{row.customer_name} — {row.risk_tier} ({row.customer_id})": row.customer_id
            for row in combined.itertuples()
        }
        selected_label = st.selectbox("Select a customer", options=list(option_map.keys()))
        selected_id = option_map[selected_label]
        row = combined[combined["customer_id"] == selected_id].iloc[0]

        subsection_header("Customer profile")
        tenure_display = f"{row['tenure_years']:.1f} years" if "tenure_years" in row else "—"
        st.markdown(f"""
<div class="profile-strip">
  <div class="profile-item"><div class="k">Name</div><div class="v">{row['customer_name']}</div></div>
  <div class="profile-item"><div class="k">Customer ID</div><div class="v">{row['customer_id']}</div></div>
  <div class="profile-item"><div class="k">Age</div><div class="v">{fmt_int(row['age']) if 'age' in row else '—'}</div></div>
  <div class="profile-item"><div class="k">Tenure with bank</div><div class="v">{tenure_display}</div></div>
  <div class="profile-item"><div class="k">Products held</div><div class="v">{fmt_int(row['num_products_held']) if 'num_products_held' in row else '—'}</div></div>
  <div class="profile-item"><div class="k">Location</div><div class="v">{row['city']}, {row['state']}</div></div>
</div>
""", unsafe_allow_html=True)

        subsection_header("Risk snapshot")
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn risk", fmt_decimal2(row['churn_probability']))
        with m2:
            st.markdown('<div class="profile-item"><div class="k">Risk level</div><div class="v">' +
                        risk_badge(row["risk_tier"]) + '</div></div>', unsafe_allow_html=True)
        m3.metric("Risk rank", f"{fmt_int(row['risk_decile'])} / 10")

        subsection_header("Customer motive & recommended action")
        st.markdown(f"""
<div class="motive-card">
  <p class="motive-text">{row['motive_text']}</p>
  <div class="action-box"><b>Recommended action:</b> {row['recommended_action']}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander("View underlying data"):
            drivers = row["top_drivers"]
            if drivers:
                driver_df = pd.DataFrame(drivers)
                driver_specs = [
                    ("Feature", lambda r: fmt_title(r["feature"])),
                    ("Customer value", lambda r: fmt_driver_value(r["feature"], r["customer_value"])),
                    ("Relationship to risk", lambda r: fmt_title(r["relationship_to_risk"])),
                ]
                st.markdown(html_table(driver_df, driver_specs), unsafe_allow_html=True)
            else:
                st.caption("No notable risk drivers for this customer.")

            if st.session_state.api_key:
                if st.button("Regenerate with AI", key=f"regen_{selected_id}"):
                    try:
                        with st.spinner("Generating..."):
                            new_point = logic.generate_talking_point_llm(
                                api_key=st.session_state.api_key,
                                risk_tier=row["risk_tier"],
                                probability=float(row["churn_probability"]),
                                drivers=row["top_drivers"],
                                persona=row["predicted_life_moment"],
                            )
                        st.info(new_point)
                    except ImportError:
                        st.error("Install the `anthropic` package to use this feature: pip install anthropic")
                    except Exception as e:
                        st.error(f"AI generation failed. ({e})")

        subsection_header("Notes")
        note_key = f"note_{selected_id}"
        current_note = st.session_state.notes.get(selected_id, "")
        new_note = st.text_area("Add notes from your call or review here", value=current_note, key=note_key, height=90, label_visibility="collapsed")
        st.session_state.notes[selected_id] = new_note

        f1, f2, f3 = st.columns([2, 1, 1])
        f1.markdown("**Was this a good lead?**")
        current_feedback = st.session_state.feedback.get(selected_id)
        if f2.button("👍 Good lead" if current_feedback != "up" else "✅ Good lead", key=f"up_{selected_id}"):
            st.session_state.feedback[selected_id] = "up" if current_feedback != "up" else None
            st.rerun()
        if f3.button("👎 Not useful" if current_feedback != "down" else "✅ Not useful", key=f"down_{selected_id}"):
            st.session_state.feedback[selected_id] = "down" if current_feedback != "down" else None
            st.rerun()

        st.divider()
        qc1, qc2 = st.columns([3, 1])
        with qc1:
            subsection_header("Your customer queue")
        show_all = qc2.toggle("Show all customers", value=False)

        queue = combined if show_all else combined[combined["risk_tier"] == "Elevated"]
        queue_specs = [
            ("Customer", lambda r: fmt_text(r["customer_name"])),
            ("ID", lambda r: fmt_text(r["customer_id"])),
            ("Churn Risk", lambda r: fmt_decimal2(r["churn_probability"])),
            ("Risk Level", lambda r: risk_badge(r["risk_tier"])),
        ]
        st.caption(f"Showing {min(len(queue), 50):,} of {len(queue):,} matching customers" + ("" if show_all else " (Elevated only — today's call list)"))
        st.markdown(html_table(queue.head(50), queue_specs), unsafe_allow_html=True)
