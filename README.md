# Dollar Bank — Customer Retention Dashboard

A 5-tab Streamlit app: an overview of the business problem, then upload
customer data → generate predictions → combine into one profile per customer
→ a single-screen action center for bankers to prep for a call, with
hyper-personalized, banker-ready insight generated deterministically (no
API key required).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Sample data (999 real holdout customers,
enriched with fictitious names/city/state for readability, blind — no
answer key) loads by default in Load Customers, so you can click through
the whole pipeline immediately.

## What's in here

```
app.py                      # Streamlit UI -- 5 tabs, session state, no business logic
logic.py                    # Model loading, scoring, joining, insight generation,
                              # and display formatting (all pure functions, testable)
test_logic.py                # Plain-Python tests for logic.py
test_app_integration.py       # Simulates every table/data transform app.py performs
generate_reports.py           # Regenerates the Model Performance plots in reports/
models/churn/                  # Trained GradientBoostingClassifier + artifacts
models/life_moments/           # Trained RandomForestClassifier + artifacts
sample_data/                    # Blind holdout sets, enriched with names/city/state
reports/                        # Static ROC / cumulative-gain / partial-dependence plots
requirements.txt
```

Run `python3 test_logic.py` and `python3 test_app_integration.py` any time
after changing `logic.py` or `app.py` — both run in seconds.

## Design decisions worth knowing about

- **Font-family is applied to `.stApp` only** (no universal `*` selector, no
  `!important`). An earlier version applied it to every element, which also
  overrode the icon fonts Streamlit uses for expander arrows and the select
  dropdown chevron, making them render as distorted boxes. Letting the rule
  cascade naturally fixes that without touching real text.
- **All reference tables are custom HTML, not `st.dataframe`.** Streamlit's
  native tables render onto an HTML canvas (glide-data-grid), which no CSS
  can reach — that's why the font, comma formatting, and colored risk
  badges wouldn't apply to them previously. `logic.html_table()` builds
  plain HTML instead, giving full control and visual consistency with the
  rest of the app. The formatting helpers (`fmt_currency`, `fmt_decimal2`,
  etc.) are pure functions with unit tests in `test_logic.py`.
- **Numbers are capped at 2 decimal places wherever a value is naturally
  decimal** (churn probability, confidence, utilization); values that are
  naturally whole (counts, ranks, ages) are never given a forced decimal.
  Large numbers get comma separators throughout.
- **The combined motive + recommended action is hyper-personalized and
  deterministic**, not a live LLM call. It composes three things: (1) the
  customer's dominant churn-risk theme (reduced engagement / financial
  stress / service friction), (2) their life-moment persona, and (3) their
  risk tier — 7 personas × 4 themes × 2 actionable tiers, giving up to 56
  distinct recommended actions, further varied by a secondary-driver
  addendum when a customer has risk signals spanning more than one theme.
  Service friction is never softened by persona — a complaint pattern is a
  real relationship problem regardless of what else is going on in
  someone's life. See `generate_motive_and_action` in `logic.py`.
- **No boilerplate caveats.** An earlier version appended a fixed sentence
  ("the scale of this pattern is bigger than their life stage would
  explain") to every Elevated-tier customer regardless of their actual
  data. It's been replaced with a check for a genuine secondary driver from
  a different theme — if one exists, the motive names it specifically; if
  every driver tells the same coherent story, no caveat gets manufactured.
- **Tone is deliberately warm, not surveillance-like.** Phrases like "the
  account has been inactive" or "reached out to service X times" read like
  a tracking log. Driver phrases were rewritten to sound like normal
  relationship-banking notes ("hasn't used the account much lately",
  "contacted support X times this year") while keeping the same underlying
  numbers.
- **Load Customers shows real raw feature values**, not a pre-summarized
  label that would give away what the model later predicts, with friendly
  column labels (`logic.PREVIEW_COL_LABELS`) instead of raw column names.
- **Customer Action Center is single-customer-focused**, not a big
  browsable table: a risk-sorted dropdown drives everything on screen, with
  a secondary "Your Customer Queue" table defaulting to Elevated-risk
  customers only (today's call list), toggleable to show everyone. Risk
  tier counts are shown as color-coded stat cards using the same blue
  gradient as the risk badges.
- **Notes and lead-quality feedback are session-only** — no database in
  this build, so they reset on app restart.

## Cost if you enable AI-generated insight

The "Regenerate with AI" button in the Customer Action Center is optional
and per-customer, not run automatically for the whole batch. Each call is
roughly 450 input tokens and 120 output tokens — at Claude Haiku 4.5
pricing ($1/$5 per million input/output tokens), about $0.001 per customer.
Hosting is free on Streamlit Community Cloud for a public demo app.
