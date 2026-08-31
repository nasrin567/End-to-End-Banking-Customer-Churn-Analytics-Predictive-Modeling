import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_model_bundle.pkl"

PREPROCESSING_PATH = (
    BASE_DIR
    / "predictive_modelling"
    / "processed_data"
    / "preprocessing_bundle.pkl"
)

ACCOUNT_PATH = (
    BASE_DIR / "data" / "processed" / "account.csv"
)

DEMOGRAPHIC_PATH = (
    BASE_DIR / "data" / "processed" / "demographic.csv"
)

LOCATION_PATH = (
    BASE_DIR / "data" / "processed" / "location.csv"
)


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# ============================================================
# Load Preprocessing Objects
# ============================================================

@st.cache_resource
def load_preprocessing():

    return joblib.load(PREPROCESSING_PATH)


# ============================================================
# Load Business Data
# ============================================================

@st.cache_data
def load_data():

    account = pd.read_csv(ACCOUNT_PATH)

    demographic = pd.read_csv(
        DEMOGRAPHIC_PATH
    )

    location = pd.read_csv(
        LOCATION_PATH
    )

    df = (
        demographic
        .merge(
            account,
            on="CustomerId",
            how="inner"
        )
        .merge(
            location,
            on="LocationId",
            how="inner"
        )
    )

    return df


# ============================================================
# Load Resources
# ============================================================

try:

    model_bundle = load_model()

    preprocessing_bundle = load_preprocessing()

    df = load_data()

except Exception as e:

    st.error(
        "Unable to load the model or project data."
    )

    st.exception(e)

    st.stop()


# ============================================================
# Extract Model and Preprocessing Objects
# ============================================================

model = model_bundle["model"]

scaler = preprocessing_bundle["scaler"]

num_cols = preprocessing_bundle["num_cols"]

cat_cols = preprocessing_bundle["cat_cols"]

feature_columns = (
    preprocessing_bundle["feature_columns"]
)

categorical_values = (
    preprocessing_bundle["categorical_values"]
)


# ============================================================
# GLOBAL DESIGN SYSTEM (CSS)
# ============================================================
# Applied once, used across every page. Native Streamlit
# components (st.container(border=True), st.columns, st.metric)
# do the heavy lifting; this CSS only adds polish on top of them
# so nothing depends on HTML rendering the whole page.

st.markdown(
    """
    <style>

    /* ---------- Global type & spacing ---------- */
    .block-container {
        padding-top: 3.4rem;
        padding-bottom: 3rem;
        overflow: visible;
    }

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    /* ---------- Responsive columns / cards ---------- */
    /* Keeps columns evenly distributed and card content
       stretched to the full column width whether the sidebar
       is expanded or collapsed, instead of shrinking or
       growing unevenly relative to each other. */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
    }

    div[data-testid="column"] > div {
        width: 100%;
    }

    /* ---------- Hero ---------- */
    .hero-wrap {
        padding: 14px 4px 6px 4px;
        overflow: visible;
        position: relative;
        z-index: 1;
    }

    .hero-eyebrow {
        display: inline-block;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #22d3ee;
        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.25);
        padding: 6px 14px;
        border-radius: 999px;
        margin-bottom: 16px;
        line-height: 1.4;
        white-space: nowrap;
        overflow: visible;
    }

    .hero-title {
        font-size: clamp(30px, 4vw, 46px);
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #e2e8f0, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 19px;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 6px;
    }

    .hero-desc {
        font-size: 15.5px;
        color: #94a3b8;
        line-height: 1.7;
        max-width: 780px;
    }

    .momentum-bar {
        height: 6px;
        width: 240px;
        margin-top: 20px;
        border-radius: 20px;
        background: linear-gradient(
            90deg, #3b82f6, #a855f7, #06b6d4, #3b82f6
        );
        background-size: 300% 100%;
        box-shadow: 0 0 14px rgba(99, 102, 241, 0.55);
        animation: momentum 3s linear infinite;
    }

    .momentum-caption {
        margin-top: 10px;
        font-size: 13px;
        color: #94a3b8;
        max-width: 640px;
        line-height: 1.5;
    }

    @keyframes momentum {
        0%   { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }

    /* ---------- Live hover motion on every card ---------- */
    /* Applies to every st.container(border=True) on the page
       (KPI containers, feature cards, chart cards, result
       cards) so the UI visibly responds to the cursor. */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        transition: transform 0.18s ease,
                    box-shadow 0.18s ease,
                    border-color 0.18s ease;
        border-radius: 16px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
        border-color: rgba(34, 211, 238, 0.45) !important;
    }

    .kpi-card, .feature-card-body {
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }

    /* ---------- Section labels ---------- */
    .section-label {
        font-size: 13px;
        color: #7c8794;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin: 6px 0 6px 0;
    }

    /* ---------- KPI cards ---------- */
    .kpi-card {
        border-radius: 16px;
        padding: 16px 18px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(148, 163, 184, 0.04);
        width: 100%;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 104px;
    }

    .kpi-icon {
        font-size: 22px;
        margin-bottom: 4px;
    }

    .kpi-label {
        font-size: 12.5px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 2px;
        /* reserves room for a 2-line label so a long label
           (e.g. "CHURNED CUSTOMERS") and a short one (e.g.
           "MODELS EVALUATED") produce the same card height */
        min-height: 30px;
        line-height: 15px;
    }

    .kpi-value {
        font-size: clamp(20px, 2vw, 26px);
        font-weight: 800;
        color: #f1f5f9;
    }

    .kpi-detail-btn button {
        font-size: 12.5px !important;
        padding: 2px 0 !important;
    }

    /* ---------- Feature cards (Home page) — fixed-height
       body so all three cards align and their buttons sit
       at the same vertical position regardless of how long
       each description is ---------- */
    .feature-card-body {
        display: flex;
        flex-direction: column;
        gap: 8px;
        min-height: 148px;
    }

    .feature-icon {
        font-size: 26px;
    }

    .feature-title {
        font-size: 19px;
        font-weight: 800;
        color: #f1f5f9;
    }

    .feature-desc {
        font-size: 14.5px;
        color: #94a3b8;
        line-height: 1.55;
    }

    .kpi-blue    { border-left: 3px solid #3b82f6; }
    .kpi-red     { border-left: 3px solid #ef4444; }
    .kpi-purple  { border-left: 3px solid #a855f7; }
    .kpi-green   { border-left: 3px solid #22c55e; }
    .kpi-cyan    { border-left: 3px solid #06b6d4; }
    .kpi-orange  { border-left: 3px solid #f97316; }

    /* ---------- Insight banner ---------- */
    .insight-box {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 26px;
        background: linear-gradient(
            135deg, rgba(59,130,246,0.08), rgba(168,85,247,0.06)
        );
    }

    .insight-title {
        font-size: 21px;
        font-weight: 800;
        margin-bottom: 8px;
        color: #f1f5f9;
    }

    .insight-text {
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 15px;
    }

    /* ---------- Flow chip (workflow arrows) ---------- */
    .flow-row {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin: 6px 0 4px 0;
    }

    .flow-chip {
        padding: 10px 18px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(148, 163, 184, 0.05);
        font-weight: 700;
        font-size: 14.5px;
        color: #e2e8f0;
        white-space: nowrap;
    }

    .flow-arrow {
        color: #64748b;
        font-size: 18px;
    }

    /* ---------- Step number badges ---------- */
    .step-badge {
        display: inline-block;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #22d3ee;
        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.25);
        border-radius: 8px;
        padding: 2px 10px;
        margin-bottom: 8px;
    }

    /* ---------- Callout (small insight under a chart) ---------- */
    .callout {
        border-left: 3px solid #06b6d4;
        background: rgba(6, 182, 212, 0.06);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 14px;
        color: #cbd5e1;
        margin-top: 6px;
    }

    /* ---------- Risk result cards ---------- */
    .risk-card-high {
        border-radius: 18px;
        padding: 26px;
        border: 1px solid rgba(239, 68, 68, 0.35);
        background: linear-gradient(
            135deg, rgba(239,68,68,0.14), rgba(249,115,22,0.06)
        );
    }

    .risk-card-low {
        border-radius: 18px;
        padding: 26px;
        border: 1px solid rgba(34, 197, 94, 0.35);
        background: linear-gradient(
            135deg, rgba(34,197,94,0.14), rgba(6,182,212,0.06)
        );
    }

    .risk-label-high {
        font-size: 24px;
        font-weight: 800;
        color: #fca5a5;
        margin-bottom: 6px;
    }

    .risk-label-low {
        font-size: 24px;
        font-weight: 800;
        color: #86efac;
        margin-bottom: 6px;
    }

    /* ---------- Sidebar footer ---------- */
    .sidebar-footer {
        font-size: 12px;
        color: #64748b;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* ---------- Best-model badge ---------- */
    .best-badge {
        display: inline-block;
        font-size: 12px;
        font-weight: 800;
        color: #0f172a;
        background: #22d3ee;
        border-radius: 999px;
        padding: 3px 10px;
        margin-left: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Navigation
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

pages = [
    "🏠 Home",
    "📊 Business Analytics",
    "🔮 Churn Prediction",
    "🤖 Model Performance"
]

# Streamlit forbids writing to st.session_state.page once the
# radio widget with key="page" has been instantiated in a run.
# So a button elsewhere on the page can't set
# st.session_state.page directly — instead it stores the target
# in "nav_target" and reruns; here, BEFORE the radio widget is
# created, any pending nav_target is applied to page and cleared.
# This is what actually makes the Home-page buttons navigate.
if st.session_state.get("nav_target"):
    st.session_state.page = st.session_state.nav_target
    st.session_state.nav_target = None

with st.sidebar:

    st.markdown("## 🏦 Bank Churn Intelligence")

    st.caption("Customer retention analytics platform")

    st.write("")

    # key="page" binds this widget directly to
    # st.session_state.page.
    st.radio(
        "Navigation",
        pages,
        key="page",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-footer">Customer Analytics &amp; ML</div>',
        unsafe_allow_html=True
    )

page = st.session_state.page


# ============================================================
# Small reusable rendering helpers (presentation only —
# none of these touch data, model, or prediction logic)
# ============================================================

def kpi_card(icon, label, value, color_class):

    st.markdown(
        f"""
        <div class="kpi-card {color_class}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def kpi_card_interactive(icon, label, value, color_class, detail_key):

    # Renders a KPI card plus a small "View Details" toggle.
    # Only the clicked KPI's key is stored in session_state, so
    # only that one KPI's detail panel opens (a second click on
    # the same KPI closes it again); the other KPI cards stay
    # compact regardless of which one is expanded.

    kpi_card(icon, label, value, color_class)

    st.markdown('<div class="kpi-detail-btn">', unsafe_allow_html=True)

    is_open = st.session_state.get("expanded_kpi") == detail_key

    if st.button(
        "▲ Hide" if is_open else "View Details",
        key=f"kpi_toggle_{detail_key}",
        use_container_width=True
    ):
        st.session_state.expanded_kpi = (
            None if is_open else detail_key
        )
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def callout(text):

    st.markdown(
        f'<div class="callout">💡 {text}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    # ========================================================
    # Core KPIs (computed first so the hero section below can
    # reference the real churn rate instead of a decorative,
    # unexplained number)
    # ========================================================

    total_customers = len(df)

    churned_customers = int(
        df["Churned"].sum()
    )

    churn_rate = (
        churned_customers
        / total_customers
        * 100
    )

    avg_balance = df["Balance"].mean()

    avg_salary = df["Salary"].mean()

    # ========================================================
    # HERO
    # ========================================================

    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-eyebrow">Retention Intelligence Platform</div>
            <div class="hero-title">🏦 Bank Churn Intelligence</div>
            <div class="hero-subtitle">
                Turn customer data into retention decisions.
            </div>
            <div class="hero-desc">
                This application combines business analytics,
                customer churn prediction, machine learning and
                model comparison into a single retention
                intelligence workspace — helping identify which
                customers are at risk and what to do about it.
            </div>
            <div class="momentum-bar"></div>
            <div class="momentum-caption">
                Historical Churn Rate — represents the current
                historical churn rate ({churn_rate:.2f}% of
                {total_customers:,} customers analyzed).
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ========================================================
    # KPI SECTION
    # ========================================================

    st.markdown(
        '<div class="section-label">Customer Intelligence</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        kpi_card_interactive(
            "👥", "Total Customers", f"{total_customers:,}",
            "kpi-blue", "total_customers"
        )

    with c2:
        kpi_card_interactive(
            "⚠️", "Churned Customers", f"{churned_customers:,}",
            "kpi-red", "churned_customers"
        )

    with c3:
        kpi_card_interactive(
            "📉", "Churn Rate", f"{churn_rate:.2f}%",
            "kpi-purple", "churn_rate"
        )

    with c4:
        kpi_card_interactive(
            "💰", "Avg. Balance", f"${avg_balance:,.0f}",
            "kpi-green", "avg_balance"
        )

    with c5:
        kpi_card_interactive(
            "💵", "Avg. Salary", f"${avg_salary:,.0f}",
            "kpi-cyan", "avg_salary"
        )

    with c6:
        kpi_card_interactive(
            "🤖", "Models Evaluated", "5",
            "kpi-orange", "models_evaluated"
        )

    # --------------------------------------------------------
    # Expanded KPI detail panel — wide, full-width, bordered.
    # Only renders when a KPI's "View Details" was clicked, and
    # only shows the content for that one KPI.
    # --------------------------------------------------------

    expanded_kpi = st.session_state.get("expanded_kpi")

    kpi_details = {
        "total_customers": (
            "👥 Total Customers",
            f"The dataset covers **{total_customers:,}** "
            "customers in total. This is the full customer "
            "base used across every analysis and prediction "
            "in this application — every churn rate, average, "
            "and model score below is calculated from this "
            "same population."
        ),
        "churned_customers": (
            "⚠️ Churned Customers",
            f"**{churned_customers:,}** of the "
            f"{total_customers:,} customers in the dataset "
            "have churned. Visit **Business Analytics** to see "
            "how churn breaks down by geography, age, product "
            "count, and activity status."
        ),
        "churn_rate": (
            "📉 Churn Rate",
            f"**{churn_rate:.2f}%** of customers in the dataset "
            "have churned. This is the same figure referenced "
            "by the gradient bar above the KPI section — it "
            "represents the bank's overall historical exposure "
            "to customer attrition."
        ),
        "avg_balance": (
            "💰 Average Balance",
            f"The average account balance across all customers "
            f"is **${avg_balance:,.0f}**. Balance is one of the "
            "financial inputs used by the churn prediction "
            "model on the **Churn Prediction** page."
        ),
        "avg_salary": (
            "💵 Average Salary",
            f"The average customer salary in the dataset is "
            f"**${avg_salary:,.0f}**. Salary feeds into the "
            "model's Balance-to-Salary ratio feature used "
            "during prediction."
        ),
        "models_evaluated": (
            "🤖 Models Evaluated",
            "Five machine-learning approaches were evaluated "
            "for this problem: **Logistic Regression, Random "
            "Forest, SVM Polynomial, SVM RBF, and XGBoost**. "
            "See the **Model Performance** page for their "
            "Recall, Precision, F1, and ROC-AUC scores."
        ),
    }

    if expanded_kpi and expanded_kpi in kpi_details:

        title, detail_text = kpi_details[expanded_kpi]

        with st.container(border=True):

            st.subheader(title)

            st.write(detail_text)

    st.write("")
    st.divider()

    # ========================================================
    # FEATURE CARDS
    # ========================================================

    st.markdown(
        '<div class="section-label">Platform Overview</div>',
        unsafe_allow_html=True
    )

    st.header("What do you want to explore?")

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.markdown(
                """
                <div class="feature-card-body">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Business Analytics</div>
                    <div class="feature-desc">
                        Discover where and why customers churn
                        through demographic, geographic, account
                        and financial analysis.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Explore Analytics  →",
                key="home_analytics",
                use_container_width=True
            ):
                st.session_state.nav_target = "📊 Business Analytics"
                st.rerun()

    with col2:

        with st.container(border=True):

            st.markdown(
                """
                <div class="feature-card-body">
                    <div class="feature-icon">🔮</div>
                    <div class="feature-title">Churn Prediction</div>
                    <div class="feature-desc">
                        Estimate individual customer churn
                        probability and classify customer risk.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Predict Customer Risk  →",
                key="home_prediction",
                use_container_width=True
            ):
                st.session_state.nav_target = "🔮 Churn Prediction"
                st.rerun()

    with col3:

        with st.container(border=True):

            st.markdown(
                """
                <div class="feature-card-body">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-title">Model Performance</div>
                    <div class="feature-desc">
                        Compare five machine-learning approaches
                        using Recall, Precision, F1-score and
                        ROC-AUC.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Compare Models  →",
                key="home_models",
                use_container_width=True
            ):
                st.session_state.nav_target = "🤖 Model Performance"
                st.rerun()

    st.write("")
    st.divider()

    # ========================================================
    # RETENTION INTELLIGENCE
    # ========================================================

    st.markdown(
        '<div class="section-label">Product Workflow</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="insight-box">
            <div class="insight-title">⚡ Retention Intelligence</div>
            <div class="insight-text">
                This platform turns raw historical behavior into a
                clear retention decision — moving step by step from
                data to action.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        """
        <div class="flow-row">
            <div class="flow-chip">📚 Historical Behavior</div>
            <div class="flow-arrow">→</div>
            <div class="flow-chip">📊 Business Analytics</div>
            <div class="flow-arrow">→</div>
            <div class="flow-chip">🤖 Machine Learning</div>
            <div class="flow-arrow">→</div>
            <div class="flow-chip">🔮 Churn Probability</div>
            <div class="flow-arrow">→</div>
            <div class="flow-chip">⚠️ Customer Risk</div>
            <div class="flow-arrow">→</div>
            <div class="flow-chip">🎯 Retention Action</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.divider()

    # ========================================================
    # DECISION WORKFLOW
    # ========================================================

    st.markdown(
        '<div class="section-label">Decision Workflow</div>',
        unsafe_allow_html=True
    )

    st.header("How the system works")

    col1, col2, col3, col4 = st.columns(4)

    steps = [
        ("01", "Analyze", "Understand customer behavior and "
                           "discover the strongest churn patterns."),
        ("02", "Predict", "Estimate the probability that an "
                           "individual customer will churn."),
        ("03", "Prioritize", "Identify high-risk customers who "
                              "may require retention attention."),
        ("04", "Act", "Use the prediction to support targeted "
                      "retention decisions."),
    ]

    for col, (num, title, desc) in zip(
        [col1, col2, col3, col4], steps
    ):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="step-badge">{num}</div>',
                    unsafe_allow_html=True
                )
                st.subheader(title)
                st.write(desc)


# ============================================================
# PAGE 2 — BUSINESS ANALYTICS
# ============================================================

if page == "📊 Business Analytics":

    st.markdown(
        '<div class="section-label">Business Analytics</div>',
        unsafe_allow_html=True
    )

    st.header("Business Analytics")

    st.write(
        "Explore customer segments, churn patterns, geography, "
        "account behavior and financial characteristics."
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_customers = len(df)

    churned_customers = int(
        df["Churned"].sum()
    )

    churn_rate = (
        churned_customers
        / total_customers
        * 100
    )

    avg_balance = df["Balance"].mean()

    avg_salary = df["Salary"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        kpi_card("👥", "Total Customers", f"{total_customers:,}", "kpi-blue")

    with c2:
        kpi_card("⚠️", "Churned Customers", f"{churned_customers:,}", "kpi-red")

    with c3:
        kpi_card("📉", "Churn Rate", f"{churn_rate:.2f}%", "kpi-purple")

    with c4:
        kpi_card("💰", "Avg. Balance", f"${avg_balance:,.0f}", "kpi-green")

    with c5:
        kpi_card("💵", "Avg. Salary", f"${avg_salary:,.0f}", "kpi-cyan")

    st.write("")
    st.divider()

    # --------------------------------------------------------
    # Churn by Geography
    # --------------------------------------------------------

  
    with st.container(border=True):

        st.subheader("🌍 Churn by Geography")

        geography_churn = (
            df.groupby("Geography")["Churned"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            geography_churn,
            x="Geography",
            y="Churned",
            text="Churned",
            color="Geography",
            color_discrete_sequence=[
                "#3B82F6",  # Canada
                "#22C55E",  # France
                "#F59E0B",  # Germany
                "#A855F7",  # Spain
                "#EF4444",  # UK
                "#06B6D4"   # USA
            ]
        )

        fig.update_layout(
            height=450,
            xaxis={
                "title": None,
                "tickangle": 0
            },
            yaxis={
                "title": "Churned Customers"
            },
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            )
        )

        fig.update_traces(
            texttemplate="%{text}",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        top_geo = geography_churn.iloc[0]["Geography"]

        callout(
            f"{top_geo} has the highest number of churned "
            f"customers among all geographies in the dataset."
        )

    # --------------------------------------------------------
    # Churn by Age
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader("🎂 Churn by Age")

        age_churn = (
            df.groupby("Age")["Churned"]
            .mean()
            .mul(100)
        )

        st.line_chart(
            age_churn,
            x_label="Age (Years)",
            y_label="Churn Rate (%)"
        )

        peak_age = age_churn.idxmax()

        callout(
            f"Churn rate peaks around age {peak_age}, at "
            f"{age_churn.max():.1f}%."
        )

    st.write("")

    # --------------------------------------------------------
    # Churn by Number of Products
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader("📦 Churn by Number of Products")

        product_churn = (
            df.groupby("NumProducts")["Churned"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig = px.bar(
            product_churn,
            x="NumProducts",
            y="Churned",
            text="Churned",
            color="NumProducts",
            color_discrete_sequence=[
                "#3B82F6",
                "#22C55E",
                "#F59E0B",
                "#EF4444"
            ]
        )

        fig.update_layout(
            height=450,

            xaxis={
                "title": "Number of Products",
                "tickangle": 90
            },

            yaxis={
                "title": "Churn Rate (%)",
                "range": [0, 110]
            },

            showlegend=False,

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            )
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        worst_products = product_churn.loc[
            product_churn["Churned"].idxmax(),
            "NumProducts"
        ]

        worst_churn_rate = product_churn["Churned"].max()

        callout(
            f"Customers holding {int(worst_products)} product(s) show "
            f"the highest churn rate at {worst_churn_rate:.1f}%."
        )

    st.write("")  

    # --------------------------------------------------------
    # Churn by Active Status
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader("🟢 Churn by Active Status")

        active_churn = (
            df.groupby("IsActive")["Churned"]
            .mean()
            .mul(100)
            .reset_index()
        )
         # Convert 0/1 into readable labels
        active_churn["Status"] = active_churn["IsActive"].map({
            0: "Inactive",
            1: "Active"
        })

        fig = px.bar(
            active_churn,
            x="Status",
            y="Churned",
            text="Churned",
            color="Status",
            color_discrete_sequence=[
                "#EF4444",
                "#22C55E"
            ]
        )

        fig.update_layout(
            height=450,

            xaxis={
                "title": "Active Status",
                "tickangle": 90
            },

            yaxis={
                "title": "Churn Rate (%)",
                "range": [0, 30]
            },

            showlegend=False,

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            )
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
            
        )

        callout(
            "Inactive members tend to have a higher churn rate "
            "than active members — compare the two bars above."

            
        )
    st.write("")

# ============================================================
# PAGE 3 — CHURN PREDICTION
# ============================================================

elif page == "🔮 Churn Prediction":

    st.markdown(
        '<div class="section-label">Risk Assessment</div>',
        unsafe_allow_html=True
    )

    st.header(
        "Customer Churn Prediction"
    )

    st.write(
        "Assess the likelihood that this customer will churn."
    )

    st.write("")

    # --------------------------------------------------------
    # Input Fields — grouped into logical sections
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader("👤 Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                sorted(
                    df["Gender"]
                    .dropna()
                    .unique()
                )
            )

        with col2:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=40
            )

        with col3:

            geography = st.selectbox(
                "Geography",
                sorted(
                    df["Geography"]
                    .dropna()
                    .unique()
                )
            )

    st.write("")

    with st.container(border=True):

        st.subheader("💰 Financial Profile")

        col1, col2 = st.columns(2)

        with col1:

            salary = st.number_input(
                "Salary",
                min_value=0.0,
                value=float(
                    df["Salary"].median()
                )
            )

        with col2:

            balance = st.number_input(
                "Balance",
                min_value=0.0,
                value=float(
                    df["Balance"].median()
                )
            )

    st.write("")

    with st.container(border=True):

        st.subheader("🏦 Account Profile")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            tenure = st.number_input(
                "Tenure",
                min_value=0,
                max_value=20,
                value=5
            )

        with col2:

            num_products = st.number_input(
                "Number of Products",
                min_value=1,
                max_value=10,
                value=2
            )

        with col3:

            has_credit_card = st.selectbox(
                "Has Credit Card",
                [0, 1],
                format_func=lambda x:
                    "Yes" if x == 1 else "No"
            )

        with col4:

            is_active = st.selectbox(
                "Active Member",
                [0, 1],
                format_func=lambda x:
                    "Yes" if x == 1 else "No"
            )

    st.write("")

    predict_button = st.button(
        "🔮 PREDICT CUSTOMER RISK",
        use_container_width=True,
        type="primary"
    )


    # ========================================================
    # Prediction
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # Create raw customer dataframe
            # ------------------------------------------------

            input_df = pd.DataFrame({

                "Gender": [gender],

                "Age": [age],

                "Salary": [salary],

                "Geography": [geography],

                "Tenure": [tenure],

                "Balance": [balance],

                "NumProducts": [num_products],

                "HasCreditCard": [
                    has_credit_card
                ],

                "IsActive": [
                    is_active
                ]
            })


            # ------------------------------------------------
            # Feature Engineering
            # ------------------------------------------------

            input_df[
                "BalanceSalaryRatio"
            ] = (
                input_df["Balance"]
                / input_df["Salary"].replace(
                    0,
                    np.nan
                )
            )

            input_df[
                "TenureByAge"
            ] = (
                input_df["Tenure"]
                / input_df["Age"]
            )

            input_df = input_df.fillna(0)


            # ------------------------------------------------
            # Numerical Columns
            # ------------------------------------------------

            # The original preprocessing scaler was fitted
            # with Churned included.
            #
            # We must therefore provide the scaler with
            # the same columns it saw during fitting.

            scaler_input = pd.DataFrame(
                0.0,
                index=input_df.index,
                columns=num_cols
            )

            for col in num_cols:

                if col in input_df.columns:

                    scaler_input[col] = (
                        input_df[col]
                    )

                elif col == "Churned":

                    # Placeholder only because the scaler
                    # was originally fitted with Churned.
                    #
                    # This column is removed before prediction.
                    scaler_input[col] = 0


            scaled_values = scaler.transform(
                scaler_input
            )

            scaled_df = pd.DataFrame(
                scaled_values,
                columns=num_cols,
                index=input_df.index
            )


            # ------------------------------------------------
            # Remove Churned
            # ------------------------------------------------

            if "Churned" in scaled_df.columns:

                scaled_df = scaled_df.drop(
                    columns=["Churned"]
                )


            # ------------------------------------------------
            # Add categorical variables
            # ------------------------------------------------

            for col in cat_cols:

                for value in categorical_values[col]:

                    scaled_df[
                        f"{col}_{value}"
                    ] = np.where(
                        input_df[col] == value,
                        1,
                        0
                    )


            # ------------------------------------------------
            # Remove original categorical columns
            # ------------------------------------------------

            # Combine numerical and categorical features

            processed_input = scaled_df.copy()


            # ------------------------------------------------
            # Backfill any raw pass-through features
            # ------------------------------------------------
            #
            # num_cols (scaled) and cat_cols (one-hot dummies)
            # don't necessarily cover every training feature.
            # Flags such as HasCreditCard / IsActive are often
            # used as-is (0/1) without scaling or dummy-encoding.
            # Without this step, any such column silently falls
            # back to 0 in the reindex below regardless of what
            # the user selected, so it can never affect the
            # prediction. This copies the raw input value through
            # for any training feature that neither loop produced.

            for col in feature_columns:

                if (
                    col not in processed_input.columns
                    and col in input_df.columns
                ):

                    processed_input[col] = input_df[col]


            # ------------------------------------------------
            # Exact Training Feature Order
            # ------------------------------------------------

            processed_input = processed_input.reindex(
                columns=feature_columns,
                fill_value=0
            )


            # ------------------------------------------------
            # Safety Check
            # ------------------------------------------------

            if processed_input.shape[1] != len(
                feature_columns
            ):

                raise ValueError(
                    "Prediction features do not match "
                    "the training features."
                )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------
            prediction_raw = model.predict(
               processed_input
            )[0]

            probabilities = model.predict_proba(
                processed_input
            )[0]

            churn_class_index = list(model.classes_).index(True)

            probability = float(
                probabilities[churn_class_index]
            )

            prediction = bool(prediction_raw)

            probability_percent = probability * 100
            

            probability_percent = (
                probability * 100
            )


            # =================================================
            # Prediction Result
            # =================================================

            st.write("")
            st.divider()

            st.markdown(
                '<div class="section-label">Result</div>',
                unsafe_allow_html=True
            )

            with st.expander("🔍 Show technical details (debug)"):

                st.caption(
                    "Exactly what was sent to the model — use "
                    "this to confirm your inputs are actually "
                    "reaching the prediction."
                )

                st.dataframe(
                    processed_input,
                    use_container_width=True
                )

            if prediction:

                with st.container():

                    st.markdown(
                        f"""
                        <div class="risk-card-high">
                            <div class="risk-label-high">
                                ⚠️ HIGH CHURN RISK
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.write("")

                col1, col2 = st.columns([1, 2])

                with col1:

                    st.metric(
                        "Churn Probability",
                        f"{probability_percent:.1f}%"
                    )

                with col2:

                    st.warning(
                        "**Recommended Action**\n\n"
                        "Prioritize this customer for a "
                        "retention campaign and review their "
                        "engagement, product usage, and "
                        "account activity."
                    )

            else:

                with st.container():

                    st.markdown(
                        f"""
                        <div class="risk-card-low">
                            <div class="risk-label-low">
                                ✅ LOW CHURN RISK
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.write("")

                col1, col2 = st.columns([1, 2])

                with col1:

                    st.metric(
                        "Churn Probability",
                        f"{probability_percent:.1f}%"
                    )

                with col2:

                    st.info(
                        "**Recommended Action**\n\n"
                        "Continue normal customer engagement "
                        "and monitor future behavioral changes."
                    )


        except Exception as e:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(e)


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="section-label">Model Comparison</div>',
        unsafe_allow_html=True
    )

    st.header(
        "Model Performance"
    )

    st.write(
        "Comparison of the five machine-learning models "
        "evaluated for customer churn prediction."
    )

    st.write("")

    # ========================================================
    # MODEL PERFORMANCE DATA
    # ========================================================

    results = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest",
            "SVM Polynomial",
            "SVM RBF",
            "XGBoost"
        ],

        "Recall": [
            0.6520416711,
            0.6604278075,
            0.6711229947,
            0.7112229465,
            0.4304812834
        ],

        "Precision": [
            0.3283983849,
            0.4824218750,
            0.4960474308,
            0.4745153298,
            0.7488372093
        ],

        "F1": [
            0.4368845121,
            0.5575620767,
            0.5704545455,
            0.5689839572,
            0.5466893039
        ],

        "ROC-AUC": [
            0.7430787142,
            0.8306924246,
            0.8174833225,
            0.8302681689,
            0.8394883280
        ]
    })


    # ========================================================
    # BEST MODEL INDICATORS
    # ========================================================

    best_recall_model = results.loc[
        results["Recall"].idxmax(),
        "Model"
    ]

    best_precision_model = results.loc[
        results["Precision"].idxmax(),
        "Model"
    ]

    best_f1_model = results.loc[
        results["F1"].idxmax(),
        "Model"
    ]

    best_roc_model = results.loc[
        results["ROC-AUC"].idxmax(),
        "Model"
    ]


    # ========================================================
    # BEST-PERFORMING MODELS
    # ========================================================

    with st.container(border=True):

        st.subheader(
            "🏆 Best-Performing Models"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            kpi_card(
                "🎯",
                "Best Recall",
                best_recall_model,
                "kpi-blue"
            )

        with c2:

            kpi_card(
                "🔎",
                "Best Precision",
                best_precision_model,
                "kpi-purple"
            )

        with c3:

            kpi_card(
                "⚖️",
                "Best F1",
                best_f1_model,
                "kpi-green"
            )

        with c4:

            kpi_card(
                "📈",
                "Best ROC-AUC",
                best_roc_model,
                "kpi-cyan"
            )


    st.write("")


    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    with st.container(border=True):

        st.subheader(
            "📋 Comparison Table"
        )

        styled_results = (
            results.style
            .highlight_max(
                subset=[
                    "Recall",
                    "Precision",
                    "F1",
                    "ROC-AUC"
                ],
                color="rgba(34, 211, 238, 0.25)"
            )
            .format({
                "Recall": "{:.3f}",
                "Precision": "{:.3f}",
                "F1": "{:.3f}",
                "ROC-AUC": "{:.3f}"
            })
        )

        st.dataframe(
            styled_results,
            use_container_width=True,
            hide_index=True
        )


    st.write("")


    # ========================================================
    # PERFORMANCE COMPARISON CHART
    # ========================================================

    with st.container(border=True):

        st.subheader(
            "📊 Performance Comparison"
        )

        # Convert the dataframe from wide format
        # to long format for Plotly.
        chart_data = results.melt(
            id_vars="Model",
            value_vars=[
                "Recall",
                "Precision",
                "F1",
                "ROC-AUC"
            ],
            var_name="Metric",
            value_name="Score"
        )


        # ----------------------------------------------------
        # Plotly grouped bar chart
        # ----------------------------------------------------

        fig = px.bar(
            chart_data,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text="Score"
        )


        # ----------------------------------------------------
        # Chart formatting
        # ----------------------------------------------------

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            cliponaxis=False
        )


        fig.update_layout(

            height=520,

            xaxis=dict(
                title=None,
                tickangle=0,
                categoryorder="array",
                categoryarray=[
                    "Logistic Regression",
                    "Random Forest",
                    "SVM Polynomial",
                    "SVM RBF",
                    "XGBoost"
                ],
                tickfont=dict(
                    size=12
                )
            ),

            yaxis=dict(
                title="Score",
                range=[0, 1],
                tickformat=".1f"
            ),

            legend=dict(
                title="Metric",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),

            margin=dict(
                l=30,
                r=30,
                t=60,
                b=80
            ),

            bargap=0.18,
            bargroupgap=0.08
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Model comparison insight
        # ----------------------------------------------------

        callout(
            f"{best_roc_model} achieves the highest ROC-AUC "
            f"({results['ROC-AUC'].max():.3f}), while "
            f"{best_recall_model} achieves the highest Recall "
            f"({results['Recall'].max():.3f})."
        )