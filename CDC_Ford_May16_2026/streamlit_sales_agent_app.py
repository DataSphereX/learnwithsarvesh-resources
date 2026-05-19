import base64
import os
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    START = None
    END = None

# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# STATE DEFINITION FOR MULTI-AGENT ORCHESTRATION
# ============================================================================

class AnalystState(TypedDict):
    """Shared state for multi-agent pipeline."""
    question: str
    route: str
    answer: str
    reasoning: str
    agent_name: str


st.set_page_config(
    page_title="Chintu, the AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = Path(r"D:\Mentoring\learwithsarvesh\git_repos\learnwithsarvesh\lws_logo.png")

st.markdown(
    """
    <style>
    :root {
        --bg: #07151f;
        --panel: rgba(8, 23, 34, 0.82);
        --panel-strong: rgba(11, 31, 46, 0.95);
        --border: rgba(104, 185, 189, 0.22);
        --text: #ecf7f7;
        --muted: #9db4b8;
        --accent: #4cd1c5;
        --accent-2: #7fffd4;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(76, 209, 197, 0.16), transparent 30%),
            radial-gradient(circle at bottom right, rgba(16, 61, 89, 0.35), transparent 35%),
            linear-gradient(180deg, #061019 0%, #091a25 55%, #051018 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(135deg, rgba(9, 32, 46, 0.92), rgba(4, 15, 23, 0.86));
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        line-height: 1.05;
        color: var(--text);
    }

    .hero p {
        margin: 0.45rem 0 0;
        color: var(--muted);
        font-size: 1rem;
    }

    .subtle-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        backdrop-filter: blur(8px);
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.8rem 0 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(13, 39, 55, 0.96), rgba(8, 23, 34, 0.9));
        border: 1px solid rgba(76, 209, 197, 0.18);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        min-height: 92px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text);
        word-break: break-word;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(6, 20, 30, 0.96), rgba(9, 27, 39, 0.95));
        border-right: 1px solid rgba(76, 209, 197, 0.16);
    }

    section[data-testid="stSidebar"] .stImage {
        border-radius: 16px;
    }

    .logo-card {
        background: linear-gradient(180deg, #f8fbfd 0%, #eef4f8 100%);
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 24px;
        padding: 1rem 1.1rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
    }

    .chat-bubble {
        border: 1px solid rgba(76, 209, 197, 0.18);
        border-radius: 18px;
        padding: 0.8rem 0.95rem;
        margin: 0.35rem 0 0.8rem;
        background: rgba(10, 26, 38, 0.82);
    }

    .route-label {
        display: inline-block;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.4rem;
    }

    .stChatMessage {
        background: transparent !important;
    }

    .stButton>button {
        border-radius: 999px;
        border: 1px solid rgba(76, 209, 197, 0.28);
        background: linear-gradient(135deg, rgba(24, 72, 86, 0.98), rgba(14, 40, 58, 0.96));
        color: #f1fbfb;
        padding: 0.6rem 1rem;
        transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        border-color: rgba(127, 255, 212, 0.5);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
    }

    .stTextInput>div>div>input,
    .stFileUploader,
    .stDataFrame,
    .stMarkdown,
    .stSelectbox,
    .stMultiSelect {
        color: var(--text);
    }

    .footer-note {
        color: var(--muted);
        font-size: 0.85rem;
        margin-top: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip().lower().replace(" ", "_") for column in cleaned.columns]
    unnamed_columns = [column for column in cleaned.columns if column.startswith("unnamed")]
    if unnamed_columns:
        cleaned = cleaned.drop(columns=unnamed_columns)
    return cleaned


def _prepare_sales_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = _clean_columns(df)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "revenue" not in df.columns and {"quantity", "unit_price"}.issubset(df.columns):
        df["revenue"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0) * pd.to_numeric(
            df["unit_price"], errors="coerce"
        ).fillna(0)

    if "month" not in df.columns and "date" in df.columns:
        df["month"] = df["date"].dt.to_period("M").astype(str)

    if "quarter" not in df.columns and "date" in df.columns:
        df["quarter"] = "Q" + df["date"].dt.quarter.astype("Int64").astype(str)

    if "components" in df.columns and "component_count" not in df.columns:
        df["component_count"] = df["components"].fillna("").astype(str).apply(
            lambda value: len([item for item in value.split(",") if item.strip()])
        )

    return df


def _infer_columns(df: pd.DataFrame) -> Dict[str, str]:
    candidates = {
        "date": ["date", "order_date", "created_at"],
        "region": ["region", "market", "territory"],
        "product": ["product", "item", "sku"],
        "sales_rep": ["sales_rep", "salesperson", "rep", "owner"],
        "quantity": ["quantity", "qty", "units"],
        "revenue": ["revenue", "sales", "amount", "total_sales"],
        "components": ["components", "component_list", "parts"],
        "trend_type": ["trend_type", "trend"],
        "component_count": ["component_count", "num_components"],
    }

    mapping: Dict[str, str] = {}
    for canonical_name, options in candidates.items():
        for option in options:
            if option in df.columns:
                mapping[canonical_name] = option
                break
    return mapping


def _format_value(value: float, metric_name: str) -> str:
    if metric_name == "revenue":
        return f"${value:,.0f}"
    return f"{value:,.0f}"


def _top_summary(df: pd.DataFrame, columns: Dict[str, str]) -> str:
    pieces: List[str] = []

    if columns.get("revenue"):
        revenue_series = pd.to_numeric(df[columns["revenue"]], errors="coerce").fillna(0)
        pieces.append(f"Total revenue: {_format_value(float(revenue_series.sum()), 'revenue')}")

    if columns.get("product") and columns.get("revenue"):
        product_revenue = df.groupby(columns["product"])[columns["revenue"]].sum().sort_values(ascending=False)
        if not product_revenue.empty:
            pieces.append(
                f"Top product: {product_revenue.index[0]} ({_format_value(float(product_revenue.iloc[0]), 'revenue')})"
            )

    if columns.get("region") and columns.get("revenue"):
        region_revenue = df.groupby(columns["region"])[columns["revenue"]].sum().sort_values(ascending=False)
        if not region_revenue.empty:
            pieces.append(
                f"Top region: {region_revenue.index[0]} ({_format_value(float(region_revenue.iloc[0]), 'revenue')})"
            )

    return " | ".join(pieces) if pieces else "No summary available yet."


def _chart_data(df: pd.DataFrame, columns: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    charts: Dict[str, pd.DataFrame] = {}

    if columns.get("date") and columns.get("revenue"):
        monthly = (
            df.assign(month=df[columns["date"]].dt.to_period("M").astype(str))
            .groupby("month")[columns["revenue"]]
            .sum()
            .reset_index()
        )
        monthly.columns = ["month", "revenue"]
        charts["monthly_revenue"] = monthly

    if columns.get("product") and columns.get("revenue"):
        product_revenue = (
            df.groupby(columns["product"])[columns["revenue"]].sum().sort_values(ascending=False).reset_index()
        )
        product_revenue.columns = ["product", "revenue"]
        charts["product_revenue"] = product_revenue

    if columns.get("region") and columns.get("revenue"):
        region_revenue = (
            df.groupby(columns["region"])[columns["revenue"]].sum().sort_values(ascending=False).reset_index()
        )
        region_revenue.columns = ["region", "revenue"]
        charts["region_revenue"] = region_revenue

    return charts


def _render_header() -> None:
    left, right = st.columns([3.8, 1.4], vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="hero">
                <h1>Chintu, the AI Data Analyst</h1>
                <p>Upload a sales CSV, ask business questions, and let the agent route each question to the right specialist.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if LOGO_PATH.exists():
            logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
            st.markdown(
                f"""
                <div class="logo-card" style="display:flex; align-items:center; justify-content:center; min-height:160px;">
                    <img src="data:image/png;base64,{logo_b64}" style="max-width:100%; height:auto; display:block; object-fit:contain; mix-blend-mode:multiply;" />
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="subtle-card" style="text-align:center; padding:1.25rem;">
                    <div style="font-size:1.2rem; font-weight:700; color:#ecf7f7;">Learn With Sarvesh</div>
                    <div style="color:#9db4b8; margin-top:0.3rem;">Courses, clarity, confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_metrics(df: pd.DataFrame, columns: Dict[str, str]) -> None:
    total_revenue = None
    top_product = None
    top_region = None

    if columns.get("revenue"):
        total_revenue = _format_value(float(pd.to_numeric(df[columns["revenue"]], errors="coerce").fillna(0).sum()), "revenue")

    if columns.get("product") and columns.get("revenue"):
        product_revenue = df.groupby(columns["product"])[columns["revenue"]].sum().sort_values(ascending=False)
        if not product_revenue.empty:
            top_product = f"{product_revenue.index[0]}  |  {_format_value(float(product_revenue.iloc[0]), 'revenue')}"

    if columns.get("region") and columns.get("revenue"):
        region_revenue = df.groupby(columns["region"])[columns["revenue"]].sum().sort_values(ascending=False)
        if not region_revenue.empty:
            top_region = f"{region_revenue.index[0]}  |  {_format_value(float(region_revenue.iloc[0]), 'revenue')}"

    metric_html = f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total rows</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total revenue</div>
            <div class="metric-value">{total_revenue or 'Not available'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Top product</div>
            <div class="metric-value">{top_product or 'Not available'}</div>
        </div>
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)

    if top_region:
        st.markdown(
            f"""
            <div class="subtle-card">
                <div class="metric-label">Top region</div>
                <div class="metric-value">{top_region}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _detect_intent_fallback(question: str) -> str:
    """Fallback keyword-based intent detection."""
    text = question.lower()

    if any(word in text for word in ["why", "drop", "decrease", "anomaly", "issue", "problem", "down"]):
        return "anomaly"
    if any(word in text for word in ["trend", "growth", "growing", "over time", "monthly", "pattern"]):
        return "trend"
    if any(word in text for word in ["region", "product", "sales rep", "rep", "breakdown", "compare", "which"]):
        return "breakdown"
    if any(word in text for word in ["recommend", "should we", "what should", "action", "do next", "fix"]):
        return "recommendation"
    if any(word in text for word in ["component", "parts", "component count"]):
        return "component"
    return "summary"


def _detect_intent(question: str) -> str:
    """LLM-based intent detection using GPT-4o-mini."""
    try:
        prompt = f"""You are an expert at categorizing business analyst questions.

Classify this analyst question into one of these categories:
1. 'anomaly' - Questions about unusual drops, problems, anomalies, or unexpected patterns
2. 'trend' - Questions about trends, growth, declining products, or performance over time
3. 'breakdown' - Questions asking for breakdowns by region, product, sales rep, or dimensions
4. 'recommendation' - Questions asking for recommendations, actions, strategies, or what to do
5. 'component' - Questions about products with specific components or complexity
6. 'summary' - General questions or when unsure

Question: "{question}"

Respond with ONLY the category name (e.g., 'anomaly', 'trend', etc.)"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert business intelligence classifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=20
        )
        
        intent = response.choices[0].message.content.strip().lower()
        
        # Validate intent is one of our categories
        valid_intents = {"anomaly", "trend", "breakdown", "recommendation", "component", "summary"}
        if intent in valid_intents:
            return intent
        else:
            return _detect_intent_fallback(question)
    
    except Exception as e:
        # Fallback to keyword-based detection if LLM fails
        return _detect_intent_fallback(question)


def _monthly_metric(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    if "date" not in columns:
        return pd.DataFrame()

    metric_name = columns.get("revenue") or columns.get("quantity")
    if not metric_name:
        return pd.DataFrame()

    working = df.copy()
    working["year_month"] = working[columns["date"]].dt.to_period("M").astype(str)
    monthly = working.groupby("year_month")[metric_name].sum().reset_index()
    monthly.columns = ["month", "metric"]
    return monthly


def _answer_anomaly(df: pd.DataFrame, columns: Dict[str, str]) -> str:
    monthly = _monthly_metric(df, columns)
    if monthly.empty:
        return "I need a date column plus revenue or quantity to detect anomalies."

    avg_metric = monthly["metric"].mean()
    worst_row = monthly.loc[monthly["metric"].idxmin()]
    best_row = monthly.loc[monthly["metric"].idxmax()]
    gap_pct = ((avg_metric - worst_row["metric"]) / avg_metric) * 100 if avg_metric else 0

    return (
        f"**Anomaly finding**\n\n"
        f"Worst month: {worst_row['month']} with {_format_value(float(worst_row['metric']), 'revenue')}\n"
        f"Average monthly value: {_format_value(float(avg_metric), 'revenue')}\n"
        f"Gap vs average: {gap_pct:.1f}%\n"
        f"Best month: {best_row['month']}\n\n"
        f"Analyst takeaway: this is the first place to investigate for a root cause."
    )


def _answer_trend(df: pd.DataFrame, columns: Dict[str, str]) -> str:
    product_column = columns.get("product")
    metric_column = columns.get("revenue") or columns.get("quantity")
    if not product_column or not metric_column:
        return "I need product and revenue or quantity columns to show a trend."

    if "date" not in columns:
        product_totals = df.groupby(product_column)[metric_column].sum().sort_values(ascending=False)
        lines = ["**Product performance snapshot**\n\n"]
        for product, metric in product_totals.items():
            lines.append(f"- {product}: {_format_value(float(metric), metric_column)}\n")
        return "".join(lines)

    working = df.copy()
    working["year_month"] = working[columns["date"]].dt.to_period("M")
    monthly = working.groupby(["year_month", product_column])[metric_column].sum().unstack(fill_value=0)

    if monthly.empty:
        return "No trend data available."

    trend_items = []
    for product in monthly.columns:
        series = monthly[product]
        first_value = float(series.iloc[0])
        last_value = float(series.iloc[-1])
        pct_change = ((last_value - first_value) / first_value) * 100 if first_value else 0
        trend_items.append((product, pct_change))

    trend_items.sort(key=lambda item: item[1], reverse=True)
    lines = ["**Trend analysis**\n\n"]
    for product, pct_change in trend_items:
        lines.append(f"- {product}: {pct_change:+.1f}%\n")

    lines.append("\nAnalyst takeaway: compare the growth leader against the declining product and ask why the mix changed."
    )
    return "".join(lines)


def _answer_breakdown(df: pd.DataFrame, columns: Dict[str, str]) -> str:
    parts = ["**Business breakdown**\n\n"]
    has_output = False

    if columns.get("region") and columns.get("revenue"):
        region_perf = df.groupby(columns["region"])[columns["revenue"]].sum().sort_values(ascending=False)
        total = float(region_perf.sum()) or 1.0
        parts.append("**By region**\n")
        for region, revenue in region_perf.items():
            pct = (float(revenue) / total) * 100
            parts.append(f"- {region}: {_format_value(float(revenue), 'revenue')} ({pct:.1f}%)\n")
        has_output = True

    if columns.get("product") and columns.get("revenue"):
        product_perf = df.groupby(columns["product"])[columns["revenue"]].sum().sort_values(ascending=False)
        parts.append("\n**By product**\n")
        for product, revenue in product_perf.items():
            parts.append(f"- {product}: {_format_value(float(revenue), 'revenue')}\n")
        has_output = True

    if columns.get("sales_rep") and columns.get("revenue"):
        rep_perf = df.groupby(columns["sales_rep"])[columns["revenue"]].sum().sort_values(ascending=False)
        parts.append("\n**By sales rep**\n")
        for rep, revenue in rep_perf.head(5).items():
            parts.append(f"- {rep}: {_format_value(float(revenue), 'revenue')}\n")
        has_output = True

    if not has_output:
        return "I need region, product, sales rep, and revenue columns to show a breakdown."

    parts.append("\nAnalyst takeaway: this tells you where to focus without opening multiple dashboards.")
    return "".join(parts)


def _answer_recommendation(df: pd.DataFrame, columns: Dict[str, str]) -> str:
    product_column = columns.get("product")
    revenue_column = columns.get("revenue")
    if not product_column or not revenue_column:
        return "I need product and revenue columns to make recommendations."

    product_perf = df.groupby(product_column)[revenue_column].sum().sort_values(ascending=False)
    if product_perf.empty:
        return "I couldn't find enough product data to make recommendations."

    top_product = product_perf.index[0]
    weak_product = product_perf.index[-1]

    recommendations = [
        f"1. Double down on **{top_product}** because it is your strongest product.",
        f"2. Review **{weak_product}** for pricing, quality, or demand issues.",
    ]

    if columns.get("region"):
        region_perf = df.groupby(columns["region"])[revenue_column].sum().sort_values(ascending=False)
        recommendations.append(f"3. Replicate the playbook from **{region_perf.index[0]}** into weaker regions.")

    return "**Recommended actions**\n\n" + "\n".join(recommendations) + "\n\nAnalyst takeaway: this is what you tell your manager or business lead."


def _answer_component(df: pd.DataFrame, columns: Dict[str, str]) -> str:
    parts = ["**Component analysis**\n\n"]
    has_output = False

    if columns.get("component_count") and columns.get("product"):
        counts = df.groupby(columns["product"])[columns["component_count"]].max().sort_values(ascending=False)
        parts.append("Component count by product:\n")
        for product, count in counts.items():
            parts.append(f"- {product}: {int(count)} components\n")
        has_output = True

    if columns.get("components") and columns.get("product"):
        parts.append("\nSample component mapping:\n")
        sample = df[[columns["product"], columns["components"]]].drop_duplicates().head(5)
        for _, row in sample.iterrows():
            parts.append(f"- {row[columns['product']]}: {row[columns['components']]}\n")
        has_output = True

    if not has_output:
        return "I need component-related columns to analyze this question."

    parts.append("\nAnalyst takeaway: use this when the question is about product complexity or routing to a specialist agent.")
    return "".join(parts)


def _answer_summary(df: pd.DataFrame, columns: Dict[str, str]) -> str:
    summary = _top_summary(df, columns)
    return f"**Quick summary**\n\n{summary}"


def _humanize_with_business_translator(technical_answer: str, question: str, agent_type: str) -> str:
    """
    Business Translator Agent: Converts technical specialist output to business language.
    Adds Executive Summary, Key Findings, Recommended Actions, and Follow-up Questions.
    """
    try:
        prompt = f"""You are a business communications expert for sales analysts.

Your job: Transform technical data into clear business language that executives understand.

TECHNICAL DATA FROM SPECIALIST AGENT:
{technical_answer}

ANALYST'S ORIGINAL QUESTION:
"{question}"

AGENT TYPE: {agent_type}

Create a response with:

1. EXECUTIVE SUMMARY (2-3 sentences in strict business terms - no jargon)
2. KEY FINDINGS (2-3 bullet points with specific business impact)
3. RECOMMENDED ACTIONS (2-3 concrete next steps)
4. FOLLOW-UP QUESTIONS (2-3 questions the analyst should investigate)

Format exactly as:

EXECUTIVE SUMMARY
[Your summary]

KEY FINDINGS
• [Finding 1]
• [Finding 2]
• [Finding 3]

RECOMMENDED ACTIONS
1. [Action 1]
2. [Action 2]
3. [Action 3]

FOLLOW-UP QUESTIONS
1. [Question 1]
2. [Question 2]
3. [Question 3]

Remember: Use business language. Be specific. Be actionable."""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business communication specialist who translates data into executive insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        # If translator fails, return the technical answer with a note
        return f"{technical_answer}\n\n*(Note: Business translation unavailable - showing raw analysis)*"


# ============================================================================
# STATGRAPH AGENT NODES
# ============================================================================

def _route_node(state: AnalystState, sales_df: pd.DataFrame, sales_columns: Dict[str, str]) -> AnalystState:
    """Node 1: Router - Detects intent and routes to specialist."""
    intent = _detect_intent(state["question"])
    route, reasoning = None, "Intent detected"
    
    # Map intent to agent name
    if intent == "anomaly":
        route = "anomaly_agent"
        agent_name = "Anomaly Detective"
    elif intent == "trend":
        route = "trend_agent"
        agent_name = "Trend Analyst"
    elif intent == "breakdown":
        route = "breakdown_agent"
        agent_name = "Breakdown Expert"
    elif intent == "recommendation":
        route = "recommendation_agent"
        agent_name = "Recommendation Strategist"
    elif intent == "component":
        route = "component_agent"
        agent_name = "Component Specialist"
    else:
        route = "summary_agent"
        agent_name = "Summary Agent"
    
    state["route"] = route
    state["agent_name"] = agent_name
    state["reasoning"] = reasoning
    return state


def _specialist_node(state: AnalystState, sales_df: pd.DataFrame, sales_columns: Dict[str, str]) -> AnalystState:
    """Node 2: Specialist Agent - Performs technical analysis based on route."""
    route = state["route"]
    
    if route == "anomaly_agent":
        answer = _answer_anomaly(sales_df, sales_columns)
    elif route == "trend_agent":
        answer = _answer_trend(sales_df, sales_columns)
    elif route == "breakdown_agent":
        answer = _answer_breakdown(sales_df, sales_columns)
    elif route == "recommendation_agent":
        answer = _answer_recommendation(sales_df, sales_columns)
    elif route == "component_agent":
        answer = _answer_component(sales_df, sales_columns)
    else:
        answer = _answer_summary(sales_df, sales_columns)
    
    state["answer"] = answer
    return state


def _translator_node(state: AnalystState) -> AnalystState:
    """Node 3: Business Translator - Humanizes specialist output."""
    technical_answer = state["answer"]
    question = state["question"]
    agent_name = state["agent_name"]
    
    # Pass through translator
    business_answer = _humanize_with_business_translator(technical_answer, question, agent_name)
    state["answer"] = business_answer
    state["route"] = f"{state['route']} → Business Translator"
    
    return state


def _build_agent_graph(sales_df: pd.DataFrame, sales_columns: Dict[str, str]):
    """Build the StateGraph for multi-agent orchestration."""
    if not LANGGRAPH_AVAILABLE:
        return None
    
    graph = StateGraph(AnalystState)
    
    # Define nodes with closure to capture sales_df and sales_columns
    def route_node(state):
        return _route_node(state, sales_df, sales_columns)
    
    def specialist_node(state):
        return _specialist_node(state, sales_df, sales_columns)
    
    def translator_node(state):
        return _translator_node(state)
    
    # Add nodes
    graph.add_node("route", route_node)
    graph.add_node("specialist", specialist_node)
    graph.add_node("translator", translator_node)
    
    # Add edges (pipeline: route → specialist → translator → end)
    graph.add_edge(START, "route")
    graph.add_edge("route", "specialist")
    graph.add_edge("specialist", "translator")
    graph.add_edge("translator", END)
    
    return graph.compile()


def _run_agent_graph(question: str, sales_df: pd.DataFrame, sales_columns: Dict[str, str]) -> AnalystState:
    """Execute the agent graph pipeline."""
    # Try to use StateGraph if available
    if LANGGRAPH_AVAILABLE:
        graph = _build_agent_graph(sales_df, sales_columns)
        if graph:
            initial_state = {
                "question": question,
                "route": "",
                "answer": "",
                "reasoning": "",
                "agent_name": ""
            }
            result = graph.invoke(initial_state)
            return result
    
    # Fallback to direct function calls if StateGraph not available
    state = {
        "question": question,
        "route": "",
        "answer": "",
        "reasoning": "",
        "agent_name": ""
    }
    state = _route_node(state, sales_df, sales_columns)
    state = _specialist_node(state, sales_df, sales_columns)
    state = _translator_node(state)
    return state


def answer_question(df: pd.DataFrame, question: str, columns: Dict[str, str]) -> Tuple[str, str]:
    """Execute the full multi-agent pipeline using StateGraph."""
    result = _run_agent_graph(question, df, columns)
    return result["route"], result["answer"]


def _suggested_questions(columns: Dict[str, str]) -> List[str]:
    suggestions = [
        "Why is revenue down in Q2?",
        "Which product is growing fastest and which one is declining?",
        "Which region is performing best and which region needs attention?",
        "What should we do next to improve revenue?",
    ]

    if columns.get("components") or columns.get("component_count"):
        suggestions.append("Which product has the highest component complexity?")

    return suggestions


def _process_question(question: str) -> None:
    sales_df = st.session_state.sales_df
    if sales_df is None:
        return

    sales_columns = st.session_state.sales_columns
    intent, answer = answer_question(sales_df, question, sales_columns)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"{answer}\n\n*Routed to: {intent.replace('_', ' ').title()}*",
            "route": intent.replace("_", " ").title(),
        }
    )


_render_header()
st.caption("Upload a sales CSV and ask business questions in plain English.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sales_df" not in st.session_state:
    st.session_state.sales_df = None
if "sales_columns" not in st.session_state:
    st.session_state.sales_columns = {}
if "file_name" not in st.session_state:
    st.session_state.file_name = None

with st.sidebar:
    st.header("How to use")
    st.write("1. Upload a sales CSV")
    st.write("2. Ask analyst-style questions")
    st.write("3. Read the routed answer")
    st.divider()
    st.subheader("Suggested questions")
    if st.session_state.sales_columns:
        for item in _suggested_questions(st.session_state.sales_columns):
            st.write(f"- {item}")
    else:
        st.write("Upload a file to see suggested questions.")
        st.markdown('<div class="footer-note">Built for live demos and analyst walkthroughs.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload sales CSV",
    type=["csv"],
    help="Expected columns: date, product, region, revenue, quantity, sales_rep, components, component_count, trend_type",
)

if uploaded_file is not None and st.session_state.file_name != uploaded_file.name:
    try:
        st.session_state.sales_df = _prepare_sales_data(uploaded_file)
        st.session_state.sales_columns = _infer_columns(st.session_state.sales_df)
        st.session_state.file_name = uploaded_file.name
        st.session_state.messages = []
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")

if st.session_state.sales_df is not None:
    sales_df = st.session_state.sales_df
    sales_columns = st.session_state.sales_columns

    _render_metrics(sales_df, sales_columns)

    charts = _chart_data(sales_df, sales_columns)

    tab_one, tab_two, tab_three = st.tabs(["Revenue trend", "Product mix", "Regional view"])

    with tab_one:
        if "monthly_revenue" in charts:
            st.line_chart(charts["monthly_revenue"].set_index("month"), y="revenue", use_container_width=True)
        else:
            st.info("A date and revenue column are needed to show a monthly revenue trend.")

    with tab_two:
        if "product_revenue" in charts:
            st.bar_chart(charts["product_revenue"].set_index("product"), y="revenue", use_container_width=True)
        else:
            st.info("A product and revenue column are needed to show product mix.")

    with tab_three:
        if "region_revenue" in charts:
            st.bar_chart(charts["region_revenue"].set_index("region"), y="revenue", use_container_width=True)
        else:
            st.info("A region and revenue column are needed to show the regional view.")

    with st.expander("Preview data", expanded=False):
        st.dataframe(sales_df.head(10), use_container_width=True)

    st.divider()
    st.subheader("Chat with the analyst agent")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("route"):
                st.markdown(f"<span class='route-label'>Agent route: {message['route']}</span>", unsafe_allow_html=True)
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about the sales data...")
    if user_question:
        _process_question(user_question)
        st.rerun()

    st.divider()
    st.subheader("Quick prompts")
    prompt_columns = st.columns(2)
    suggestions = _suggested_questions(sales_columns)

    for index, prompt in enumerate(suggestions[:4]):
        if prompt_columns[index % 2].button(prompt, use_container_width=True, key=f"prompt_{index}_{prompt}"):
            _process_question(prompt)
            st.rerun()
else:
    st.info("Upload your sales CSV to start the chatbot.")
    st.write("If you upload the generated sales file from your notebook, the app will also use the component and trend fields when available.")
