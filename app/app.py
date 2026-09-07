import sys
import re
from html import escape
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# PROJECT SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv()


# ============================================================
# PROJECT MODULES
# ============================================================

from src.data.loader import load_data
from src.ml.predict import predict_risk
from src.explainability.shap_explainer import explain_prediction
from src.rules.rule_engine import load_business_rules

from src.talk_to_data.nl_to_sql import generate_sql
from src.talk_to_data.sql_validator import validate_sql
from src.talk_to_data.query_runner import run_query
from src.talk_to_data.answer_generator import generate_business_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="CR",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# THEME / PREMIUM UI
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --navy: #102A43;
        --navy-2: #173B5E;
        --ink: #17202A;
        --muted: #667788;
        --line: #DCE4EA;
        --surface: #FFFFFF;
        --bg: #F4F7F9;
        --green: #20A36A;
        --green-dark: #14754B;
        --green-soft: #E8F7F0;
        --amber: #D69E2E;
        --amber-soft: #FFF7E1;
        --red: #C94A4A;
        --red-soft: #FDECEC;
        --blue: #2B78C5;
        --blue-soft: #EAF3FB;
    }

    /* ---------- Base ---------- */

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(32,163,106,0.07), transparent 24%),
            radial-gradient(circle at 92% 8%, rgba(43,120,197,0.06), transparent 22%),
            var(--bg);
        color: var(--ink);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    /* ---------- Hero ---------- */

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #102A43 0%, #173B5E 58%, #176B53 100%);
        border-radius: 22px;
        padding: 28px 32px 30px;
        margin-bottom: 18px;
        box-shadow: 0 18px 45px rgba(16, 42, 67, 0.15);
        animation: heroIn 0.65s ease-out both;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -90px;
        top: -125px;
        border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }

    .hero-kicker {
        color: #BFEBD8;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 7px;
    }

    .hero-title {
        color: white;
        font-size: 2.45rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.035em;
        margin: 0;
    }

    .hero-subtitle {
        color: #DDE9F2;
        font-size: 0.98rem;
        margin-top: 10px;
    }

    .hero-chip-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .hero-chip {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.15);
        color: #F3F8FB;
        border-radius: 999px;
        padding: 7px 11px;
        font-size: 0.78rem;
        backdrop-filter: blur(8px);
    }

    /* ---------- Navigation ---------- */

    button[data-baseweb="tab"] {
        font-size: 0.91rem;
        font-weight: 750;
        color: #5A6B79;
        padding: 12px 22px;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--green-dark);
    }

    [data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, var(--green), #48C98F) !important;
        height: 3px !important;
    }

    /* ---------- Section headers ---------- */

    .section-head {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 18px;
        margin: 22px 0 12px;
    }

    .section-title {
        color: var(--navy);
        font-size: 1.42rem;
        line-height: 1.2;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .section-caption {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: 5px;
    }

    /* ---------- KPI cards ---------- */

    .kpi-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 19px;
        min-height: 118px;
        box-shadow: 0 8px 22px rgba(16, 42, 67, 0.05);
        animation: riseIn 0.55s ease-out both;
    }

    .kpi-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.80rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-value {
        color: var(--navy);
        font-size: 1.78rem;
        font-weight: 820;
        margin-top: 7px;
        letter-spacing: -0.025em;
    }

    .kpi-note {
        color: #788795;
        font-size: 0.78rem;
        margin-top: 3px;
    }

    .kpi-accent-green {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: var(--green);
        box-shadow: 0 0 0 5px var(--green-soft);
    }

    .kpi-accent-blue {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: var(--blue);
        box-shadow: 0 0 0 5px var(--blue-soft);
    }

    .kpi-accent-amber {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: var(--amber);
        box-shadow: 0 0 0 5px var(--amber-soft);
    }

    .kpi-accent-red {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: var(--red);
        box-shadow: 0 0 0 5px var(--red-soft);
    }

    /* ---------- Cards ---------- */

    .panel,
    .reason-card,
    .rule-card,
    .answer-card,
    .profile-card,
    .insight-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(16, 42, 67, 0.045);
    }

    .panel {
        padding: 18px;
    }

    .profile-card {
        padding: 17px 18px;
        min-height: 144px;
    }

    .profile-label {
        color: #7A8997;
        font-size: 0.77rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .profile-value {
        color: var(--navy);
        font-size: 1.05rem;
        font-weight: 780;
        margin-top: 7px;
    }

    .profile-detail {
        color: #637282;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-top: 8px;
    }

    .reason-card {
        padding: 17px 18px;
        margin-bottom: 11px;
        border-left: 4px solid var(--red);
    }

    .reason-card.positive {
        border-left-color: var(--red);
    }

    .reason-card.negative {
        border-left-color: var(--green);
    }

    .reason-title,
    .rule-title {
        color: var(--navy);
        font-weight: 800;
        font-size: 0.95rem;
    }

    .reason-subtitle {
        color: #71808D;
        font-size: 0.77rem;
        margin-top: 3px;
    }

    .reason-value {
        color: var(--navy);
        font-weight: 780;
        font-size: 1.02rem;
        margin-top: 9px;
    }

    .reason-text,
    .rule-text {
        color: #5E6D7B;
        line-height: 1.55;
        font-size: 0.86rem;
        margin-top: 7px;
    }

    .impact-pill {
        display: inline-block;
        margin-top: 10px;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 750;
    }

    .impact-up {
        background: var(--red-soft);
        color: #A63D3D;
    }

    .impact-down {
        background: var(--green-soft);
        color: #18724A;
    }

    .answer-card {
        padding: 19px 20px;
        border-left: 5px solid var(--green);
        color: #253849;
        line-height: 1.65;
        font-size: 0.98rem;
    }

    .answer-meta {
        display: inline-block;
        background: var(--green-soft);
        color: var(--green-dark);
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 0.70rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .insight-card {
        padding: 16px 17px;
        min-height: 125px;
        margin-bottom: 10px;
    }

    .insight-number {
        color: var(--green-dark);
        font-size: 0.76rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .insight-title {
        color: var(--navy);
        font-size: 0.94rem;
        font-weight: 800;
        margin-top: 7px;
    }

    .insight-text {
        color: #637282;
        font-size: 0.82rem;
        line-height: 1.48;
        margin-top: 5px;
    }

    /* ---------- Buttons / controls ---------- */

    .stButton > button {
        border: 1px solid #C9D6DD;
        background: #FFFFFF;
        color: var(--navy);
        border-radius: 10px;
        font-weight: 750;
        min-height: 42px;
        box-shadow: 0 2px 7px rgba(16,42,67,0.04);
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: var(--green);
        box-shadow: 0 6px 14px rgba(32,163,106,0.14);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #176B53, #20A36A);
        color: white;
        border: none;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    textarea,
    input {
        border-radius: 10px !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 11px;
        overflow: hidden;
    }

    details {
        border: 1px solid var(--line) !important;
        border-radius: 11px !important;
        background: white !important;
    }

    [data-testid="stAlert"] {
        border-radius: 11px;
    }

    /* ---------- Status strip ---------- */

    .status-strip {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 15px;
        margin: 9px 0 18px;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: var(--green);
        border-radius: 50%;
        box-shadow: 0 0 0 4px var(--green-soft);
    }

    .status-text {
        color: #5C6B78;
        font-size: 0.80rem;
        margin-right: 16px;
    }

    /* ---------- Risk hero ---------- */

    .risk-banner {
        background:
            linear-gradient(135deg, rgba(16,42,67,0.98), rgba(23,107,83,0.96));
        color: white;
        border-radius: 19px;
        padding: 21px 23px;
        margin: 8px 0 18px;
        box-shadow: 0 13px 30px rgba(16,42,67,0.13);
        animation: riseIn 0.55s ease-out both;
    }

    .risk-kicker {
        color: #BFEBD8;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .risk-title {
        color: white;
        font-size: 1.65rem;
        font-weight: 820;
        margin-top: 3px;
    }

    .risk-copy {
        color: #DDEAEF;
        font-size: 0.86rem;
        line-height: 1.5;
        margin-top: 6px;
    }

    .risk-pill {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.76rem;
        font-weight: 760;
        margin-top: 12px;
    }

    /* ---------- Chat ---------- */

    .chat-user {
        background: #EAF3FB;
        border: 1px solid #D6E6F4;
        border-radius: 14px 14px 5px 14px;
        padding: 13px 15px;
        color: #21435F;
        margin: 8px 0 8px 18%;
    }

    .chat-assistant {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-left: 4px solid var(--green);
        border-radius: 14px 14px 14px 5px;
        padding: 14px 16px;
        color: #2C4050;
        margin: 8px 18% 12px 0;
        box-shadow: 0 7px 18px rgba(16,42,67,0.04);
    }

    .chat-label {
        color: #738390;
        font-size: 0.70rem;
        font-weight: 820;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 5px;
    }

    .source-badge {
        display: inline-block;
        background: #F2F6F8;
        color: #667784;
        border: 1px solid #DDE6EB;
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 0.68rem;
        font-weight: 750;
    }

    /* ---------- Footer ---------- */

    .footer {
        margin-top: 34px;
        padding: 17px 0 5px;
        color: #7C8A96;
        font-size: 0.75rem;
        text-align: center;
    }

    /* ---------- Premium Motion ---------- */

    .hero {
        animation: heroIn 0.75s cubic-bezier(.22,.61,.36,1) both;
    }

    .hero:before {
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        left: -120px;
        bottom: -230px;
        border-radius: 50%;
        background: rgba(32,163,106,0.13);
        filter: blur(3px);
        animation: orbDrift 9s ease-in-out infinite alternate;
        pointer-events: none;
    }

    .hero-chip {
        transition: transform .25s ease, background .25s ease, border-color .25s ease;
    }

    .hero-chip:hover {
        transform: translateY(-3px);
        background: rgba(255,255,255,0.16);
        border-color: rgba(255,255,255,0.26);
    }

    .kpi-card {
        animation:
            riseIn .58s cubic-bezier(.22,.61,.36,1) both,
            softFloat 5.5s ease-in-out 1.2s infinite alternate;
    }

    .kpi-card:nth-child(2) { animation-delay: .08s, 1.3s; }
    .kpi-card:nth-child(3) { animation-delay: .16s, 1.4s; }
    .kpi-card:nth-child(4) { animation-delay: .24s, 1.5s; }

    .profile-card,
    .reason-card,
    .rule-card,
    .answer-card,
    .insight-card,
    .panel {
        transition:
            transform .25s ease,
            box-shadow .25s ease,
            border-color .25s ease;
    }

    .profile-card:hover,
    .reason-card:hover,
    .rule-card:hover,
    .insight-card:hover {
        transform: translateY(-4px);
        border-color: #C5D5DE;
        box-shadow: 0 15px 30px rgba(16,42,67,0.10);
    }

    .answer-card:hover {
        box-shadow: 0 13px 27px rgba(32,163,106,0.10);
    }

    .reason-card:has(.impact-up) {
        animation: slideFromLeft .55s cubic-bezier(.22,.61,.36,1) both;
    }

    .reason-card:has(.impact-down) {
        animation: slideFromRight .55s cubic-bezier(.22,.61,.36,1) both;
    }

    .risk-banner {
        position: relative;
        overflow: hidden;
        animation: riskReveal .70s cubic-bezier(.22,.61,.36,1) both;
    }

    .risk-banner:after {
        content: "";
        position: absolute;
        top: 0;
        left: -35%;
        width: 30%;
        height: 100%;
        background: linear-gradient(
            100deg,
            transparent,
            rgba(255,255,255,0.10),
            transparent
        );
        transform: skewX(-18deg);
        animation: shine 4.2s ease-in-out 0.8s infinite;
        pointer-events: none;
    }

    .risk-pill {
        transition: transform .25s ease, background .25s ease;
    }

    .risk-pill:hover {
        transform: scale(1.04);
        background: rgba(255,255,255,0.18);
    }

    .status-dot {
        animation: pulseDot 2.1s ease-in-out infinite;
    }

    .chat-user {
        animation: chatIn .45s cubic-bezier(.22,.61,.36,1) both;
    }

    .chat-assistant {
        animation: chatInAssistant .52s cubic-bezier(.22,.61,.36,1) both;
    }

    .source-badge {
        transition: transform .22s ease;
    }

    .source-badge:hover {
        transform: translateY(-1px);
    }

    /* Streamlit buttons */
    .stButton > button {
        position: relative;
        overflow: hidden;
        transition:
            transform .2s ease,
            box-shadow .2s ease,
            border-color .2s ease,
            background .2s ease !important;
    }

    .stButton > button:after {
        content: "";
        position: absolute;
        top: 0;
        left: -90%;
        width: 55%;
        height: 100%;
        background: linear-gradient(
            105deg,
            transparent,
            rgba(255,255,255,0.28),
            transparent
        );
        transform: skewX(-22deg);
    }

    .stButton > button:hover:after {
        animation: buttonShine .65s ease forwards;
    }

    /* Animated tab indicator */
    [data-baseweb="tab-highlight"] {
        transform-origin: left center;
        animation: tabGrow .35s cubic-bezier(.22,.61,.36,1) both;
    }

    @keyframes heroIn {
        0% {
            opacity: 0;
            transform: translateY(18px) scale(.985);
            filter: blur(3px);
        }
        65% { filter: blur(0); }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    @keyframes riseIn {
        0% {
            opacity: 0;
            transform: translateY(18px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes riskReveal {
        0% {
            opacity: 0;
            transform: translateY(16px) scale(.99);
            clip-path: inset(8% 0 8% 0 round 19px);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
            clip-path: inset(0 0 0 0 round 19px);
        }
    }

    @keyframes slideFromLeft {
        0% {
            opacity: 0;
            transform: translateX(-14px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes slideFromRight {
        0% {
            opacity: 0;
            transform: translateX(14px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes chatIn {
        from {
            opacity: 0;
            transform: translateX(18px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes chatInAssistant {
        from {
            opacity: 0;
            transform: translateX(-18px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes orbDrift {
        0% { transform: translate(0, 0) scale(1); }
        100% { transform: translate(70px, -25px) scale(1.14); }
    }

    @keyframes softFloat {
        0% { transform: translateY(0); }
        100% { transform: translateY(-3px); }
    }

    @keyframes shine {
        0%, 55% { left: -35%; opacity: 0; }
        65% { opacity: 1; }
        85%, 100% { left: 120%; opacity: 0; }
    }

    @keyframes buttonShine {
        from { left: -90%; }
        to { left: 125%; }
    }

    @keyframes pulseDot {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 0 4px rgba(32,163,106,0.14);
        }
        50% {
            transform: scale(1.18);
            box-shadow: 0 0 0 7px rgba(32,163,106,0.06);
        }
    }

    @keyframes tabGrow {
        from { transform: scaleX(.35); opacity: .35; }
        to { transform: scaleX(1); opacity: 1; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
        }
    }

    @media (max-width: 900px) {
        .hero-title { font-size: 1.85rem; }
        .chat-user { margin-left: 4%; }
        .chat-assistant { margin-right: 4%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

FEATURE_LABELS = {
    "EXT_SOURCE_1": "External Credit Indicator 1",
    "EXT_SOURCE_2": "External Credit Indicator 2",
    "EXT_SOURCE_3": "External Credit Indicator 3",
    "AMT_CREDIT": "Loan Amount",
    "AMT_GOODS_PRICE": "Goods Value",
    "AMT_ANNUITY": "Loan Annuity",
    "AMT_INCOME_TOTAL": "Total Income",
    "AGE_YEARS": "Age",
    "DAYS_BIRTH": "Age",
    "EMPLOYMENT_YEARS": "Employment Duration",
    "DAYS_EMPLOYED": "Employment Duration",
    "BUREAU_DEBT_TO_CREDIT": "Existing Debt Burden",
    "BUREAU_CREDIT_COUNT": "Previous Credit Count",
    "BUREAU_ACTIVE_COUNT": "Active Previous Credits",
    "BUREAU_OVERDUE_COUNT": "Previous Overdue Credits",
    "BUREAU_CREDIT_SUM": "Previous Credit Amount",
    "BUREAU_DEBT_SUM": "Previous Debt Amount",
    "BUREAU_OVERDUE_SUM": "Previous Overdue Amount",
    "PREV_APPLICATION_COUNT": "Previous Applications",
    "PREV_APPROVED_COUNT": "Previous Approved Applications",
    "PREV_REFUSED_COUNT": "Previous Refused Applications",
    "PREV_REFUSAL_RATE": "Previous Application Refusal Rate",
    "AVG_PAYMENT_DELAY": "Average Payment Delay",
    "LATE_PAYMENT_COUNT": "Number of Late Payments",
    "PAYMENT_TO_INSTALLMENT_RATIO": "Payment-to-Installment Ratio",
    "CREDIT_INCOME_RATIO": "Credit-to-Income Ratio",
    "ANNUITY_INCOME_RATIO": "Annuity-to-Income Ratio",
    "CREDIT_GOODS_RATIO": "Credit-to-Goods Ratio",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Recent Social Circle Defaults",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Social Circle Recent Defaults",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Social Circle Credit Observations",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Social Circle Credit Observations",
    "DAYS_ID_PUBLISH": "ID Document History",
    "DAYS_REGISTRATION": "Registration History",
    "DAYS_LAST_PHONE_CHANGE": "Phone Change History",
}

FEATURE_EXPLANATIONS = {
    "EXT_SOURCE_1": "An external credit indicator. In this model, a lower value is associated with greater estimated default risk.",
    "EXT_SOURCE_2": "An external credit indicator. In this model, a lower value is associated with greater estimated default risk.",
    "EXT_SOURCE_3": "An external credit indicator. In this model, a lower value is associated with greater estimated default risk.",
    "AMT_CREDIT": "The requested loan amount. Its contribution reflects how this applicant compares with patterns learned from the training data.",
    "AMT_GOODS_PRICE": "The value of the goods financed by the loan. Its contribution is specific to this applicant and the trained model.",
    "AMT_ANNUITY": "The periodic loan payment amount. The model uses it together with other financial variables.",
    "AMT_INCOME_TOTAL": "Reported income used by the model as part of the applicant's financial profile.",
    "AGE_YEARS": "Applicant age. The contribution reflects the relationship learned by the model from historical applications.",
    "EMPLOYMENT_YEARS": "Approximate employment duration, derived from the original employment history field.",
    "BUREAU_DEBT_TO_CREDIT": "Existing bureau debt relative to previous credit. Higher values indicate a heavier existing debt burden.",
    "PREV_REFUSAL_RATE": "Share of previous applications that were refused. Higher values are associated with higher observed risk in the analyzed data.",
    "AVG_PAYMENT_DELAY": "Average delay on installment payments. Larger positive delays indicate weaker repayment timing.",
    "LATE_PAYMENT_COUNT": "Number of recorded late payments. Higher counts are associated with higher observed risk.",
    "PAYMENT_TO_INSTALLMENT_RATIO": "Ratio of recorded payments to expected installment amounts.",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Number of people in the applicant's social circle with recent defaults.",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Number of people in the applicant's social circle with recent defaults over a longer window.",
}


def clean_feature_name(feature):
    feature = str(feature)

    if feature.startswith("num__"):
        raw = feature[5:]
        return FEATURE_LABELS.get(raw, raw.replace("_", " ").title())

    if feature.startswith("cat__"):
        raw = feature[5:]

        for original, label in FEATURE_LABELS.items():
            prefix = original + "_"
            if raw.startswith(prefix):
                category = raw[len(prefix):]
                return f"{label}: {category.replace('_', ' ')}"

        return raw.replace("_", " ").title()

    return feature.replace("_", " ").title()


def raw_feature_name(feature):
    feature = str(feature)

    if feature.startswith("num__"):
        return feature[5:]

    return feature


def readable_value(feature, applicant):
    raw = raw_feature_name(feature)

    if raw == "DAYS_BIRTH":
        value = applicant.iloc[0].get("AGE_YEARS")
        return f"{float(value):.1f} years" if pd.notna(value) else "Not available"

    if raw == "DAYS_EMPLOYED":
        value = applicant.iloc[0].get("EMPLOYMENT_YEARS")
        return f"{float(value):.1f} years" if pd.notna(value) else "Not available"

    if raw not in applicant.columns:
        return "Not available"

    value = applicant.iloc[0][raw]

    if pd.isna(value):
        return "Not available"

    if raw in {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY"}:
        return f"₹{float(value):,.0f}"

    if raw in {"AGE_YEARS", "EMPLOYMENT_YEARS"}:
        return f"{float(value):.1f} years"

    if "RATE" in raw:
        rate = float(value)
        if 0 <= rate <= 1:
            return f"{rate * 100:.1f}%"
        return f"{rate:.1f}%"

    if "RATIO" in raw:
        return f"{float(value):.2f}"

    return f"{float(value):.2f}"


def feature_explanation(feature, applicant=None):
    """
    Explain the applicant's actual feature value in business language.
    The value itself is always read from the applicant dataframe.
    """
    raw = raw_feature_name(feature)

    if applicant is not None and raw in applicant.columns:
        value = applicant.iloc[0][raw]

        if pd.notna(value):
            value = float(value) if isinstance(value, (int, float)) else value

            if raw == "AVG_PAYMENT_DELAY":
                if float(value) < 0:
                    return (
                        f"The average payment delay is {float(value):.2f} days. "
                        f"The negative value means payments were recorded earlier than the "
                        f"reference due date on average; it is therefore a repayment-timing "
                        f"measure rather than a negative payment amount."
                    )
                elif float(value) == 0:
                    return (
                        "The average payment delay is 0 days, meaning the recorded payment timing "
                        "was, on average, aligned with the reference due date."
                    )
                return (
                    f"The average payment delay is {float(value):.2f} days. "
                    "A positive value means payments were recorded later than the reference "
                    "due date on average, providing a repayment-timing signal to the model."
                )

            if raw == "LATE_PAYMENT_COUNT":
                return (
                    f"The applicant has {float(value):.0f} recorded late payments. "
                    "This is a count of delayed repayment instances in the available installment "
                    "history, so a larger count indicates more repeated late-payment behaviour."
                )

            if raw == "PREV_REFUSAL_RATE":
                rate = float(value) * 100 if float(value) <= 1 else float(value)
                return (
                    f"The previous application refusal rate is {rate:.1f}%. "
                    "This represents the share of the applicant's previous applications that "
                    "were refused and provides context about prior application outcomes."
                )

            if raw == "BUREAU_DEBT_TO_CREDIT":
                return (
                    f"The existing debt burden ratio is {float(value):.2f}. "
                    "It compares historical bureau debt with historical credit exposure and "
                    "helps indicate how much existing borrowing is represented in the applicant's "
                    "credit history."
                )

            if raw in {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}:
                return (
                    f"This external credit indicator is {float(value):.2f}. "
                    "Lower values on this model feature are associated with higher estimated "
                    "default risk in the trained model."
                )

            if raw == "AGE_YEARS":
                return (
                    f"The applicant is {float(value):.1f} years old. "
                    "Age contributes as one part of the overall profile, and its SHAP contribution "
                    "shows how this particular value influenced the model estimate."
                )

            if raw == "EMPLOYMENT_YEARS":
                return (
                    f"The applicant has {float(value):.1f} years of employment history. "
                    "This provides context on employment duration and contributes to the model "
                    "alongside the applicant's other financial and demographic characteristics."
                )

    return FEATURE_EXPLANATIONS.get(
        raw,
        "This feature contributes to the applicant's predicted risk based on patterns learned by the model."
    )


def profile_value(applicant, column, fallback="Not available"):
    if column not in applicant.columns:
        return fallback

    value = applicant.iloc[0][column]

    if pd.isna(value):
        return fallback

    if column in {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"}:
        return f"₹{float(value):,.0f}"

    if column in {"AGE_YEARS", "EMPLOYMENT_YEARS"}:
        return f"{float(value):.1f} years"

    if column == "PREV_REFUSAL_RATE":
        return f"{float(value) * 100:.1f}%"

    return str(value)


def build_profile_summary(applicant, probability, band):
    age = profile_value(applicant, "AGE_YEARS", "Unknown age")
    employment = profile_value(applicant, "EMPLOYMENT_YEARS", "unknown tenure")
    income_type = profile_value(applicant, "NAME_INCOME_TYPE", "unspecified income type")
    education = profile_value(applicant, "NAME_EDUCATION_TYPE", "unspecified education")
    credit = profile_value(applicant, "AMT_CREDIT", "unknown loan amount")
    late = profile_value(applicant, "LATE_PAYMENT_COUNT", "0")

    sentence = (
        f"This applicant is {age}, has {employment} of employment history, "
        f"and is classified in the {income_type.lower()} income category. "
        f"The requested loan is {credit}, and the profile shows {late} recorded late payments."
    )

    if band == "High":
        conclusion = (
            f"The model estimates a {probability:.1%} probability of default, placing the applicant "
            f"in the High risk band. The result should be reviewed alongside the explanatory factors below."
        )
    elif band == "Medium":
        conclusion = (
            f"The model estimates a {probability:.1%} probability of default, placing the applicant "
            f"in the Medium risk band. The explanatory factors show which profile characteristics "
            f"most influenced that estimate."
        )
    else:
        conclusion = (
            f"The model estimates a {probability:.1%} probability of default, placing the applicant "
            f"in the Low risk band. The explanatory factors show which profile characteristics "
            f"supported that estimate."
        )

    return sentence, conclusion


def style_figure(fig, height=390):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=18, r=22, t=25, b=24),
        font=dict(family="Arial", color="#33414F"),
        hoverlabel=dict(bgcolor="#102A43", font_color="white"),
        transition=dict(duration=550, easing="cubic-in-out"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#DCE4EA")
    fig.update_yaxes(showgrid=True, gridcolor="#EDF1F3", zeroline=False)
    return fig



def format_number(value, decimals=2):
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def chart_explanation(title, data, metric_col, label_col=None):
    """
    Generate a data-driven explanation for a chart.
    Values come directly from the dataframe; the function contains
    interpretation logic, not hardcoded dataset values.
    """
    if data is None or data.empty or metric_col not in data.columns:
        return "The chart does not contain enough information to produce an interpretation."

    metric = pd.to_numeric(data[metric_col], errors="coerce").dropna()

    if metric.empty:
        return "The chart does not contain enough numeric information to produce an interpretation."

    highest_idx = data[metric_col].astype(float).idxmax()
    lowest_idx = data[metric_col].astype(float).idxmin()

    high_value = float(data.loc[highest_idx, metric_col])
    low_value = float(data.loc[lowest_idx, metric_col])

    high_label = str(data.loc[highest_idx, label_col]) if label_col else "the highest group"
    low_label = str(data.loc[lowest_idx, label_col]) if label_col else "the lowest group"

    spread = abs(high_value - low_value)

    if "rate" in metric_col.lower() or "percentage" in metric_col.lower():
        value_sentence = (
            f"{high_label} has the highest observed rate at {high_value:.2f}%, "
            f"while {low_label} has the lowest at {low_value:.2f}%. "
            f"The gap between the two groups is {spread:.2f} percentage points."
        )
    else:
        value_sentence = (
            f"{high_label} has the highest observed value at {format_number(high_value)}, "
            f"while {low_label} has the lowest at {format_number(low_value)}."
        )

    return value_sentence


def default_distribution_explanation(default_data):
    total = int(default_data["Applications"].sum())
    default_count = int(
        default_data.loc[default_data["Status"] == "Default", "Applications"].iloc[0]
    )
    rate = (default_count / total * 100) if total else 0

    return (
        f"The portfolio contains {total:,} applications, of which {default_count:,} are recorded "
        f"as defaults ({rate:.2f}%). This shows that default is the minority outcome in the dataset, "
        f"which is why class imbalance needs to be considered when evaluating the model."
    )


def repayment_explanation(repayment_data):
    if repayment_data is None or repayment_data.empty:
        return "There are no repayment-delay values available for comparison."

    values = dict(
        zip(
            repayment_data["Status"].astype(str),
            repayment_data["Average_Delay"].astype(float),
        )
    )

    if "Default" in values and "No Default" in values:
        default_delay = values["Default"]
        no_default_delay = values["No Default"]

        direction = "higher" if default_delay > no_default_delay else "lower"
        difference = abs(default_delay - no_default_delay)

        return (
            f"Applications recorded as Default have an average payment delay of "
            f"{default_delay:.2f}, compared with {no_default_delay:.2f} for No Default applications. "
            f"The default group is therefore {direction} by {difference:.2f} units on average. "
            f"This suggests repayment timing contains useful behavioural information, although the "
            f"chart describes an association in the historical data rather than proving a cause."
        )

    return (
        "The chart compares average payment timing across the available target groups. "
        "Differences in repayment timing can provide behavioural context when assessing credit risk."
    )


def missingness_explanation(missing):
    if missing is None or missing.empty:
        return "No missing-value information is available."

    top = missing.iloc[-1]
    second = missing.iloc[-2] if len(missing) > 1 else None

    text = (
        f"{top['Feature']} has the highest missing share among the displayed features at "
        f"{float(top['Missing_Percentage']):.1f}%. "
    )

    if second is not None:
        text += (
            f"{second['Feature']} follows at {float(second['Missing_Percentage']):.1f}%. "
        )

    text += (
        "High missingness reduces the amount of direct information available for some applicants, "
        "so the preprocessing pipeline needs to handle it consistently through feature selection, "
        "imputation, or both."
    )
    return text


def shap_explanation_text(explanation):
    """
    Explain the contribution map using the actual SHAP dataframe.
    """
    if explanation is None or explanation.empty:
        return "No SHAP contributions are available for this applicant."

    work = explanation.copy()
    work["shap_value"] = pd.to_numeric(work["shap_value"], errors="coerce")
    work = work.dropna(subset=["shap_value"])

    if work.empty:
        return "No valid SHAP contributions are available for this applicant."

    highest = work.loc[work["shap_value"].idxmax()]
    lowest = work.loc[work["shap_value"].idxmin()]

    high_name = clean_feature_name(highest["feature"])
    low_name = clean_feature_name(lowest["feature"])
    high_value = float(highest["shap_value"])
    low_value = float(lowest["shap_value"])

    return (
        f"{high_name} is the strongest factor pushing this prediction upward, with a SHAP contribution "
        f"of {high_value:+.3f}. {low_name} is the strongest factor pulling the prediction downward, "
        f"with a contribution of {low_value:+.3f}. A SHAP value describes the direction and strength "
        f"of a feature's contribution to this applicant's prediction; it is not the applicant's raw "
        f"feature value and its sign does not describe whether the underlying feature itself is good or bad."
    )


def call_generate_sql(question, conversation):
    """
    Keep compatibility with both the contextual and earlier NL-to-SQL signatures.
    The contextual implementation is preferred.
    """
    try:
        return generate_sql(question, conversation=conversation)
    except TypeError:
        return generate_sql(question)


def call_business_answer(question, sql, result, conversation):
    """
    Keep compatibility with contextual and earlier answer-generator signatures.
    """
    try:
        return generate_business_answer(
            question,
            sql,
            result,
            conversation=conversation,
        )
    except TypeError:
        return generate_business_answer(
            question,
            sql,
            result,
        )

def query_chart(result):
    """
    Build a useful visual for grouped Talk-to-Data results.
    Returns None for scalar results.
    """

    if result is None or result.empty or len(result.columns) < 2:
        return None

    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    text_cols = [c for c in result.columns if c not in numeric_cols]

    if not numeric_cols or not text_cols:
        return None

    if len(result) < 2 or len(result) > 15:
        return None

    y_col = numeric_cols[-1]
    x_col = text_cols[0]

    fig = px.bar(
        result,
        x=x_col,
        y=y_col,
        text=y_col,
        color=x_col,
        color_discrete_sequence=[
            "#20A36A",
            "#2B78C5",
            "#D69E2E",
            "#8B5CF6",
            "#E56B6F",
            "#3FA7D6",
        ],
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y}}<extra></extra>",
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title=y_col.replace("_", " ").title(),
    )

    return style_figure(fig, 360)


def render_scalar_kpis(result):
    """
    KPI cards for single-row Talk-to-Data results, in the app's own
    visual language (same markup as the EDA/prediction tabs), instead of
    leaving scalar answers with no visual at all.
    """
    if result is None or result.empty or len(result) != 1:
        return False

    accents = ["blue", "green", "amber", "red"]
    cols = st.columns(min(len(result.columns), 4))

    for col_widget, col_name, accent in zip(cols, result.columns[:4], accents):
        value = result.iloc[0][col_name]
        label = col_name.replace("_", " ").title()
        is_pct = any(h in col_name.lower() for h in ("rate", "pct", "percent"))

        try:
            display_value = f"{float(value):,.2f}{'%' if is_pct else ''}"
        except (TypeError, ValueError):
            display_value = str(value)

        col_widget.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top">
                    <div class="kpi-label">{escape(label)}</div>
                    <div class="kpi-accent-{accent}"></div>
                </div>
                <div class="kpi-value">{escape(display_value)}</div>
                <div class="kpi-note">From your question</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return True


def style_evidence_table(result):
    """Format numeric columns (esp. rate/pct columns) before showing raw evidence."""
    numeric_cols = result.select_dtypes(include="number").columns
    fmt = {}
    for col in numeric_cols:
        is_pct = any(h in col.lower() for h in ("rate", "pct", "percent"))
        fmt[col] = "{:,.2f}%" if is_pct else "{:,.2f}"
    return result.style.format(fmt)


def detect_applicant_id(question):
    """
    Best-effort detection of an applicant-level explainability question,
    e.g. "why did applicant 100234 get flagged as high risk?" There is no
    SQL query that can explain a single prediction, so these questions are
    routed straight to the model + SHAP instead of the talk-to-data path.
    """
    q = question.lower()
    explain_words = ("why", "explain", "reason", "driving", "drove", "factors")
    if not any(w in q for w in explain_words):
        return None

    match = re.search(r"\b(\d{5,})\b", question)
    return int(match.group(1)) if match else None


def render_applicant_explanation(applicant_id, dataframe, decision_threshold):
    """Answer an applicant-specific 'why' question from the trained model + SHAP."""
    applicant = dataframe[dataframe["SK_ID_CURR"] == applicant_id]

    if applicant.empty:
        message = f"I couldn't find an applicant with ID {applicant_id} in the dataset."
        st.error(message)
        return message

    prediction = predict_risk(applicant)
    band = prediction["risk_band"]
    probability = prediction["probability"]

    explanation = explain_prediction(applicant)
    narrative = shap_explanation_text(explanation)

    answer = (
        f"Applicant {applicant_id} falls in the {band} risk band, with an "
        f"estimated default probability of {probability:.1%} against a "
        f"decision threshold of {decision_threshold:.1%}. {narrative}"
    )

    st.markdown('<div class="chat-answer-title">Analysis</div>', unsafe_allow_html=True)
    st.markdown(answer)

    accent = "red" if band == "High" else "amber" if band == "Medium" else "green"
    k1, k2 = st.columns(2)

    k1.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-label">Risk Band</div>
                <div class="kpi-accent-{accent}"></div>
            </div>
            <div class="kpi-value">{escape(band)}</div>
            <div class="kpi-note">Model-estimated risk band</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    k2.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-label">Default Probability</div>
                <div class="kpi-accent-blue"></div>
            </div>
            <div class="kpi-value">{probability:.1%}</div>
            <div class="kpi-note">Raw model probability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return answer


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_rules():
    return load_business_rules()


@st.cache_resource
def get_connection(dataframe):
    connection = duckdb.connect()
    connection.register("applications", dataframe)
    return connection


df = get_data()
rules = get_rules()
con = get_connection(df)

threshold_path = BASE_DIR / "models" / "threshold.pkl"
threshold = float(joblib.load(threshold_path))


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_draft" not in st.session_state:
    st.session_state.chat_draft = None


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker"><span style="display:inline-flex;align-items:center;gap:7px;"><span class="status-dot" style="display:inline-block;"></span>AI-powered decision support</span></div>
        <div class="hero-title">Credit Risk Intelligence Platform</div>
        <div class="hero-subtitle">
            Home Credit Default Risk · Explainable ML · Talk-to-Data
        </div>
        <div class="hero-chip-row">
            <span class="hero-chip">LightGBM Risk Engine</span>
            <span class="hero-chip">SHAP Explanations</span>
            <span class="hero-chip">Natural Language Analytics</span>
            <span class="hero-chip">Business Rules</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

overview_tab, risk_tab, rules_tab, data_tab = st.tabs(
    [
        "Overview",
        "Risk Assessment",
        "Decision Rules",
        "Ask the Data",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Portfolio Pulse</div>
                <div class="section-caption">
                    A decision-oriented view of the Home Credit portfolio.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_apps = len(df)
    default_count = int(df["TARGET"].sum())
    default_rate = df["TARGET"].mean() * 100

    k1, k2, k3, k4 = st.columns(4)

    k1.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-label">Applications</div>
                <div class="kpi-accent-blue"></div>
            </div>
            <div class="kpi-value">{total_apps:,}</div>
            <div class="kpi-note">Applicant records analysed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k2.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-label">Observed Default Rate</div>
                <div class="kpi-accent-red"></div>
            </div>
            <div class="kpi-value">{default_rate:.2f}%</div>
            <div class="kpi-note">Historical target rate in the dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k3.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-label">Defaulted Applications</div>
                <div class="kpi-accent-amber"></div>
            </div>
            <div class="kpi-value">{default_count:,}</div>
            <div class="kpi-note">Observed TARGET = 1</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k4.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-label">Model ROC-AUC</div>
                <div class="kpi-accent-green"></div>
            </div>
            <div class="kpi-value">0.7738</div>
            <div class="kpi-note">Validation discrimination score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="status-strip">
            <span class="status-dot"></span>
            <span class="status-text">Data loaded</span>
            <span class="status-dot"></span>
            <span class="status-text">Risk model ready</span>
            <span class="status-dot"></span>
            <span class="status-text">SHAP ready</span>
            <span class="status-dot"></span>
            <span class="status-text">Talk-to-Data ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # EDA charts
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="section-head">
                <div>
                    <div class="section-title">Default Distribution</div>
                    <div class="section-caption">Target balance across applications.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        default_data = pd.DataFrame(
            {
                "Status": ["Default", "No Default"],
                "Applications": [
                    int((df["TARGET"] == 1).sum()),
                    int((df["TARGET"] == 0).sum()),
                ],
            }
        )

        fig = px.pie(
            default_data,
            names="Status",
            values="Applications",
            hole=0.63,
            color="Status",
            color_discrete_map={
                "Default": "#C94A4A",
                "No Default": "#20A36A",
            },
        )

        fig.update_traces(
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>%{value:,} applications<br>%{percent}<extra></extra>",
        )

        fig.add_annotation(
            text=f"{default_rate:.1f}%<br><span style='font-size:12px'>default rate</span>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=22, color="#102A43"),
        )

        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.10),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=45),
            height=395,
            transition=dict(duration=700, easing="cubic-in-out"),
        )

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown(
            f"""
            <div class="interpretation-card">
                <div class="interpretation-label">What the chart means</div>
                <div class="insight-text">{escape(default_distribution_explanation(default_data))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-head">
                <div>
                    <div class="section-title">Default Rate by Gender</div>
                    <div class="section-caption">Observed default rate for sufficiently large groups.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gender_data = con.execute(
            """
            SELECT
                CASE
                    WHEN CODE_GENDER = 'M' THEN 'Male'
                    WHEN CODE_GENDER = 'F' THEN 'Female'
                    ELSE 'Unknown'
                END AS Gender,
                COUNT(*) AS Applications,
                ROUND(AVG(TARGET) * 100, 2) AS Default_Rate
            FROM applications
            GROUP BY Gender
            HAVING COUNT(*) >= 1000
            ORDER BY Default_Rate DESC
            """
        ).fetchdf()

        fig = px.bar(
            gender_data,
            x="Gender",
            y="Default_Rate",
            text="Default_Rate",
            color="Gender",
            color_discrete_sequence=["#C94A4A", "#2B78C5", "#20A36A"],
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_layout(
            showlegend=False,
            yaxis_title="Default Rate (%)",
            xaxis_title="",
        )

        fig = style_figure(fig, 395)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown(
            f"""
            <div class="interpretation-card">
                <div class="interpretation-label">What the chart means</div>
                <div class="insight-text">
                    {escape(chart_explanation("Gender", gender_data, "Default_Rate", "Gender"))}
                    The difference is descriptive: it shows how observed default rates vary across
                    the groups in this historical dataset and does not by itself establish causation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Where Risk Concentrates</div>
                <div class="section-caption">Observed default rates by education level.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    education_data = con.execute(
        """
        SELECT
            NAME_EDUCATION_TYPE AS Education,
            COUNT(*) AS Applications,
            ROUND(AVG(TARGET) * 100, 2) AS Default_Rate
        FROM applications
        GROUP BY NAME_EDUCATION_TYPE
        HAVING COUNT(*) >= 1000
        ORDER BY Default_Rate DESC
        """
    ).fetchdf()

    fig = px.bar(
        education_data,
        x="Default_Rate",
        y="Education",
        orientation="h",
        text="Default_Rate",
        color="Default_Rate",
        color_continuous_scale=["#2B78C5", "#20A36A", "#D69E2E", "#C94A4A"],
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Default rate: %{x:.2f}%<extra></extra>",
    )

    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Default Rate (%)",
        yaxis_title="",
    )

    fig = style_figure(fig, 440)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown(
        f"""
        <div class="interpretation-card">
            <div class="interpretation-label">What the chart means</div>
            <div class="insight-text">
                {escape(chart_explanation("Education", education_data, "Default_Rate", "Education"))}
                These differences describe the historical portfolio; they should not be interpreted
                as education causing default.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Income + repayment
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="section-head">
                <div>
                    <div class="section-title">Default Rate by Income Type</div>
                    <div class="section-caption">Observed portfolio differences across income categories.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        income_data = con.execute(
            """
            SELECT
                NAME_INCOME_TYPE AS Income_Type,
                COUNT(*) AS Applications,
                ROUND(AVG(TARGET) * 100, 2) AS Default_Rate
            FROM applications
            GROUP BY NAME_INCOME_TYPE
            HAVING COUNT(*) >= 1000
            ORDER BY Default_Rate DESC
            """
        ).fetchdf()

        fig = px.bar(
            income_data,
            x="Income_Type",
            y="Default_Rate",
            text="Default_Rate",
            color="Default_Rate",
            color_continuous_scale=["#2B78C5", "#20A36A", "#D69E2E", "#C94A4A"],
        )

        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(
            coloraxis_showscale=False,
            yaxis_title="Default Rate (%)",
            xaxis_title="",
            xaxis_tickangle=-18,
        )

        fig = style_figure(fig, 410)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown(
            f"""
            <div class="interpretation-card">
                <div class="interpretation-label">What the chart means</div>
                <div class="insight-text">
                    {escape(chart_explanation("Income", income_data, "Default_Rate", "Income_Type"))}
                    The result highlights differences in observed portfolio risk across income groups;
                    other financial and applicant characteristics can also contribute to those differences.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-head">
                <div>
                    <div class="section-title">Repayment Behaviour</div>
                    <div class="section-caption">Average payment delay across observed target groups.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        repayment_data = con.execute(
            """
            SELECT
                DEFAULT_STATUS AS Status,
                ROUND(AVG(AVG_PAYMENT_DELAY), 2) AS Average_Delay
            FROM applications
            WHERE AVG_PAYMENT_DELAY IS NOT NULL
            GROUP BY DEFAULT_STATUS
            """
        ).fetchdf()

        fig = px.bar(
            repayment_data,
            x="Status",
            y="Average_Delay",
            text="Average_Delay",
            color="Status",
            color_discrete_map={
                "Default": "#C94A4A",
                "No Default": "#20A36A",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            cliponaxis=False,
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title="Average Payment Delay",
            xaxis_title="",
        )

        fig = style_figure(fig, 410)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown(
            f"""
            <div class="interpretation-card">
                <div class="interpretation-label">What the chart means</div>
                <div class="insight-text">{escape(repayment_explanation(repayment_data))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Portfolio Takeaways</div>
                <div class="section-caption">Business observations that are directly supported by the analysis.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    male_rate = float(
        gender_data.loc[gender_data["Gender"] == "Male", "Default_Rate"].iloc[0]
    ) if "Male" in gender_data["Gender"].values else None

    female_rate = float(
        gender_data.loc[gender_data["Gender"] == "Female", "Default_Rate"].iloc[0]
    ) if "Female" in gender_data["Gender"].values else None

    top_education = education_data.iloc[0]
    lowest_education = education_data.iloc[-1]

    if male_rate is not None and female_rate is not None:
        gender_text = f"Male applicants show {male_rate:.2f}% observed default rate versus {female_rate:.2f}% for female applicants."
    else:
        gender_text = "Gender-level results are available in the chart above."

    insight_cols = st.columns(3)

    insight_cards = [
        (
            "01",
            "Default concentration",
            f"The portfolio's observed default rate is {default_rate:.2f}%, so class imbalance is an important modelling consideration.",
        ),
        (
            "02",
            "Gender difference",
            gender_text,
        ),
        (
            "03",
            "Education difference",
            f"{top_education['Education']} has the highest observed default rate at {top_education['Default_Rate']:.2f}%, while {lowest_education['Education']} has the lowest at {lowest_education['Default_Rate']:.2f}%.",
        ),
    ]

    for col, (number, title, text) in zip(insight_cols, insight_cards):
        with col:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-number">Insight {number}</div>
                    <div class="insight-title">{escape(title)}</div>
                    <div class="insight-text">{escape(text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Data Quality Lens</div>
                <div class="section-caption">Features with the largest observed missing-value shares.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    missing.columns = ["Feature", "Missing_Percentage"]

    fig = px.bar(
        missing.sort_values("Missing_Percentage"),
        x="Missing_Percentage",
        y="Feature",
        orientation="h",
        text="Missing_Percentage",
        color="Missing_Percentage",
        color_continuous_scale=["#BFEBD8", "#20A36A", "#D69E2E"],
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Missing Values (%)",
        yaxis_title="",
    )

    fig = style_figure(fig, 430)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown(
        f"""
        <div class="interpretation-card">
            <div class="interpretation-label">What the chart means</div>
            <div class="insight-text">{escape(missingness_explanation(missing))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

with risk_tab:

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Applicant Risk Assessment</div>
                <div class="section-caption">
                    One applicant, one interpretable risk view.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    applicant_ids = df["SK_ID_CURR"].dropna().astype(int).tolist()

    applicant_id = st.selectbox(
        "Select applicant",
        applicant_ids,
        index=0,
    )

    if st.button("Assess Applicant", type="primary", use_container_width=False):
        applicant = df[df["SK_ID_CURR"] == applicant_id]

        if applicant.empty:
            st.error("Applicant ID not found.")
        else:

            prediction = predict_risk(applicant)

            score = float(prediction["risk_score"])
            probability = float(prediction["probability"])
            band = str(prediction["risk_band"])

            # Risk banner
            if band == "High":
                band_text = "Higher estimated default risk"
            elif band == "Medium":
                band_text = "Moderate estimated default risk"
            else:
                band_text = "Lower estimated default risk"

            st.markdown(
                f"""
                <div class="risk-banner">
                    <div class="risk-kicker">Applicant {applicant_id}</div>
                    <div class="risk-title">{band} risk · {score:.2f}% estimated risk score</div>
                    <div class="risk-copy">
                        {band_text}. The score is based on historical patterns learned by the model
                        and should be used as decision support alongside human review.
                    </div>
                    <span class="risk-pill">Decision threshold: {threshold:.4f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Gauge
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={
                        "suffix": "%",
                        "font": {"size": 34, "color": "#102A43"},
                    },
                    title={
                        "text": "Estimated Default Risk",
                        "font": {"size": 14, "color": "#667788"},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 0,
                            "tickcolor": "rgba(0,0,0,0)",
                        },
                        "bar": {
                            "color": "#C94A4A" if band == "High" else "#D69E2E" if band == "Medium" else "#20A36A",
                            "thickness": 0.22,
                        },
                        "bgcolor": "#EEF3F5",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 30], "color": "#E8F7F0"},
                            {"range": [30, 66.22], "color": "#FFF7E1"},
                            {"range": [66.22, 100], "color": "#FDECEC"},
                        ],
                    },
                )
            )

            gauge.update_layout(
                height=245,
                margin=dict(l=25, r=25, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Arial"},
            )

            g1, g2, g3 = st.columns([1.25, 1.0, 1.0])

            with g1:
                st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})

            with g2:
                st.markdown(
                    f"""
                    <div class="kpi-card" style="margin-top:20px;">
                        <div class="kpi-label">Risk Category</div>
                        <div class="kpi-value">{escape(band)}</div>
                        <div class="kpi-note">Model-estimated risk band</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with g3:
                st.markdown(
                    f"""
                    <div class="kpi-card" style="margin-top:20px;">
                        <div class="kpi-label">Default Probability</div>
                        <div class="kpi-value">{probability:.1%}</div>
                        <div class="kpi-note">Raw model probability</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.info(
                "Decision-support only: this score reflects patterns learned from historical data "
                "and should not be used as an automatic loan approval or rejection decision."
            )

            # ----------------------------------------------------
            # Applicant profile
            # ----------------------------------------------------

            st.markdown(
                """
                <div class="section-head">
                    <div>
                        <div class="section-title">Applicant Profile</div>
                        <div class="section-caption">A human-readable snapshot instead of a raw feature table.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            profile_groups = [
                (
                    "Applicant",
                    profile_value(applicant, "NAME_CONTRACT_TYPE"),
                    f"{profile_value(applicant, 'NAME_FAMILY_STATUS')} · {profile_value(applicant, 'NAME_HOUSING_TYPE')}",
                ),
                (
                    "Financial profile",
                    profile_value(applicant, "AMT_CREDIT"),
                    f"Income {profile_value(applicant, 'AMT_INCOME_TOTAL')} · Annuity {profile_value(applicant, 'AMT_ANNUITY')}",
                ),
                (
                    "Background",
                    profile_value(applicant, "AGE_YEARS"),
                    f"{profile_value(applicant, 'NAME_EDUCATION_TYPE')} · {profile_value(applicant, 'NAME_INCOME_TYPE')}",
                ),
                (
                    "Repayment history",
                    profile_value(applicant, "LATE_PAYMENT_COUNT"),
                    f"Previous applications: {profile_value(applicant, 'PREV_APPLICATION_COUNT')} · Refusal rate: {profile_value(applicant, 'PREV_REFUSAL_RATE')}",
                ),
            ]

            p1, p2, p3, p4 = st.columns(4)

            for col, (label, value, detail) in zip([p1, p2, p3, p4], profile_groups):
                with col:
                    st.markdown(
                        f"""
                        <div class="profile-card">
                            <div class="profile-label">{escape(label)}</div>
                            <div class="profile-value">{escape(value)}</div>
                            <div class="profile-detail">{escape(detail)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            summary_1, summary_2 = build_profile_summary(applicant, probability, band)

            st.markdown(
                f"""
                <div class="panel" style="margin-top:12px;">
                    <div class="profile-label">Applicant snapshot</div>
                    <div style="color:#294052; font-size:0.92rem; line-height:1.6; margin-top:7px;">
                        {escape(summary_1)}
                    </div>
                    <div style="color:#5E6D7B; font-size:0.85rem; line-height:1.55; margin-top:8px;">
                        {escape(summary_2)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ----------------------------------------------------
            # Explainability
            # ----------------------------------------------------

            st.markdown(
                """
                <div class="section-head">
                    <div>
                        <div class="section-title">Why this score?</div>
                        <div class="section-caption">
                            SHAP contributions translated into business language.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            explanation = explain_prediction(applicant).copy()

            hidden_features = {
                "num__DAYS_BIRTH",
                "num__DAYS_EMPLOYED",
            }

            explanation = explanation[
                ~explanation["feature"].isin(hidden_features)
            ].copy()

            positive = (
                explanation[explanation["shap_value"] > 0]
                .sort_values("shap_value", ascending=False)
                .head(5)
            )

            negative = (
                explanation[explanation["shap_value"] < 0]
                .sort_values("shap_value", ascending=True)
                .head(5)
            )

            left, right = st.columns(2)

            with left:
                st.markdown("### Factors increasing risk")

                if positive.empty:
                    st.info("No strong increasing-risk factors were identified.")
                else:
                    for _, row in positive.iterrows():

                        feature = row["feature"]
                        label = clean_feature_name(feature)
                        value = readable_value(feature, applicant)
                        shap_value = abs(float(row["shap_value"]))

                        st.markdown(
                            f"""
                            <div class="reason-card positive">
                                <div class="reason-title">{escape(label)}</div>
                                <div class="reason-subtitle">SHAP contribution: +{shap_value:.3f}</div>
                                <div class="reason-value">Applicant value: {escape(value)}</div>
                                <div class="reason-text">
                                    {escape(feature_explanation(feature, applicant))}
                                    <br><br>
                                    This applicant's value is contributing to a higher model estimate
                                    relative to the model baseline. The SHAP contribution is a model
                                    explanation, not a standalone measure of credit quality.
                                </div>
                                <span class="impact-pill impact-up">Pushes the prediction upward</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with right:
                st.markdown("### Factors reducing risk")

                if negative.empty:
                    st.info("No strong reducing-risk factors were identified.")
                else:
                    for _, row in negative.iterrows():

                        feature = row["feature"]
                        label = clean_feature_name(feature)
                        value = readable_value(feature, applicant)
                        shap_value = abs(float(row["shap_value"]))

                        st.markdown(
                            f"""
                            <div class="reason-card negative">
                                <div class="reason-title">{escape(label)}</div>
                                <div class="reason-subtitle">SHAP contribution: −{shap_value:.3f}</div>
                                <div class="reason-value">Applicant value: {escape(value)}</div>
                                <div class="reason-text">
                                    {escape(feature_explanation(feature, applicant))}
                                    <br><br>
                                    This applicant's value is contributing to a lower model estimate
                                    relative to the model baseline. The negative SHAP sign refers to
                                    model contribution, not to whether the underlying feature value is negative.
                                </div>
                                <span class="impact-pill impact-down">Pulls the prediction downward</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # SHAP visual
            chart_df = explanation.copy()
            chart_df["Feature"] = chart_df["feature"].apply(clean_feature_name)
            chart_df["Contribution"] = chart_df["shap_value"]

            chart_df = chart_df.sort_values(
                "Contribution",
                key=lambda s: s.abs(),
                ascending=True,
            ).tail(10)

            fig = px.bar(
                chart_df,
                x="Contribution",
                y="Feature",
                orientation="h",
                color="Contribution",
                color_continuous_scale=["#20A36A", "#EEF3F5", "#C94A4A"],
            )

            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="SHAP Contribution",
                yaxis_title="",
            )

            fig = style_figure(fig, 430)

            st.markdown("### Contribution map")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            st.markdown(
                f"""
                <div class="interpretation-card">
                    <div class="interpretation-label">How to interpret the map</div>
                    <div class="insight-text">{escape(shap_explanation_text(explanation))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("View technical SHAP values"):
                technical = explanation[["feature", "shap_value", "impact"]].copy()
                technical["Feature"] = technical["feature"].apply(clean_feature_name)
                technical["Applicant Value"] = technical["feature"].apply(
                    lambda f: readable_value(f, applicant)
                )
                technical = technical[
                    ["Feature", "Applicant Value", "shap_value", "impact"]
                ]
                technical.columns = [
                    "Feature",
                    "Applicant Value",
                    "SHAP Value",
                    "Impact",
                ]

                st.dataframe(
                    technical,
                    width="stretch",
                    hide_index=True,
                )


# ============================================================
# CREDIT RULES
# ============================================================

with rules_tab:

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Decision Rules</div>
                <div class="section-caption">
                    Business-readable patterns derived from the model and observed data.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "These are observed risk signals derived from the analysed data and model behaviour. "
        "They support review and interpretation; they are not automatic lending policies."
    )

    st.markdown(
        f"""
        <div class="interpretation-card">
            <div class="interpretation-label">How these rules are derived</div>
            <div class="insight-text">
                The rules summarise relationships observed during the analysis and are linked to
                variables that were useful for understanding model behaviour. A rule indicates
                a risk pattern in the historical dataset; it does not mean the condition alone
                determines whether an applicant will default.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for index, rule in enumerate(rules, start=1):
        st.markdown(
            f"""
            <div class="reason-card">
                <div class="reason-title">
                    {index:02d} · {escape(str(rule["Rule"]))}
                </div>
                <div class="reason-text">
                    <b>Condition:</b> {escape(str(rule["Condition"]))}<br>
                    <b>Interpretation:</b> {escape(str(rule["Interpretation"]))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="panel" style="margin-top:18px;">
            <div class="profile-label">How to read these rules</div>
            <div style="color:#294052; line-height:1.6; margin-top:7px; font-size:0.90rem;">
                The rules summarise observed relationships that were useful for interpreting
                model behaviour. They are intentionally written as risk signals rather than
                deterministic approval or rejection rules.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ASK THE DATA
# ============================================================

with data_tab:

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">Ask the Credit Data</div>
                <div class="section-caption">
                    Ask anything about the credit-risk dataset. Follow-up questions use the recent conversation.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggested_questions = [
        "What is the overall default rate?",
        "Are male applicants more likely to default than female applicants?",
        "Which education group has the highest default rate?",
        "How many applicants defaulted?",
        "What is the average credit amount?",
        "Which income group has the highest default rate?",
        "What is the average previous application refusal rate?",
        "What is the average age of applicants who defaulted?",
    ]

    # Suggested prompts are optional shortcuts, not the supported question set.
    suggestion_cols = st.columns(4)

    for idx, prompt_text in enumerate(suggested_questions[:4]):
        with suggestion_cols[idx]:
            if st.button(
                prompt_text,
                key=f"suggestion_{idx}",
                use_container_width=True,
            ):
                st.session_state.chat_draft = prompt_text

    question = st.chat_input(
        "Ask a question about the credit data...",
        key="credit_chat_input",
    )

    if question is None:
        question = st.session_state.pop("chat_draft", None)

    if st.session_state.chat_history:
        st.caption(
            "Conversation context is available for natural follow-up questions such as "
            '"What about defaulters?" or "Is that higher for males?"'
        )

    if question and question.strip():

        question = question.strip()

        recent_history = st.session_state.chat_history[-6:]

        try:

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                applicant_id = detect_applicant_id(question)

                if applicant_id is not None:
                    with st.spinner("Looking up applicant..."):
                        answer = render_applicant_explanation(
                            applicant_id, df, threshold
                        )

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "sql": f"-- Explainability lookup for applicant {applicant_id}",
                            "result": [],
                            "answer": answer,
                        }
                    )

                else:
                    with st.spinner("Analysing the question..."):

                        sql = call_generate_sql(
                            question,
                            recent_history,
                        )

                        if not sql:
                            st.error(
                                "I couldn't translate that question into a safe data query. "
                                "Please rephrase it."
                            )
                        else:

                            valid, message = validate_sql(sql)

                            if not valid:
                                st.error(message)

                            else:

                                result = run_query(con, sql)

                                answer = call_business_answer(
                                    question,
                                    sql,
                                    result,
                                    recent_history,
                                )

                                if not answer or not answer.strip():
                                    answer = (
                                        "I found data for your question, but I could not generate "
                                        "a reliable explanation. Please try the question again."
                                    )

                                st.markdown(
                                    '<div class="chat-answer-title">Analysis</div>',
                                    unsafe_allow_html=True,
                                )

                                st.markdown(
                                    answer
                                )

                                # ----------------------------------------
                                # Dynamic visual explanation
                                # ----------------------------------------

                                if len(result) == 1:
                                    render_scalar_kpis(result)
                                    result_fig = None
                                else:
                                    result_fig = query_chart(result)

                                if result_fig is not None:

                                    st.markdown(
                                        '<div class="section-title" style="margin-top:14px;">What the data looks like</div>',
                                        unsafe_allow_html=True,
                                    )

                                    st.plotly_chart(
                                        result_fig,
                                        width="stretch",
                                        config={"displayModeBar": False},
                                    )

                                    # Explain the actual chart result.
                                    numeric_cols = result.select_dtypes(
                                        include="number"
                                    ).columns.tolist()

                                    text_cols = [
                                        c for c in result.columns
                                        if c not in numeric_cols
                                    ]

                                    if numeric_cols and text_cols:

                                        visual_explanation = chart_explanation(
                                            "Talk to Data",
                                            result,
                                            numeric_cols[-1],
                                            text_cols[0],
                                        )

                                        st.markdown(
                                            f"""
                                            <div class="interpretation-card">
                                                <div class="interpretation-label">What this visual means</div>
                                                <div class="insight-text">
                                                    {escape(visual_explanation)}
                                                    The chart is descriptive: it shows what is present
                                                    in the returned records and does not by itself establish causation.
                                                </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                # ----------------------------------------
                                # Evidence
                                # ----------------------------------------

                                with st.expander("See the evidence behind this answer"):

                                    st.dataframe(
                                        style_evidence_table(result),
                                        width="stretch",
                                        hide_index=True,
                                    )

                                with st.expander("View generated SQL"):
                                    st.code(
                                        sql,
                                        language="sql",
                                    )

                                st.session_state.chat_history.append(
                                    {
                                        "question": question,
                                        "sql": sql,
                                        "result": result.to_dict(orient="records"),
                                        "answer": answer,
                                    }
                                )

        except Exception as exc:
            st.error(
                f"Unable to process the question: {exc}"
            )

    # --------------------------------------------------------
    # Render previous turns as a real chatbot transcript
    # --------------------------------------------------------

    if st.session_state.chat_history:

        st.markdown(
            '<div class="section-title">Conversation</div>',
            unsafe_allow_html=True,
        )

        # Chronological order, like a real chat window.
        for item in st.session_state.chat_history:

            with st.chat_message("user"):
                st.markdown(
                    str(item.get("question", ""))
                )

            with st.chat_message("assistant"):
                st.markdown(
                    str(item.get("answer", ""))
                )

        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()


# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Credit Risk Intelligence · Explainable ML · Talk-to-Data · Decision support, not automated lending.
    </div>
    """,
    unsafe_allow_html=True,
)