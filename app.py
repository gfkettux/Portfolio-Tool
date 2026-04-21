"""
app.py — Portfolio Insight Tool
UI layer only. All data fetching lives in data.py, math in analysis.py,
AI calls inline here, portfolio persistence in storage.py.
"""

import streamlit as st
import pandas as pd
import anthropic
import io
import re
import json
from datetime import datetime

from data import (
    get_ticker_data,
    get_spy_benchmark,
    get_risk_free_rate,
    RISK_FREE_RATE_FALLBACK,
)
from analysis import (
    compute_portfolio_stats,
    compute_correlation_matrix,
    compute_weighted_beta,
    compute_weighted_volatility,
    build_portfolio_context,
)
from storage import (
    list_portfolios,
    save_portfolio,
    load_portfolio,
    delete_portfolio,
)

st.set_page_config(page_title="Portfolio Insight Tool", layout="centered")

# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f5f5f5 !important;
    color: #111111;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 900px; padding: 0 1.5rem 4rem 1.5rem !important; }

/* HEADER */
.pit-header { padding: 2.5rem 0 1.5rem 0; border-bottom: 1px solid #e5e7eb; }
.pit-header-title { font-size: 1.5rem; font-weight: 600; color: #111111; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; }
.pit-header-sub { font-size: 0.875rem; color: #6b7280; font-weight: 400; margin: 0; }

/* METRICS BAR */
.pit-metrics-bar { background:#fff; border-bottom:1px solid #e5e7eb; padding:1rem 0; display:flex; }
.pit-metric-item { flex:1; padding:0 1.25rem; border-right:1px solid #e5e7eb; }
.pit-metric-item:first-child { padding-left:0; }
.pit-metric-item:last-child { border-right:none; }
.pit-metric-label { font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#9ca3af; margin-bottom:0.3rem; }
.pit-metric-value { font-family:'DM Mono',monospace; font-size:1.2rem; font-weight:500; color:#111111; line-height:1; }
.pit-metric-value.pos { color:#16a34a; }
.pit-metric-value.neg { color:#dc2626; }
.pit-metric-value.warn { color:#dc2626; }
.pit-metric-value.good { color:#16a34a; }

/* SECTION LABEL */
.pit-label { font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#9ca3af; margin:0 0 1rem 0; padding-left:0.7rem; border-left:2px solid #2563eb; }

/* SURFACE */
.pit-surface { background:#fff; border:0.5px solid #e5e7eb; border-radius:8px; padding:1.5rem; margin-bottom:1rem; }

/* INPUT HINT */
.pit-input-hint { font-size:0.8rem; color:#6b7280; margin-bottom:0.75rem; }

/* HOLDINGS TABLE */
.h-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.h-table th { font-size:0.62rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#9ca3af; padding:0.4rem 0.75rem; text-align:left; border-bottom:1px solid #e5e7eb; }
.h-table th.r { text-align:right; }
.h-table td { padding:0.65rem 0.75rem; border-bottom:0.5px solid #f3f4f6; color:#111111; vertical-align:middle; }
.h-table td.r { text-align:right; font-family:'DM Mono',monospace; font-size:0.82rem; }
.h-table tr:last-child td { border-bottom:none; }
.h-table .tk { font-weight:600; letter-spacing:0.03em; font-size:0.875rem; }
.h-table .badge { display:inline-block; font-size:0.6rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; padding:0.1rem 0.4rem; border-radius:4px; background:#f3f4f6; color:#6b7280; }
.h-table .badge.etf { background:#eff6ff; color:#2563eb; }
.h-table .neg { color:#dc2626; }
.h-table .pos { color:#16a34a; }

/* BENCHMARK TABLE */
.b-table { width:100%; border-collapse:collapse; font-size:0.875rem; }
.b-table th { font-size:0.62rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#9ca3af; padding:0.4rem 0.75rem; border-bottom:1px solid #e5e7eb; text-align:left; }
.b-table th:not(:first-child) { text-align:right; }
.b-table td { padding:0.7rem 0.75rem; border-bottom:0.5px solid #f3f4f6; color:#111111; }
.b-table td:not(:first-child) { text-align:right; font-family:'DM Mono',monospace; font-size:0.85rem; }
.b-table tr:last-child td { border-bottom:none; }
.b-table .row-label { color:#374151; font-weight:500; font-size:0.85rem; }
.b-table .pos { color:#16a34a; }
.b-table .neg { color:#dc2626; }

/* CORRELATION TABLE */
.c-table { border-collapse:collapse; font-family:'DM Mono',monospace; font-size:0.82rem; }
.c-table th { font-size:0.62rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#9ca3af; padding:0.4rem 0.75rem; text-align:center; border-bottom:1px solid #e5e7eb; }
.c-table th:first-child { text-align:left; }
.c-table td { padding:0.5rem 0.75rem; text-align:center; font-weight:500; border-bottom:0.5px solid #f3f4f6; min-width:60px; }
.c-table td:first-child { text-align:left; font-family:'DM Sans',sans-serif; font-size:0.75rem; font-weight:600; color:#374151; text-transform:uppercase; letter-spacing:0.04em; min-width:60px; }
.c-table tr:last-child td { border-bottom:none; }

/* SECTOR TABLE */
.s-table { width:100%; border-collapse:collapse; font-size:0.875rem; }
.s-table th { font-size:0.62rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#9ca3af; padding:0.4rem 0.75rem; border-bottom:1px solid #e5e7eb; text-align:left; }
.s-table th.r { text-align:right; }
.s-table td { padding:0.65rem 0.75rem; border-bottom:0.5px solid #f3f4f6; color:#374151; vertical-align:middle; }
.s-table td.r { text-align:right; font-family:'DM Mono',monospace; font-size:0.82rem; color:#111111; white-space:nowrap; }
.s-table tr:last-child td { border-bottom:none; }
.s-bar-bg { background:#f3f4f6; border-radius:2px; height:4px; width:100%; min-width:80px; }
.s-bar-fill { background:#2563eb; border-radius:2px; height:4px; }

/* SAVED PORTFOLIOS */
.saved-row { display:flex; align-items:center; padding:0.6rem 0; border-bottom:0.5px solid #f3f4f6; gap:0.75rem; }
.saved-row:last-child { border-bottom:none; }
.saved-name { flex:1; font-size:0.875rem; font-weight:500; color:#111111; }
.saved-tickers { font-size:0.75rem; color:#6b7280; font-family:'DM Mono',monospace; }

/* AI ANALYSIS */
.ai-text { font-size:0.9375rem; line-height:1.8; color:#374151; font-weight:400; }

/* FOLLOW-UP Q&A */
.qa-divider { border:none; border-top:1px solid #e5e7eb; margin:1.5rem 0 1.25rem 0; }
.qa-question { font-size:0.8rem; font-weight:600; color:#374151; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem; }
.qa-answer { font-size:0.9375rem; line-height:1.8; color:#374151; margin-bottom:1.25rem; padding-bottom:1.25rem; border-bottom:0.5px solid #f3f4f6; }
.qa-answer:last-child { border-bottom:none; }

/* CAPTION */
.pit-caption { font-size:0.72rem; color:#9ca3af; margin-top:0.875rem; line-height:1.6; }

/* ALERTS */
.pit-warn { background:#fffbeb; border:0.5px solid #fcd34d; border-radius:6px; padding:0.6rem 0.875rem; font-size:0.8rem; color:#92400e; margin-top:0.5rem; line-height:1.5; }
.pit-alert { background:#fef2f2; border:0.5px solid #fca5a5; border-radius:6px; padding:0.6rem 0.875rem; font-size:0.8rem; color:#991b1b; margin-top:0.5rem; line-height:1.5; }
.pit-success { background:#f0fdf4; border:0.5px solid #86efac; border-radius:6px; padding:0.6rem 0.875rem; font-size:0.8rem; color:#166534; margin-top:0.5rem; line-height:1.5; }

/* BUTTONS */
div.stButton > button { background-color:#2563eb !important; color:white !important; border:none !important; border-radius:6px !important; padding:0.6rem 1.5rem !important; font-family:'DM Sans',sans-serif !important; font-size:0.875rem !important; font-weight:500 !important; cursor:pointer !important; width:100% !important; margin-top:0.75rem !important; }
div.stButton > button:hover { background-color:#1d4ed8 !important; }
div[data-testid="stFormSubmitButton"] > button { background-color:#2563eb !important; color:white !important; border:none !important; border-radius:6px !important; padding:0.6rem 1.5rem !important; font-family:'DM Sans',sans-serif !important; font-size:0.875rem !important; font-weight:500 !important; width:100% !important; margin-top:0.75rem !important; }
div[data-testid="stFormSubmitButton"] > button:hover { background-color:#1d4ed8 !important; }

/* STREAMLIT OVERRIDES */
.stTextInput input { border-radius:6px !important; border:0.5px solid #e5e7eb !important; font-family:'DM Sans',sans-serif !important; font-size:0.875rem !important; background:#fff !important; color:#111111 !important; padding:0.5rem 0.75rem !important; }
.stTextInput input:focus { border-color:#2563eb !important; box-shadow:0 0 0 2px rgba(37,99,235,0.1) !important; }
.stNumberInput input { border-radius:6px !important; border:0.5px solid #e5e7eb !important; font-family:'DM Mono',monospace !important; font-size:0.875rem !important; background:#fff !important; color:#111111 !important; }
.stTextArea textarea { border-radius:6px !important; border:0.5px solid #e5e7eb !important; font-family:'DM Mono',monospace !important; font-size:0.85rem !important; background:#fff !important; color:#111111 !important; }
.stTextArea textarea:focus { border-color:#2563eb !important; box-shadow:0 0 0 2px rgba(37,99,235,0.1) !important; }
div[data-testid="stRadio"] label { font-family:'DM Sans',sans-serif !important; font-size:0.875rem !important; color:#374151 !important; }
.stProgress > div > div { background-color:#2563eb !important; }
label[data-testid="stWidgetLabel"] p { font-family:'DM Sans',sans-serif !important; font-size:0.8rem !important; color:#6b7280 !important; font-weight:500 !important; }

/* MISC */
.pit-divider { border:none; border-top:0.5px solid #e5e7eb; margin:1.5rem 0; }
.pit-empty { text-align:center; padding:3rem 1.5rem; color:#9ca3af; font-size:0.875rem; }
.pit-empty-icon { font-size:2rem; margin-bottom:0.75rem; opacity:0.4; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
AI_TIMEOUT_SECONDS = 45

try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    st.error("API key not found. Please add ANTHROPIC_API_KEY to .streamlit/secrets.toml")
    st.stop()

# ============================================================
# SESSION STATE
# ============================================================
for key, default in [
    ("results_ready", False),
    ("df", None),
    ("summary", {}),
    ("ai_analysis", ""),
    ("ai_failed", False),
    ("active_tab", "summary"),
    ("pending_holdings", {}),
    ("qa_history", []),
    ("portfolio_context", ""),
    ("save_msg", ""),        # feedback after save/delete
    ("save_msg_type", ""),   # "success" or "error"
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# PURE HELPERS (UI-level only)
# ============================================================

def fix_dollar_formatting(text: str) -> str:
    return text.replace("`", "")


def corr_cell_style(val) -> str:
    try:
        val = float(val)
    except Exception:
        return ""
    if val >= 0.8:   return "background:#ef4444;color:white;"
    if val >= 0.6:   return "background:#f97316;color:white;"
    if val >= 0.4:   return "background:#fbbf24;color:#111111;"
    if val >= 0.2:   return "background:#bbf7d0;color:#111111;"
    if val >= 0.0:   return "background:#f9fafb;color:#374151;"
    if val >= -0.2:  return "background:#eff6ff;color:#1e40af;"
    return "background:#dbeafe;color:#1e40af;"


def parse_paste_input(raw_text: str, mode: str = "dollars") -> tuple:
    results, errors = {}, []
    for i, line in enumerate(raw_text.strip().split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[\s\t]+", line)
        if len(parts) < 2:
            errors.append(f"Line {i}: '{line}' — expected TICKER and a number")
            continue
        raw_ticker = parts[0].upper()
        raw_value = parts[1].replace("%", "").replace("$", "").replace(",", "")
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"Line {i}: '{line}' — '{parts[1]}' is not a valid number")
            continue
        if value <= 0:
            errors.append(f"Line {i}: '{raw_ticker}' — value must be greater than zero")
            continue
        if mode == "percentages" and value > 100:
            errors.append(f"Line {i}: '{raw_ticker}' — {value}% exceeds 100%")
            continue
        if raw_ticker in results:
            errors.append(f"Line {i}: '{raw_ticker}' appears more than once — using first entry")
            continue
        results[raw_ticker] = value
    return results, errors


def percentages_to_dollars(pct_dict: dict, total_value: float) -> dict:
    return {t: round(p / 100 * total_value, 2) for t, p in pct_dict.items()}


def parse_csv_upload(file_bytes: bytes) -> tuple:
    warnings, holdings = [], {}
    try:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), skiprows=1)
        df.columns = [str(c).strip() for c in df.columns]

        ticker_col = next((c for c in ["Symbol", "Ticker", "TICKER", "SYMBOL", "Symb"] if c in df.columns), None)
        value_col  = next((c for c in ["Current Value", "Market Value", "Value", "Mkt Val", "Market Val", "Current Val", "Amount"] if c in df.columns), None)

        if ticker_col is None:
            return {}, ["Could not find a ticker/symbol column. Expected 'Symbol' or 'Ticker'."]
        if value_col is None:
            return {}, ["Could not find a value column. Expected 'Current Value' or 'Market Value'."]

        skipped_rows = []
        for _, row in df.iterrows():
            raw_ticker = str(row[ticker_col]).strip()
            raw_value  = str(row[value_col]).strip()
            if not raw_ticker or raw_ticker in ("--", "nan", "N/A", ""):
                continue
            if any(kw in raw_ticker.upper() for kw in ["CASH", "SWEEP", "MONEY MARKET", "MM "]):
                continue
            cleaned = re.sub(r"[$,\s]", "", raw_value).replace("(", "-").replace(")", "")
            try:
                amount = float(cleaned)
            except ValueError:
                skipped_rows.append(f"{raw_ticker} (could not parse '{raw_value}')")
                continue
            if amount <= 0:
                continue
            ticker = raw_ticker.upper()
            holdings[ticker] = holdings.get(ticker, 0) + round(amount, 2)

        if skipped_rows:
            warnings.append("Skipped: " + ", ".join(skipped_rows))
        return holdings, warnings
    except Exception as e:
        return {}, [f"Failed to parse CSV: {e}"]


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def run_analysis(holdings: dict) -> None:
    """
    Stage 1 — fetch data and compute metrics, write to session state.
    Stage 2 — AI analysis (failure here still shows all metrics).
    """
    rows, missing_data_notes, warnings, skipped, ticker_histories = [], [], [], [], {}

    progress = st.progress(0, text="Fetching market data…")
    tickers = list(holdings.keys())

    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"Fetching {ticker}…")
        data = get_ticker_data(ticker)

        if not data["valid"]:
            skipped.append(f"{ticker} — {data.get('error', 'unknown error')}")
            continue

        if data["missing"]:
            if data["is_etf"]:
                missing_data_notes.append(f"{ticker} is an ETF — sector/beta not applicable")
            else:
                missing_data_notes.append(f"{ticker} missing: {', '.join(data['missing'])}")

        warnings.extend(data.get("warnings", []))
        ticker_histories[ticker] = data["history"]
        rows.append({
            "Ticker":          ticker,
            "Amount Invested": holdings[ticker],
            "Sector":          data["sector"],
            "Beta":            data["beta"],
            "Volatility %":    data["volatility"],
            "Max Drawdown %":  data["max_drawdown"],
            "Type":            "ETF" if data["is_etf"] else "Stock",
        })

    progress.empty()

    if not rows:
        st.error("No valid tickers found. Check your ticker symbols and try again.")
        return

    df = pd.DataFrame(rows)
    total = df["Amount Invested"].sum()
    df["Allocation %"] = (df["Amount Invested"] / total * 100).round(2)

    weighted_beta       = compute_weighted_beta(df, total)
    weighted_volatility = compute_weighted_volatility(df, total)

    sector_breakdown = df.groupby("Sector")["Allocation %"].sum().round(2).to_dict()
    concentrated_sectors = {
        sec: pct for sec, pct in sector_breakdown.items()
        if pct > 60 and sec not in ["ETF (diversified)", "Unknown"]
    }

    spy_data         = get_spy_benchmark(existing_histories=ticker_histories)
    risk_free_rate   = get_risk_free_rate()

    portfolio_return, portfolio_max_drawdown, sharpe_ratio = compute_portfolio_stats(
        df, holdings, ticker_histories, weighted_volatility, risk_free_rate
    )

    spy_sharpe = None
    if spy_data and spy_data["volatility"] and spy_data["volatility"] > 0:
        spy_sharpe = round((spy_data["one_year_return"] - risk_free_rate) / spy_data["volatility"], 2)

    corr_matrix, high_corr_pairs = compute_correlation_matrix(ticker_histories)

    summary = {
        "total": total, "weighted_beta": weighted_beta,
        "weighted_volatility": weighted_volatility,
        "sector_breakdown": sector_breakdown,
        "concentrated_sectors": concentrated_sectors,
        "skipped": skipped, "missing_data_notes": missing_data_notes,
        "warnings": warnings, "spy_data": spy_data,
        "portfolio_return": portfolio_return,
        "portfolio_max_drawdown": portfolio_max_drawdown,
        "sharpe_ratio": sharpe_ratio, "spy_sharpe": spy_sharpe,
        "risk_free_rate": risk_free_rate,
        "corr_matrix": corr_matrix, "high_corr_pairs": high_corr_pairs,
        "holdings": holdings,
    }

    portfolio_context = build_portfolio_context(
        df, summary, portfolio_return, portfolio_max_drawdown,
        sharpe_ratio, spy_data, spy_sharpe, risk_free_rate,
        high_corr_pairs, corr_matrix,
    )

    # Write metrics now — AI failure won't wipe these
    st.session_state.update({
        "results_ready":     True,
        "df":                df,
        "active_tab":        "summary",
        "summary":           summary,
        "portfolio_context": portfolio_context,
        "qa_history":        [],
        "ai_analysis":       "",
        "ai_failed":         False,
    })

    # Stage 2: AI
    prompt = (
        "You are a senior portfolio analyst. Give the investor a direct, unvarnished read of their portfolio. "
        "Write like you're sending a quick internal note to a colleague — terse, specific, no hedging. "
        "No markdown, no bold, no bullet points. Plain prose only. "
        "No phrases like 'it may be worth considering' or 'investors should be aware'. "
        "Call things what they are. If risk is high, say so. If diversification is poor, say so. "
        "Do not pad. Every sentence should carry information.\n\n"
        + portfolio_context + "\n\n"
        "Write 4-5 sentences covering:\n"
        "1. What this portfolio is actually concentrated in and what that exposure means.\n"
        "2. Whether the returns justify the risk — use beta, volatility, drawdown, and Sharpe vs SPY specifically.\n"
        "3. Which holdings drive risk and whether any correlations create hidden concentration.\n"
        "4. The one specific thing this investor needs to watch."
    )

    with st.spinner("Generating analysis…"):
        try:
            client = anthropic.Anthropic(api_key=api_key, timeout=AI_TIMEOUT_SECONDS)
            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            st.session_state.ai_analysis = fix_dollar_formatting(message.content[0].text)
        except anthropic.APITimeoutError:
            st.session_state.ai_failed = True
            st.session_state.ai_analysis = (
                "Analysis timed out. Your portfolio metrics above are accurate. "
                "Try switching to the AI Analysis tab and refreshing."
            )
        except Exception:
            st.session_state.ai_failed = True
            st.session_state.ai_analysis = (
                "Analysis could not be generated right now. "
                "Your portfolio metrics are accurate."
            )


def answer_question(question: str) -> str:
    prompt = (
        "You are a senior portfolio analyst. Answer a client follow-up question. "
        "Use actual numbers from their portfolio. No markdown, no bullets. Plain prose. "
        "Be terse. If the answer is short, keep it short.\n\n"
        + st.session_state.portfolio_context + "\n\n"
        "Client question: " + question
    )
    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=AI_TIMEOUT_SECONDS)
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return fix_dollar_formatting(message.content[0].text)
    except anthropic.APITimeoutError:
        return "The AI service timed out. Please try again."
    except Exception:
        return "Could not generate an answer right now. Please try again."


# ============================================================
# PAGE HEADER
# ============================================================
st.markdown("""
<div class="pit-header">
    <div class="pit-header-title">Portfolio Insight Tool</div>
    <div class="pit-header-sub">Institutional-grade portfolio analysis</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================
if not st.session_state.results_ready:
    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    # ---- Saved portfolios panel ----
    saved_names = list_portfolios()
    if saved_names:
        st.markdown('<div class="pit-surface">', unsafe_allow_html=True)
        st.markdown('<div class="pit-label">Saved Portfolios</div>', unsafe_allow_html=True)

        for name in saved_names:
            holdings_preview = load_portfolio(name) or {}
            ticker_str = "  ·  ".join(list(holdings_preview.keys())[:6])
            if len(holdings_preview) > 6:
                ticker_str += f"  +{len(holdings_preview)-6} more"

            col_name, col_load, col_del = st.columns([4, 1, 1])
            with col_name:
                st.markdown(
                    f'<div class="saved-name">{name}'
                    f'<br><span class="saved-tickers">{ticker_str}</span></div>',
                    unsafe_allow_html=True
                )
            with col_load:
                if st.button("Load", key=f"load_{name}"):
                    holdings = load_portfolio(name)
                    if holdings:
                        run_analysis(holdings)
                        st.rerun()
            with col_del:
                if st.button("Delete", key=f"del_{name}"):
                    ok, msg = delete_portfolio(name)
                    st.session_state.save_msg = msg
                    st.session_state.save_msg_type = "success" if ok else "error"
                    st.rerun()

        if st.session_state.save_msg:
            cls = "pit-success" if st.session_state.save_msg_type == "success" else "pit-warn"
            st.markdown(f'<div class="{cls}">{st.session_state.save_msg}</div>', unsafe_allow_html=True)
            st.session_state.save_msg = ""

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Input panel ----
    st.markdown('<div class="pit-surface">', unsafe_allow_html=True)
    st.markdown('<div class="pit-label">Portfolio Input</div>', unsafe_allow_html=True)

    input_method = st.radio(
        "Input method",
        ["Enter holdings", "Paste a list", "Percentages", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    holdings: dict = {}
    input_errors: list = []

    # METHOD 1: one by one
    if input_method == "Enter holdings":
        st.markdown('<p class="pit-input-hint">Enter a ticker and dollar amount, then click Add.</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_ticker = st.text_input("Ticker", placeholder="e.g. AAPL", label_visibility="collapsed", key="new_ticker").upper().strip()
        with col2:
            new_amount = st.number_input("Amount ($)", min_value=0.0, step=100.0, label_visibility="collapsed", key="new_amount")
        with col3:
            if st.button("Add", key="add_btn"):
                if new_ticker and new_amount > 0:
                    st.session_state.pending_holdings[new_ticker] = new_amount
                    st.rerun()

        if st.session_state.pending_holdings:
            st.markdown('<hr class="pit-divider">', unsafe_allow_html=True)
            rows_html = "".join(f'<tr><td class="tk">{tk}</td><td class="r">${amt:,.0f}</td></tr>'
                                for tk, amt in st.session_state.pending_holdings.items())
            st.markdown(f'<table class="h-table"><thead><tr><th>Ticker</th><th class="r">Amount</th></tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)
            col_a, col_b = st.columns([3, 1])
            with col_b:
                if st.button("Clear all", key="clear_btn"):
                    st.session_state.pending_holdings = {}
                    st.rerun()
        holdings = st.session_state.pending_holdings.copy()

    # METHOD 2: paste dollars
    elif input_method == "Paste a list":
        st.markdown('<p class="pit-input-hint">One holding per line — TICKER AMOUNT. Commas OK: <code>AAPL 5,000</code></p>', unsafe_allow_html=True)
        raw = st.text_area("", height=120, placeholder="AAPL 5000\nMSFT 3000\nSPY 2000", label_visibility="collapsed")
        if raw.strip():
            holdings, input_errors = parse_paste_input(raw, mode="dollars")

    # METHOD 3: percentages
    elif input_method == "Percentages":
        st.markdown('<p class="pit-input-hint">One holding per line — TICKER PERCENTAGE. Must sum to 100.</p>', unsafe_allow_html=True)
        total_value = st.number_input("Total portfolio value ($)", min_value=1.0, step=1000.0, key="pct_total_value")
        raw = st.text_area("", height=120, placeholder="AAPL 40\nMSFT 30\nSPY 30", label_visibility="collapsed")
        if raw.strip() and total_value > 0:
            pct_dict, input_errors = parse_paste_input(raw, mode="percentages")
            if pct_dict:
                total_pct = sum(pct_dict.values())
                if abs(total_pct - 100) > 0.5:
                    input_errors.append(f"Percentages sum to {total_pct:.1f}% — must add up to 100%.")
                else:
                    holdings = percentages_to_dollars(pct_dict, total_value)

    # METHOD 4: CSV
    elif input_method == "Upload CSV":
        st.markdown('<p class="pit-input-hint">Upload a brokerage CSV with a <strong>Symbol</strong> column and a <strong>Current Value</strong> or <strong>Market Value</strong> column. Cash positions are skipped automatically.</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            holdings, csv_warnings = parse_csv_upload(uploaded.read())
            for w in csv_warnings:
                st.markdown(f'<div class="pit-warn">{w}</div>', unsafe_allow_html=True)
            if holdings:
                rows_html = "".join(f'<tr><td class="tk">{tk}</td><td class="r">${amt:,.0f}</td></tr>' for tk, amt in holdings.items())
                st.markdown('<hr class="pit-divider">', unsafe_allow_html=True)
                st.markdown(f'<table class="h-table"><thead><tr><th>Ticker</th><th class="r">Value</th></tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)

    for err in input_errors:
        st.markdown(f'<div class="pit-warn">{err}</div>', unsafe_allow_html=True)

    if st.button("Analyze Portfolio"):
        if not holdings:
            st.error("Please enter at least one holding.")
        else:
            run_analysis(holdings)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# RESULTS
# ============================================================
if st.session_state.results_ready:
    s  = st.session_state.summary
    df = st.session_state.df

    port_return = s.get("portfolio_return")
    sharpe      = s.get("sharpe_ratio")
    drawdown    = s.get("portfolio_max_drawdown")
    vol         = s.get("weighted_volatility")

    return_prefix = "+" if (port_return is not None and port_return >= 0) else ""
    return_class  = "pos" if (port_return is not None and port_return >= 0) else "neg"
    sharpe_class  = "good" if (sharpe is not None and sharpe >= 1.0) else ("warn" if sharpe is not None else "")
    dd_str        = f"{drawdown}%" if drawdown is not None else "N/A"
    vol_str       = f"{vol}%" if vol is not None else "N/A"
    sharpe_str    = str(sharpe) if sharpe is not None else "N/A"
    return_val    = f"{port_return}%" if port_return is not None else "N/A"

    # ---- Metrics bar ----
    st.markdown(f"""
    <div class="pit-metrics-bar">
        <div class="pit-metric-item"><div class="pit-metric-label">Total Value</div><div class="pit-metric-value">${s['total']:,.0f}</div></div>
        <div class="pit-metric-item"><div class="pit-metric-label">1-Year Return</div><div class="pit-metric-value {return_class}">{return_prefix}{return_val}</div></div>
        <div class="pit-metric-item"><div class="pit-metric-label">Beta</div><div class="pit-metric-value">{s['weighted_beta']}</div></div>
        <div class="pit-metric-item"><div class="pit-metric-label">Volatility</div><div class="pit-metric-value">{vol_str}</div></div>
        <div class="pit-metric-item"><div class="pit-metric-label">Max Drawdown</div><div class="pit-metric-value neg">{dd_str}</div></div>
        <div class="pit-metric-item"><div class="pit-metric-label">Sharpe Ratio</div><div class="pit-metric-value {sharpe_class}">{sharpe_str}</div></div>
    </div>""", unsafe_allow_html=True)

    # ---- Tabs ----
    TABS = [("summary","Summary"),("benchmark","Benchmark"),("correlation","Correlation"),
            ("holdings","Holdings"),("sectors","Sectors"),("analysis","AI Analysis")]

    st.markdown('<div style="background:#fff;border-bottom:1px solid #e5e7eb;margin-bottom:1.5rem;">', unsafe_allow_html=True)
    tab_cols = st.columns(len(TABS))
    for i, (col, (tab_id, tab_label)) in enumerate(zip(tab_cols, TABS)):
        with col:
            is_active = st.session_state.active_tab == tab_id
            style = (
                "background:transparent!important;color:#2563eb!important;"
                "border-bottom:2px solid #2563eb!important;border-radius:0!important;"
                "padding:0.75rem 0.25rem!important;font-size:0.8rem!important;"
                "font-weight:500!important;width:100%!important;margin:0!important;box-shadow:none!important;"
            ) if is_active else (
                "background:transparent!important;color:#6b7280!important;"
                "border:none!important;border-radius:0!important;"
                "padding:0.75rem 0.25rem!important;font-size:0.8rem!important;"
                "font-weight:500!important;width:100%!important;margin:0!important;box-shadow:none!important;"
            )
            st.markdown(f'<style>div[data-testid="stColumn"]:nth-child({i+1}) div.stButton>button{{{style}}}</style>', unsafe_allow_html=True)
            if st.button(tab_label, key=f"tab_{tab_id}"):
                st.session_state.active_tab = tab_id
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    active = st.session_state.active_tab

    # ---- Save portfolio + New Analysis row ----
    col_save, col_name, col_new = st.columns([1, 2, 1])
    with col_new:
        if st.button("← New Analysis"):
            st.session_state.results_ready = False
            st.session_state.pending_holdings = {}
            st.session_state.qa_history = []
            st.session_state.portfolio_context = ""
            st.rerun()
    with col_name:
        save_name = st.text_input("Portfolio name", placeholder="e.g. Retirement account", label_visibility="collapsed", key="save_name_input")
    with col_save:
        if st.button("Save Portfolio"):
            current_holdings = s.get("holdings", {})
            if not save_name.strip():
                st.session_state.save_msg = "Enter a name before saving."
                st.session_state.save_msg_type = "error"
            elif current_holdings:
                ok, msg = save_portfolio(save_name.strip(), current_holdings)
                st.session_state.save_msg = msg
                st.session_state.save_msg_type = "success" if ok else "error"
            st.rerun()

    if st.session_state.save_msg:
        cls = "pit-success" if st.session_state.save_msg_type == "success" else "pit-warn"
        st.markdown(f'<div class="{cls}">{st.session_state.save_msg}</div>', unsafe_allow_html=True)
        st.session_state.save_msg = ""

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    # ============================================================
    # TAB: SUMMARY
    # ============================================================
    if active == "summary":
        st.markdown('<div class="pit-surface">', unsafe_allow_html=True)
        st.markdown('<div class="pit-label">Portfolio Composition</div>', unsafe_allow_html=True)
        rows_html = ""
        for _, row in df.iterrows():
            badge_cls = "etf" if row["Type"] == "ETF" else ""
            dd    = row["Max Drawdown %"]
            vol_v = row["Volatility %"]
            dd_html  = f'<span class="neg">{dd}%</span>' if pd.notna(dd) else "—"
            vol_html = f'{vol_v}%' if pd.notna(vol_v) else "—"
            rows_html += f"""<tr>
                <td class="tk">{row['Ticker']}</td>
                <td><span class="badge {badge_cls}">{row['Type']}</span></td>
                <td class="r">${row['Amount Invested']:,.0f}</td>
                <td class="r">{row['Allocation %']}%</td>
                <td>{row['Sector']}</td>
                <td class="r">{row['Beta']}</td>
                <td class="r">{vol_html}</td>
                <td class="r">{dd_html}</td>
            </tr>"""
        st.markdown(f'<table class="h-table"><thead><tr><th>Ticker</th><th>Type</th><th class="r">Amount</th><th class="r">Alloc</th><th>Sector</th><th class="r">Beta</th><th class="r">Vol</th><th class="r">Max DD</th></tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="pit-surface">', unsafe_allow_html=True)
        st.markdown('<div class="pit-label">Sector Exposure</div>', unsafe_allow_html=True)
        sector_rows = ""
        for sector, pct in sorted(s["sector_breakdown"].items(), key=lambda x: x[1], reverse=True):
            sector_rows += f'<tr><td>{sector}</td><td class="r">{pct}%</td><td><div class="s-bar-bg"><div class="s-bar-fill" style="width:{min(pct,100)}%"></div></div></td></tr>'
        st.markdown(f'<table class="s-table"><thead><tr><th>Sector</th><th class="r">Allocation</th><th></th></tr></thead><tbody>{sector_rows}</tbody></table>', unsafe_allow_html=True)
        for sec, pct in s["concentrated_sectors"].items():
            st.markdown(f'<div class="pit-alert">{pct}% in {sec} — significant concentration risk</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # TAB: BENCHMARK
    # ============================================================
    elif active == "benchmark":
        spy = s.get("spy_data")
        if spy:
            st.markdown('<div class="pit-surface">', unsafe_allow_html=True)
            st.markdown('<div class="pit-label">Portfolio vs. SPY</div>', unsafe_allow_html=True)
            p_ret     = s.get("portfolio_return")
            p_sharpe  = s.get("sharpe_ratio")
            spy_sharpe= s.get("spy_sharpe")
            rfr       = s.get("risk_free_rate", RISK_FREE_RATE_FALLBACK)
            p_ret_str = (("+" if p_ret >= 0 else "") + str(p_ret) + "%") if p_ret is not None else "N/A"
            p_ret_cls = "pos" if (p_ret is not None and p_ret >= 0) else "neg"
            s_ret_cls = "pos" if spy["one_year_return"] >= 0 else "neg"
            s_ret_str = ("+" if spy["one_year_return"] >= 0 else "") + str(spy["one_year_return"]) + "%"
            p_sh_cls  = "pos" if (p_sharpe is not None and p_sharpe >= 1.0) else "neg"
            s_sh_cls  = "pos" if (spy_sharpe is not None and spy_sharpe >= 1.0) else "neg"
            st.markdown(f"""<table class="b-table">
                <thead><tr><th>Metric</th><th>Your Portfolio</th><th>SPY</th></tr></thead>
                <tbody>
                <tr><td class="row-label">Beta</td><td>{s['weighted_beta']}</td><td>1.00</td></tr>
                <tr><td class="row-label">Annualized Volatility</td><td>{vol_str}</td><td>{spy['volatility']}%</td></tr>
                <tr><td class="row-label">1-Year Return</td><td class="{p_ret_cls}">{p_ret_str}</td><td class="{s_ret_cls}">{s_ret_str}</td></tr>
                <tr><td class="row-label">Sharpe Ratio</td><td class="{p_sh_cls}">{str(p_sharpe) if p_sharpe is not None else 'N/A'}</td><td class="{s_sh_cls}">{str(spy_sharpe) if spy_sharpe is not None else 'N/A'}</td></tr>
                <tr><td class="row-label">Max Drawdown</td><td class="neg">{dd_str}</td><td>—</td></tr>
                </tbody></table>
                <div class="pit-caption">Beta: market sensitivity — SPY=1.00 by definition &nbsp;·&nbsp; Sharpe=(Return−RFR)/Vol &nbsp;·&nbsp; Risk-free rate: {rfr}% &nbsp;·&nbsp; Sharpe ≥1.0 good · ≥2.0 strong</div>""",
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pit-empty"><div class="pit-empty-icon">◎</div>SPY benchmark data unavailable.</div>', unsafe_allow_html=True)

    # ============================================================
    # TAB: CORRELATION
    # ============================================================
    elif active == "correlation":
        corr_matrix    = s.get("corr_matrix")
        high_corr_pairs = s.get("high_corr_pairs", [])
        if corr_matrix is not None:
            st.markdown('<div class="pit-surface">', unsafe_allow_html=True)
            st.markdown('<div class="pit-label">Holdings Correlation</div>', unsafe_allow_html=True)
            tickers_c    = corr_matrix.columns.tolist()
            header_cells = "<th></th>" + "".join(f"<th>{t}</th>" for t in tickers_c)
            rows_html    = ""
            for rt in tickers_c:
                cells = f"<td>{rt}</td>"
                for ct in tickers_c:
                    val   = corr_matrix.loc[rt, ct]
                    style = corr_cell_style(val)
                    cells += f'<td style="{style}">{val}</td>'
                rows_html += f"<tr>{cells}</tr>"
            st.markdown(f'<table class="c-table"><thead><tr>{header_cells}</tr></thead><tbody>{rows_html}</tbody></table><div class="pit-caption">1.0=lockstep &nbsp;·&nbsp; 0.0=no relation &nbsp;·&nbsp; Negative(blue)=good diversifier &nbsp;·&nbsp; Red≥0.80=hidden concentration</div>', unsafe_allow_html=True)
            for pair in high_corr_pairs:
                st.markdown(f'<div class="pit-warn">{pair["pair"]} — correlation {pair["correlation"]}. These holdings move together.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pit-empty"><div class="pit-empty-icon">◎</div>Need at least 2 holdings to calculate correlation.</div>', unsafe_allow_html=True)

    # ============================================================
    # TAB: HOLDINGS
    # ============================================================
    elif active == "holdings":
        st.markdown('<div class="pit-surface">', unsafe_allow_html=True)
        st.markdown('<div class="pit-label">Full Holdings Detail</div>', unsafe_allow_html=True)
        rows_html = ""
        for _, row in df.iterrows():
            badge_cls = "etf" if row["Type"] == "ETF" else ""
            dd    = row["Max Drawdown %"]
            vol_v = row["Volatility %"]
            dd_html  = f'<span class="neg">{dd}%</span>' if pd.notna(dd) else "—"
            vol_html = f'{vol_v}%' if pd.notna(vol_v) else "—"
            rows_html += f"""<tr>
                <td class="tk">{row['Ticker']}</td><td><span class="badge {badge_cls}">{row['Type']}</span></td>
                <td class="r">${row['Amount Invested']:,.0f}</td><td class="r">{row['Allocation %']}%</td>
                <td>{row['Sector']}</td><td class="r">{row['Beta']}</td>
                <td class="r">{vol_html}</td><td class="r">{dd_html}</td></tr>"""
        st.markdown(f'<table class="h-table"><thead><tr><th>Ticker</th><th>Type</th><th class="r">Amount</th><th class="r">Alloc</th><th>Sector</th><th class="r">Beta</th><th class="r">Vol</th><th class="r">Max DD</th></tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # TAB: SECTORS
    # ============================================================
    elif active == "sectors":
        st.markdown('<div class="pit-surface">', unsafe_allow_html=True)
        st.markdown('<div class="pit-label">Sector Breakdown</div>', unsafe_allow_html=True)
        sector_rows = ""
        for sector, pct in sorted(s["sector_breakdown"].items(), key=lambda x: x[1], reverse=True):
            sector_rows += f'<tr><td>{sector}</td><td class="r">{pct}%</td><td><div class="s-bar-bg"><div class="s-bar-fill" style="width:{min(pct,100)}%"></div></div></td></tr>'
        st.markdown(f'<table class="s-table"><thead><tr><th>Sector</th><th class="r">Allocation</th><th></th></tr></thead><tbody>{sector_rows}</tbody></table>', unsafe_allow_html=True)
        for sec, pct in s["concentrated_sectors"].items():
            st.markdown(f'<div class="pit-alert">{pct}% in {sec}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # TAB: AI ANALYSIS + FOLLOW-UP Q&A
    # ============================================================
    elif active == "analysis":
        st.markdown('<div class="pit-surface">', unsafe_allow_html=True)
        st.markdown('<div class="pit-label">Analysis</div>', unsafe_allow_html=True)
        if st.session_state.ai_failed:
            st.markdown(f'<div class="pit-warn">{st.session_state.ai_analysis}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-text">{st.session_state.ai_analysis}</div>', unsafe_allow_html=True)

        if st.session_state.qa_history:
            st.markdown('<hr class="qa-divider">', unsafe_allow_html=True)
            st.markdown('<div class="pit-label">Follow-up Questions</div>', unsafe_allow_html=True)
            for qa in st.session_state.qa_history:
                st.markdown(f'<div class="qa-question">{qa["question"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="qa-answer">{qa["answer"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="pit-surface">', unsafe_allow_html=True)
        st.markdown('<div class="pit-label">Ask a Question</div>', unsafe_allow_html=True)
        st.markdown('<p class="pit-input-hint">Ask anything about your portfolio — specific holdings, risk, what-ifs.</p>', unsafe_allow_html=True)
        with st.form(key="qa_form", clear_on_submit=True):
            user_q = st.text_input("Question", placeholder="e.g. Why is my Sharpe ratio low? What happens if I sell TSLA?", label_visibility="collapsed")
            if st.form_submit_button("Ask") and user_q.strip():
                with st.spinner("Thinking…"):
                    answer = answer_question(user_q.strip())
                st.session_state.qa_history.append({"question": user_q.strip(), "answer": answer})
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Data notes ----
    if s.get("warnings") or s.get("skipped") or s.get("missing_data_notes"):
        with st.expander("Data notes"):
            for w in s.get("warnings", []):
                st.markdown(f'<div class="pit-warn">{w}</div>', unsafe_allow_html=True)
            for sk in s.get("skipped", []):
                st.markdown(f'<div class="pit-warn">Skipped: {sk}</div>', unsafe_allow_html=True)
            for note in s.get("missing_data_notes", []):
                st.caption("— " + note)
