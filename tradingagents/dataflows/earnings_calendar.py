"""Earnings calendar fetcher using yfinance.

Provides upcoming and recent earnings dates for a ticker, so agents can
flag high-uncertainty binary events (earnings reports) and adjust their
confidence or position sizing accordingly.

Returns formatted plaintext blocks ready for prompt injection. Degrades
gracefully — returns a placeholder string rather than raising.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from .stockstats_utils import yf_retry

logger = logging.getLogger(__name__)


def fetch_earnings_calendar(ticker: str, trade_date: str, window_days: int = 14) -> str:
    """Fetch earnings dates near ``trade_date`` and return a formatted summary.

    Looks for earnings dates within ``window_days`` before and after the
    trade date. Highlights whether an earnings report is imminent (within
    the next 3 trading days) as a high-risk binary event.

    Args:
        ticker: Stock ticker symbol (e.g. "TSLA")
        trade_date: The date the agent is making a decision for (YYYY-MM-DD)
        window_days: How many days before/after to scan for earnings

    Returns:
        Formatted string describing upcoming/recent earnings dates and risk flags.
    """
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        earnings_dates = yf_retry(lambda: ticker_obj.earnings_dates)
    except Exception as exc:
        logger.warning("Earnings calendar fetch failed for %s: %s", ticker, exc)
        return f"<earnings calendar unavailable for {ticker.upper()}: {type(exc).__name__}>"

    if earnings_dates is None or earnings_dates.empty:
        return f"<no earnings dates found for {ticker.upper()}>"

    # Strip tz so comparisons with naive trade_dt don't raise TypeError
    if earnings_dates.index.tz is not None:
        earnings_dates.index = earnings_dates.index.tz_localize(None)

    trade_dt = pd.Timestamp(trade_date)
    window_start = trade_dt - timedelta(days=window_days)
    window_end = trade_dt + timedelta(days=window_days)

    # earnings_dates index is a DatetimeIndex of earnings dates
    # Filter to our window
    mask = (earnings_dates.index >= window_start) & (earnings_dates.index <= window_end)
    nearby = earnings_dates[mask].copy()

    if nearby.empty:
        # Check if there's a next earnings date beyond our window
        future = earnings_dates[earnings_dates.index > trade_dt]
        if not future.empty:
            next_date = future.index.min()
            days_until = (next_date - trade_dt).days
            return (
                f"No earnings dates within {window_days} days of {trade_date}.\n"
                f"Next earnings date: {next_date.strftime('%Y-%m-%d')} ({days_until} days away).\n"
                f"Earnings risk: LOW — no imminent binary event."
            )
        return f"<no earnings dates found near {trade_date} for {ticker.upper()}>"

    # Categorize: past (already reported) vs upcoming
    past_earnings = nearby[nearby.index < trade_dt]
    upcoming_earnings = nearby[nearby.index >= trade_dt]

    lines = [f"Earnings Calendar for {ticker.upper()} (window: {window_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}):"]
    lines.append("")

    # Flag imminent earnings (within 3 calendar days)
    imminent_threshold = trade_dt + timedelta(days=3)
    imminent = upcoming_earnings[upcoming_earnings.index <= imminent_threshold]

    if not imminent.empty:
        next_earnings = imminent.index.min()
        days_until = (next_earnings - trade_dt).days
        lines.append("⚠️  HIGH-RISK BINARY EVENT WARNING ⚠️")
        lines.append(f"Earnings report in {days_until} day(s): {next_earnings.strftime('%Y-%m-%d')}")
        lines.append("")
        lines.append("CRITICAL: Earnings reports are unpredictable binary events.")
        lines.append("Historical data shows stocks can move ±10-25% on earnings surprises.")
        lines.append("Recommendation: Reduce position confidence or move to HOLD unless")
        lines.append("there is overwhelming directional evidence from multiple sources.")
        lines.append("")
    elif not upcoming_earnings.empty:
        next_earnings = upcoming_earnings.index.min()
        days_until = (next_earnings - trade_dt).days
        lines.append(f"Upcoming earnings: {next_earnings.strftime('%Y-%m-%d')} ({days_until} days away)")
        if days_until <= 7:
            lines.append("Note: Earnings within 7 days — elevated uncertainty period.")
            lines.append("Earnings risk: MODERATE")
        else:
            lines.append("Earnings risk: LOW")
        lines.append("")

    if not past_earnings.empty:
        lines.append("Recent past earnings:")
        for date in past_earnings.index:
            days_ago = (trade_dt - date).days
            row = past_earnings.loc[date]
            eps_est = row.get("EPS Estimate", None)
            eps_act = row.get("Reported EPS", None)
            surprise = ""
            if pd.notna(eps_est) and pd.notna(eps_act):
                diff = eps_act - eps_est
                pct = (diff / abs(eps_est) * 100) if eps_est != 0 else 0
                surprise = f" | EPS: {eps_act:.2f} vs est {eps_est:.2f} ({pct:+.1f}% surprise)"
            lines.append(f"  {date.strftime('%Y-%m-%d')} ({days_ago} days ago){surprise}")
        lines.append("")

    if not upcoming_earnings.empty and imminent.empty:
        lines.append("Upcoming earnings dates:")
        for date in upcoming_earnings.index:
            days_until = (date - trade_dt).days
            lines.append(f"  {date.strftime('%Y-%m-%d')} ({days_until} days away)")

    return "\n".join(lines)
