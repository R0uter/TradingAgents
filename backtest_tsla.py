"""
backtest_tsla.py — TradingAgents backtester
============================================
Runs the TradingAgents framework day-by-day over a historical window,
records each signal, then scores it against the actual next-day price move.
Simulates a $100 portfolio and prints a full P&L summary.

Usage (from the TradingAgents project root):
    python backtest_tsla.py

Requirements: project venv must be active, or run with:
    /path/to/venv/bin/python backtest_tsla.py

Config: reads DEEPSEEK_API_KEY (and other keys) from .env automatically.
Results are saved incrementally to backtest_results.csv so a partial run
can be resumed without re-running completed days.
"""

import os
import sys
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Load .env from project root ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    # python-dotenv not installed — fall back to manual parse
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

# ── Imports ───────────────────────────────────────────────────────────────────
import yfinance as yf
import pandas as pd

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# ── Configuration ─────────────────────────────────────────────────────────────
TICKER      = "TSLA"
START_DATE  = "2024-10-01"   # first analysis date (agent looks at data up to here)
END_DATE    = "2024-10-31"   # last analysis date
INITIAL_CAPITAL = 100.0      # starting portfolio value in USD
RESULTS_CSV = Path(__file__).parent / "backtest_results.csv"

# Signal → position mapping
# Buy/Overweight  → +1 (long)
# Hold            →  0 (cash)
# Underweight/Sell → -1 (short)
SIGNAL_TO_POSITION = {
    "Buy":         +1,
    "Overweight":  +1,
    "Hold":         0,
    "Underweight": -1,
    "Sell":        -1,
}

# ── LLM config — uses DeepSeek (the only key present in .env) ─────────────────
config = DEFAULT_CONFIG.copy()
config["llm_provider"]    = "deepseek"
config["deep_think_llm"]  = "deepseek-v4-flash"
config["quick_think_llm"] = "deepseek-v4-flash"
config["backend_url"]     = "https://api.deepseek.com/v1"
config["max_debate_rounds"]       = 1
config["max_risk_discuss_rounds"] = 1

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_trading_days(ticker: str, start: str, end: str) -> list[str]:
    """Return list of trading day strings (YYYY-MM-DD) in [start, end]."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return df.index.strftime("%Y-%m-%d").tolist()


def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download close prices for the window + a few extra days for next-day lookup."""
    end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=7)
    df = yf.download(
        ticker,
        start=start,
        end=end_dt.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        multi_level_index=False,
    )
    # Flatten MultiIndex columns if present (newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].rename(columns={"Close": "close"})


def next_day_return(prices: pd.DataFrame, date_str: str) -> float | None:
    """
    Return the close-to-close return for the day AFTER date_str.
    e.g. if date_str = "2024-10-01", returns (close_Oct2 - close_Oct1) / close_Oct1.
    Returns None if there is no next trading day in the data.
    """
    idx = prices.index.get_loc(date_str) if date_str in prices.index else None
    if idx is None or idx + 1 >= len(prices):
        return None
    c0 = float(prices.iloc[idx]["close"])
    c1 = float(prices.iloc[idx + 1]["close"])
    return (c1 - c0) / c0


def load_completed_days(csv_path: Path) -> set[str]:
    """Return set of dates already recorded in the results CSV."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["date"] for row in reader}


def append_result(csv_path: Path, row: dict, write_header: bool):
    """Append one result row to the CSV."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_summary(results: list[dict]):
    """Print a formatted backtest summary."""
    if not results:
        print("No results to summarise.")
        return

    df = pd.DataFrame(results)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")

    active = df[df["position"] != 0]
    wins   = active[active["pnl"] > 0]
    losses = active[active["pnl"] < 0]

    final_capital = float(df["portfolio"].iloc[-1]) if len(df) else INITIAL_CAPITAL
    total_return  = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print("\n" + "=" * 60)
    print(f"  BACKTEST SUMMARY — {TICKER}  {START_DATE} → {END_DATE}")
    print("=" * 60)
    print(f"  Days analysed   : {len(df)}")
    print(f"  Active trades   : {len(active)}  (long: {(active['position']==1).sum()}, short: {(active['position']==-1).sum()})")
    print(f"  Cash days (Hold): {(df['position']==0).sum()}")
    print(f"  Win rate        : {len(wins)}/{len(active)} = {len(wins)/len(active)*100:.1f}%" if len(active) else "  Win rate        : n/a")
    print(f"  Starting capital: ${INITIAL_CAPITAL:.2f}")
    print(f"  Final capital   : ${final_capital:.2f}")
    print(f"  Total return    : {total_return:+.2f}%")
    print(f"  Avg daily P&L   : ${df['pnl'].mean():.4f}")
    print("=" * 60)

    print("\n  Day-by-day results:")
    print(f"  {'Date':<12} {'Signal':<12} {'Pos':>4} {'Actual%':>8} {'P&L':>8} {'Portfolio':>10}")
    print("  " + "-" * 58)
    for _, r in df.iterrows():
        pos   = int(float(r["position"]))
        ret   = float(r["ret"]) * 100 if r["ret"] not in ("", None) else float("nan")
        pnl   = float(r["pnl"])
        port  = float(r["portfolio"])
        arrow = "▲" if ret > 0 else ("▼" if ret < 0 else "─")
        correct = ""
        if pos != 0:
            correct = "✓" if (pos > 0 and ret > 0) or (pos < 0 and ret < 0) else "✗"
        print(f"  {r['date']:<12} {r['signal']:<12} {pos:>+4}  {arrow}{ret:>6.2f}%  {pnl:>+7.4f}  ${port:>8.4f}  {correct}")

    print()


# ── Main backtest loop ────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  TradingAgents Backtest: {TICKER}  {START_DATE} → {END_DATE}")
    print(f"  LLM: {config['llm_provider']} / {config['deep_think_llm']}")
    print(f"  Starting capital: ${INITIAL_CAPITAL:.2f}")
    print(f"{'='*60}\n")

    # Fetch price data upfront
    print("Fetching price data from Yahoo Finance...")
    prices = get_price_data(TICKER, START_DATE, END_DATE)
    trading_days = get_trading_days(TICKER, START_DATE, END_DATE)
    print(f"Found {len(trading_days)} trading days: {trading_days[0]} → {trading_days[-1]}\n")

    # Resume support
    completed = load_completed_days(RESULTS_CSV)
    if completed:
        print(f"Resuming — {len(completed)} days already done, skipping them.\n")

    # Load existing results for portfolio tracking
    existing_results = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            existing_results = list(csv.DictReader(f))

    portfolio = float(existing_results[-1]["portfolio"]) if existing_results else INITIAL_CAPITAL

    # Initialise the agent graph (once — expensive)
    print("Initialising TradingAgentsGraph...")
    ta = TradingAgentsGraph(debug=False, config=config)
    print("Ready.\n")

    new_results = []
    write_header = not RESULTS_CSV.exists()

    for i, date in enumerate(trading_days):
        if date in completed:
            print(f"[{i+1:02d}/{len(trading_days)}] {date} — already done, skipping.")
            continue

        print(f"[{i+1:02d}/{len(trading_days)}] {date} — running agents...", flush=True)
        t0 = time.time()

        try:
            _, raw_signal = ta.propagate(TICKER, date)
        except Exception as e:
            print(f"  ⚠ propagate() failed: {e}")
            raw_signal = "Hold"

        elapsed = time.time() - t0
        signal = raw_signal.strip() if raw_signal else "Hold"
        position = SIGNAL_TO_POSITION.get(signal, 0)

        # Score against actual next-day return
        ret = next_day_return(prices, date)
        if ret is None:
            print(f"  No next-day price data for {date}, treating as Hold.")
            position = 0
            ret = 0.0

        pnl = portfolio * position * ret
        portfolio += pnl

        row = {
            "date":      date,
            "signal":    signal,
            "position":  position,
            "ret":       round(ret, 6),
            "pnl":       round(pnl, 6),
            "portfolio": round(portfolio, 6),
            "elapsed_s": round(elapsed, 1),
        }

        append_result(RESULTS_CSV, row, write_header=write_header)
        write_header = False
        new_results.append(row)

        direction = "▲" if ret > 0 else "▼"
        correct   = "✓" if position != 0 and ((position > 0 and ret > 0) or (position < 0 and ret < 0)) else ("✗" if position != 0 else "─")
        print(f"  Signal: {signal:<12} Position: {position:+d}  Actual: {direction}{ret*100:.2f}%  P&L: {pnl:+.4f}  Portfolio: ${portfolio:.4f}  {correct}  ({elapsed:.0f}s)")

    # Final summary over all results (existing + new)
    all_results = existing_results + new_results
    print_summary(all_results)
    print(f"Full results saved to: {RESULTS_CSV}\n")


if __name__ == "__main__":
    main()
