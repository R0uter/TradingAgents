import time
import logging

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has expected column names regardless of yfinance version quirks."""
    data.columns = [c.strip() for c in data.columns]
    # yfinance versions may produce 'date', 'Datetime', 'Date', or 'index' after reset_index()
    lower_map = {c.lower(): c for c in data.columns}
    if "Date" not in data.columns:
        if "date" in lower_map:
            data = data.rename(columns={lower_map["date"]: "Date"})
        elif "datetime" in lower_map:
            data = data.rename(columns={lower_map["datetime"]: "Date"})
        elif "index" in data.columns:
            # reset_index() on a nameless DatetimeIndex produces 'index'
            data = data.rename(columns={"index": "Date"})
    # Normalize OHLCV columns (some yfinance versions use lowercase)
    canonical = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    rename_map = {}
    for lower_name, proper_name in canonical.items():
        if proper_name not in data.columns and lower_name in lower_map:
            rename_map[lower_map[lower_name]] = proper_name
    if rename_map:
        data = data.rename(columns=rename_map)
    return data


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    # If Date is the index (e.g. CSV saved with index, or yfinance DatetimeIndex), promote it
    if data.index.name in ("Date", "Datetime", "date", "datetime") or (
        hasattr(data.index, "dtype") and str(data.index.dtype).startswith("datetime")
    ):
        data = data.reset_index()
    data = _normalize_columns(data)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    # Strip timezone so tz-aware yfinance dates compare cleanly with naive Timestamps
    if not data.empty and hasattr(data["Date"], "dt") and data["Date"].dt.tz is not None:
        data["Date"] = data["Date"].dt.tz_localize(None)
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 15 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.
    """
    # Reject ticker values that would escape the cache directory when
    # interpolated into the cache filename (e.g. ``../../tmp/x``).
    safe_symbol = safe_ticker_component(symbol)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    # Cache uses a fixed window (15y to today) so one file per symbol
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
    else:
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        data = data.reset_index()
        data.to_csv(data_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)

    # Filter to curr_date to prevent look-ahead bias in backtesting
    data = data[data["Date"] <= curr_date_dt]

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        # stockstats.wrap() requires the date as the DataFrame index
        if "Date" in data.columns:
            data = data.set_index("Date")
        df = wrap(data)
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator

        # Use index directly — StockDataFrame.__getitem__ intercepts "Date" as an
        # indicator name and raises KeyError; the index is always safe.
        date_strs = df.index.strftime("%Y-%m-%d")
        mask = date_strs == curr_date_str
        matching_rows = df[mask]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
