"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  1. News headlines     — Yahoo Finance (institutional framing)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

The agent does not use tool-calling; the data is in the prompt from
turn 0. The LLM produces the sentiment report in a single invocation.

See: https://github.com/TauricResearch/TradingAgents/issues/557
"""

from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.earnings_calendar import fetch_earnings_calendar


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


# Static system message — no ticker, dates, or live data.
# Kept constant so the prefix is byte-identical across all calls → cache hit.
_STATIC_SENTIMENT_SYSTEM = (
    "You are a helpful AI assistant, collaborating with other assistants."
    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop.\n"
    "You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report"
    " for the given ticker and date range, drawing on four complementary data sources provided in the user message.\n\n"
    "## How to analyze this data (best practices)\n\n"
    "1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.**"
    " A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk;"
    " 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.\n\n"
    "2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish,"
    " that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to"
    " (or vice versa, that retail is chasing while institutions are cautious).\n\n"
    "3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention;"
    " a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.\n\n"
    "4. **Distinguish opinion from event.** A news headline is an event; a StockTwits post is opinion."
    " Both are inputs but should be weighted differently in your conclusions.\n\n"
    "5. **Identify recurring narrative themes.** What topic keeps coming up across sources?"
    " That's the dominant narrative driving current sentiment.\n\n"
    "6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources"
    " returned an \"<unavailable>\" placeholder, the sentiment read is less robust — flag this caveat explicitly.\n\n"
    "7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches,"
    " competitive threats, macro headlines, etc.\n\n"
    "8. **Earnings calendar is the highest-priority risk signal.** If the earnings calendar shows a report within"
    " 3 days, this overrides all other sentiment signals. Flag imminent earnings as the #1 risk factor and recommend"
    " reduced confidence or a HOLD stance.\n\n"
    "9. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside"
    " fundamentals and technicals, not as a price call.\n\n"
    "## Output\n\n"
    "Produce a sentiment report covering, in order:\n\n"
    "1. **Overall sentiment direction** — Bullish / Bearish / Neutral / Mixed — with a brief confidence note.\n"
    "2. **Source-by-source breakdown** — what each of news / StockTwits / Reddit is telling you, with specific evidence.\n"
    "3. **Divergences, alignments, and key narratives** across sources.\n"
    "4. **Catalysts and risks** surfaced by the data.\n"
    "5. **Markdown table** at the end summarizing key sentiment signals, their direction, source, and supporting evidence."
)


def _build_data_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    instrument_context: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    earnings_block: str,
) -> str:
    """Build the human message containing all dynamic/fetched data."""
    lang = get_language_instruction()
    lang_line = f"\n{lang}" if lang else ""
    return f"""Produce a sentiment report for {ticker} covering {start_date} to {end_date}. {instrument_context}

## Data sources (pre-fetched)

### Earnings Calendar — upcoming/recent earnings dates
CRITICAL: Earnings reports are unpredictable binary events that can move a stock ±10-25% in a single session. If earnings are imminent, this MUST be prominently flagged as the dominant risk factor regardless of other sentiment signals.

<start_of_earnings_calendar>
{earnings_block}
<end_of_earnings_calendar>

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion. Engagement signal via upvote score and comment count.

<start_of_reddit>
{reddit_block}
<end_of_reddit>{lang_line}"""


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a sentiment report in a
    single LLM call.
    """

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = build_instrument_context(ticker)

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        from tradingagents.dataflows.config import get_config
        _cfg = get_config()

        news_block = get_news.func(ticker, start_date, end_date)
        if _cfg.get("disable_stocktwits"):
            stocktwits_block = "<StockTwits disabled for backtest — realtime data not applicable>"
        else:
            stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
        if _cfg.get("disable_reddit"):
            reddit_block = "<Reddit disabled for backtest — realtime data not applicable>"
        else:
            reddit_block = fetch_reddit_posts(ticker)
        earnings_block = fetch_earnings_calendar(ticker, end_date)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    _STATIC_SENTIMENT_SYSTEM,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Dynamic context (ticker, dates, all fetched data) goes into the first
        # human message so the static system message prefix is byte-identical
        # across all tickers/dates → higher cache hit rate on DeepSeek/Anthropic.
        data_msg = HumanMessage(content=_build_data_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            instrument_context=instrument_context,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
            earnings_block=earnings_block,
        ))

        # No bind_tools — the data is already in the prompt; a single LLM
        # call produces the report directly.
        chain = prompt | llm
        result = chain.invoke([data_msg] + list(state["messages"]))

        return {
            "messages": [result],
            "sentiment_report": result.content,
        }

    return sentiment_analyst_node


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
