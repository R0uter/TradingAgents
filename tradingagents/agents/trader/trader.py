"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.earnings_calendar import fetch_earnings_calendar


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")
    _disabled: list = []

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(company_name, asset_type)
        investment_plan = state["investment_plan"]
        trade_date = state.get("trade_date", "")

        # Inject earnings calendar — hard constraint for the trader
        earnings_block = ""
        if trade_date and asset_type == "stock":
            earnings_info = fetch_earnings_calendar(company_name, trade_date)
            if "HIGH-RISK BINARY EVENT" in earnings_info:
                earnings_block = (
                    f"\n\n⚠️ MANDATORY RISK OVERRIDE ⚠️\n{earnings_info}\n"
                    "RULE: When earnings are within 3 days, you MUST recommend HOLD "
                    "regardless of other signals. The risk of a ±10-25% gap makes any "
                    "directional bet a coin flip. Only override this if the investment "
                    "plan explicitly acknowledges the earnings risk and provides "
                    "overwhelming evidence from 3+ independent sources.\n"
                )
            elif "MODERATE" in earnings_info:
                earnings_block = f"\n\nEarnings Note: {earnings_info}\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
                    f"{earnings_block}"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
            _disabled=_disabled,
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
