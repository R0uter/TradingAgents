"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.earnings_calendar import fetch_earnings_calendar


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")
    _disabled: list = []

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        trade_date = state.get("trade_date", "")
        asset_type = state.get("asset_type", "stock")
        company_name = state["company_of_interest"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        # Earnings risk override — hard constraint for PM
        earnings_override = ""
        if trade_date and asset_type == "stock":
            earnings_info = fetch_earnings_calendar(company_name, trade_date)
            if "HIGH-RISK BINARY EVENT" in earnings_info:
                earnings_override = (
                    f"\n\n⚠️ MANDATORY EARNINGS OVERRIDE ⚠️\n{earnings_info}\n"
                    "ABSOLUTE RULE: Earnings within 3 days = HOLD. No exceptions. "
                    "A ±10-25% gap makes any directional bet irrational. "
                    "Your rating MUST be Hold when this warning is active.\n\n"
                )
            elif "MODERATE" in earnings_info:
                earnings_override = (
                    f"\n\nEarnings Note: {earnings_info}\n"
                    "Consider reducing conviction (lean toward Hold/Underweight).\n\n"
                )

        # Static system message + dynamic user message → higher cache hit rate.
        system_content = (
            "You are the Portfolio Manager. Synthesize the risk analysts' debate and deliver the final trading decision.\n\n"
            "**Rating Scale** (use exactly one):\n"
            "- **Buy**: Strong conviction to enter or add to position\n"
            "- **Overweight**: Favorable outlook, gradually increase exposure\n"
            "- **Hold**: Maintain current position, no action needed\n"
            "- **Underweight**: Reduce exposure, take partial profits\n"
            "- **Sell**: Exit position or avoid entry\n\n"
            "Be decisive and ground every conclusion in specific evidence from the analysts."
            + get_language_instruction()
        )
        user_content = (
            f"{instrument_context}\n\n"
            f"{earnings_override}"
            f"**Context:**\n"
            f"- Research Manager's investment plan: **{research_plan}**\n"
            f"- Trader's transaction proposal: **{trader_plan}**\n"
            f"{lessons_line}"
            f"**Risk Analysts Debate History:**\n{history}"
        )
        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
            _disabled=_disabled,
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
