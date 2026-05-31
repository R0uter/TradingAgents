from tradingagents.agents.utils.agent_utils import get_language_instruction


# Static system message — identical across all tickers/dates → cacheable prefix.
_NEUTRAL_SYSTEM = (
    "You are the Neutral Risk Analyst. Your role is to provide a balanced perspective, "
    "weighing both the potential benefits and risks of the trader's decision or plan. "
    "You prioritize a well-rounded approach, evaluating the upsides and downsides while "
    "factoring in broader market trends, potential economic shifts, and diversification "
    "strategies.\n\n"
    "Your task is to challenge both the Aggressive and Conservative Analysts, pointing out "
    "where each perspective may be overly optimistic or overly cautious. Use insights from "
    "the provided data sources to support a moderate, sustainable strategy to adjust the "
    "trader's decision.\n\n"
    "Engage actively by analyzing both sides critically, addressing weaknesses in the "
    "aggressive and conservative arguments to advocate for a more balanced approach. "
    "Challenge each of their points to illustrate why a moderate risk strategy might offer "
    "the best of both worlds, providing growth potential while safeguarding against extreme "
    "volatility. Focus on debating rather than simply presenting data, aiming to show that "
    "a balanced view can lead to the most reliable outcomes. Output conversationally as if "
    "you are speaking without any special formatting."
)


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        lang = get_language_instruction()
        lang_line = f"\n{lang}" if lang else ""

        # Stable block: trader_decision + reports identical across all 3 risk
        # debators for the same ticker/date → Anthropic caches this large block.
        reports_block = (
            f"Trader's decision:\n{trader_decision}\n\n"
            f"Market Research Report: {market_research_report}\n\n"
            f"Social Media Sentiment Report: {sentiment_report}\n\n"
            f"Latest World Affairs Report: {news_report}\n\n"
            f"Company Fundamentals Report: {fundamentals_report}"
            f"{lang_line}"
        )

        # Dynamic block: grows each round — never cached.
        dynamic_block = (
            f"\n\nCurrent conversation history: {history}\n\n"
            f"Last arguments from the aggressive analyst: {current_aggressive_response}\n\n"
            f"Last arguments from the conservative analyst: {current_conservative_response}\n\n"
            f"If there are no responses from the other viewpoints yet, present your own argument based on the available data."
        )

        messages = [
            {"role": "system", "content": _NEUTRAL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": reports_block, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": dynamic_block},
                ],
            },
        ]

        response = llm.invoke(messages)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
