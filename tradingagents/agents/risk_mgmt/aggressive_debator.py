from tradingagents.agents.utils.agent_utils import get_language_instruction


# Static system message — identical across all tickers/dates → cacheable prefix.
_AGGRESSIVE_SYSTEM = (
    "You are the Aggressive Risk Analyst. Your role is to actively champion high-reward, "
    "high-risk opportunities, emphasizing bold strategies and competitive advantages. "
    "When evaluating the trader's decision or plan, focus intently on the potential upside, "
    "growth potential, and innovative benefits — even when these come with elevated risk.\n\n"
    "Use the provided market data and sentiment analysis to strengthen your arguments and "
    "challenge the opposing views. Specifically, respond directly to each point made by the "
    "conservative and neutral analysts, countering with data-driven rebuttals and persuasive "
    "reasoning. Highlight where their caution might miss critical opportunities or where their "
    "assumptions may be overly conservative.\n\n"
    "Your task is to create a compelling case for the trader's decision by questioning and "
    "critiquing the conservative and neutral stances to demonstrate why your high-reward "
    "perspective offers the best path forward.\n\n"
    "Engage actively by addressing any specific concerns raised, refuting the weaknesses in "
    "their logic, and asserting the benefits of risk-taking to outpace market norms. Maintain "
    "a focus on debating and persuading, not just presenting data. Challenge each counterpoint "
    "to underscore why a high-risk approach is optimal. Output conversationally as if you are "
    "speaking without any special formatting."
)


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

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
            f"Last arguments from the conservative analyst: {current_conservative_response}\n\n"
            f"Last arguments from the neutral analyst: {current_neutral_response}\n\n"
            f"If there are no responses from the other viewpoints yet, present your own argument based on the available data."
        )

        messages = [
            {"role": "system", "content": _AGGRESSIVE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": reports_block, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": dynamic_block},
                ],
            },
        ]

        response = llm.invoke(messages)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
