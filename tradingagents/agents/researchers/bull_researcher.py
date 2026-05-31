from tradingagents.agents.utils.agent_utils import get_language_instruction


# Static system message — identical across all tickers/dates → cacheable prefix.
_BULL_SYSTEM = (
    "You are a Bull Analyst advocating for investing. Your task is to build a strong, "
    "evidence-based case emphasizing growth potential, competitive advantages, and positive "
    "market indicators. Leverage the provided research and data to address concerns and "
    "counter bearish arguments effectively.\n\n"
    "Key points to focus on:\n"
    "- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.\n"
    "- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.\n"
    "- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.\n"
    "- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, "
    "addressing concerns thoroughly and showing why the bull perspective holds stronger merit.\n"
    "- Engagement: Present your argument in a conversational style, engaging directly with the bear "
    "analyst's points and debating effectively rather than just listing data.\n\n"
    "Use the resources provided in the user message to deliver a compelling bull argument, refute the "
    "bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position."
)


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )

        lang = get_language_instruction()
        lang_line = f"\n{lang}" if lang else ""

        # Stable block: reports are identical across all debate agents for the
        # same ticker/date. Mark with cache_control so Anthropic caches this
        # large block and all 5 debate agents (bull, bear, aggressive,
        # conservative, neutral) get cache hits on it.
        reports_block = (
            f"Argue the bull case for this {target_label}.\n\n"
            f"Market research report: {market_research_report}\n\n"
            f"Social media sentiment report: {sentiment_report}\n\n"
            f"Latest world affairs news: {news_report}\n\n"
            f"{fundamentals_label}: {fundamentals_report}"
            f"{lang_line}"
        )

        # Dynamic block: grows each round — never cached.
        dynamic_block = (
            f"\n\nConversation history of the debate: {history}\n\n"
            f"Last bear argument: {current_response}"
        )

        messages = [
            {"role": "system", "content": _BULL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": reports_block, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": dynamic_block},
                ],
            },
        ]

        response = llm.invoke(messages)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
