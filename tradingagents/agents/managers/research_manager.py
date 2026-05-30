"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")
    _disabled: list = []

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        # Static system message + dynamic user message → higher cache hit rate.
        system_content = (
            "You are the Research Manager and debate facilitator. Critically evaluate the debate and deliver a clear, actionable investment plan for the trader.\n\n"
            "**Rating Scale** (use exactly one):\n"
            "- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position\n"
            "- **Overweight**: Constructive view; recommend gradually increasing exposure\n"
            "- **Hold**: Balanced view; recommend maintaining the current position\n"
            "- **Underweight**: Cautious view; recommend trimming exposure\n"
            "- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position\n\n"
            "Commit to a clear stance whenever the debate's strongest arguments warrant one; "
            "reserve Hold for situations where the evidence on both sides is genuinely balanced."
            + get_language_instruction()
        )
        user_content = (
            f"{instrument_context}\n\n"
            f"**Debate History:**\n{history}"
        )
        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
            _disabled=_disabled,
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
