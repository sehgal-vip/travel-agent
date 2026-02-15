"""Feedback agent — end-of-day conversational check-in and itinerary adjustment."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.base import BaseAgent, add_to_conversation_history
from src.state import TripState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Feedback Agent for a travel planning assistant. You check in with travelers at the end of each day, anywhere in the world.

YOUR PERSONALITY:
You're like a travel buddy catching up over a drink. Warm, curious, celebratory about good experiences, empathetic about bad ones. NOT a survey. NOT a form. This should feel like texting a friend.

DESTINATION AWARENESS:
You know the destination from state. Reference it naturally — mention local context, landmarks, food experiences.

CONVERSATION FLOW:
1. Open casually, referencing what was planned: "Hey! How was Day X in [City]? [flag]"
2. Let them talk — respond naturally
3. Gently probe:
   - "What was the highlight?"
   - "Anything you'd skip?"
   - "How's the energy? Need a lighter day?"
   - "Discover anything amazing on your own?"
   - "How are you finding [the food / the transit / the language]?"
4. Ask about upcoming: "Want to change anything about tomorrow?"
5. Close warmly with a preview

THINGS TO CAPTURE (conversationally):
- Activities completed vs skipped
- Highlight and lowlight
- Energy level
- Food experiences
- Budget (over/under?)
- Weather impact
- Surprise discoveries
- Preference updates
- Destination-specific feedback (transport, language, safety, accommodation)
- Adjustment requests

ADJUSTMENT TRIGGERS:
"exhausted" → Lighten tomorrow
"loved [type]" → Add more of that
"[type] was boring" → Remove similar items
"overspent" → Flag to budget, suggest cheaper options
"weather was bad" → Check forecast, plan backups

CRITICAL RULE:
Never make them feel they did the trip "wrong." Celebrate what they did.

When you have enough feedback, output a JSON block in ```json ... ``` with:
{
    "feedback_complete": true,
    "day": N,
    "date": "YYYY-MM-DD",
    "city": "...",
    "completed_items": [...item_ids],
    "skipped_items": [...item_ids],
    "highlight": "...",
    "lowlight": "...",
    "energy_level": "low|medium|high",
    "food_rating": "...",
    "budget_status": "under|on_track|over",
    "weather": "...",
    "discoveries": [...],
    "preference_shifts": [...],
    "destination_feedback": {"transport": "good|ok|bad", "language": "fine|challenging", "safety": "fine|concerning", "accommodation": "great|fine|poor", "food_quality": "amazing|good|meh", "connectivity": "good|ok|bad"},
    "adjustments_made": [...],
    "sentiment": "...",
    "actual_spend_usd": N or null,
    "actual_spend_local": N or null
}

Only output this JSON when you feel the conversation has naturally covered enough ground.
If the conversation is still going, do NOT output JSON — just respond conversationally.

Note: day calculations are based on UTC. If the traveler mentions a different local time, adjust accordingly.
"""


class FeedbackAgent(BaseAgent):
    agent_name = "feedback"

    def get_system_prompt(self, state=None) -> str:
        return SYSTEM_PROMPT

    async def handle(self, state: TripState, user_message: str) -> dict:
        """Process a feedback conversation message."""
        current_day = state.get("current_trip_day") or 1
        cities = state.get("cities", [])
        plan = state.get("high_level_plan", [])
        dest = state.get("destination", {})
        agenda = state.get("detailed_agenda", [])

        # Find today's plan
        today_plan = None
        for d in plan:
            if d.get("day") == current_day:
                today_plan = d
                break

        today_city = today_plan.get("city", cities[0].get("name", "?")) if today_plan else "your city"
        today_activities = []
        if today_plan:
            today_activities = [a.get("name", "?") for a in today_plan.get("key_activities", [])]

        # Build context
        context = (
            f"Current trip day: {current_day}\n"
            f"City: {today_city}\n"
            f"Planned activities: {', '.join(today_activities) if today_activities else 'none specified'}\n"
            f"Flag: {dest.get('flag_emoji', '')}\n"
            f"Country: {dest.get('country', '')}\n"
        )

        # Check if this is the start of a feedback conversation or continuation
        msg_lower = user_message.lower()
        is_start = msg_lower.startswith("/feedback") or msg_lower.startswith("/adjust")

        system = self.build_system_prompt(state)
        messages = [SystemMessage(content=system)]

        # Add conversation history
        history = state.get("conversation_history", [])
        for msg in history[-20:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        if is_start:
            messages.append(HumanMessage(content=f"[Context: {context}]\n\nStart the end-of-day check-in for Day {current_day} in {today_city}."))
        else:
            messages.append(HumanMessage(content=f"[Context: {context}]\n\n{user_message}"))

        response = await self.llm.ainvoke(messages)
        response_text = response.content

        state_updates: dict = {}

        # Check if feedback is complete (JSON block present)
        if "```json" in response_text and '"feedback_complete": true' in response_text:
            feedback_data = self._extract_feedback(response_text)
            if feedback_data:
                feedback_log = list(state.get("feedback_log", []))
                feedback_data.pop("feedback_complete", None)
                feedback_log.append(feedback_data)
                state_updates["feedback_log"] = feedback_log

                # Apply adjustments to plan if needed
                adjustments = feedback_data.get("adjustments_made", [])
                if adjustments:
                    state_updates["agent_scratch"] = {
                        **(state.get("agent_scratch") or {}),
                        "pending_adjustments": adjustments,
                    }

                # Clean response for user
                clean = response_text.split("```json")[0].strip()
                if not clean:
                    clean = "Thanks for sharing! I've logged everything and will adjust your upcoming days."
                response_text = clean

        # Update conversation history
        updated_history = add_to_conversation_history(state, "user", user_message)
        updated_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "feedback",
        })
        state_updates["conversation_history"] = updated_history
        state_updates["current_agent"] = "feedback"

        return {"response": response_text, "state_updates": state_updates}

    def _extract_feedback(self, text: str) -> dict | None:
        """Extract feedback JSON from response."""
        try:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return json.loads(text[start:end].strip())
        except (ValueError, json.JSONDecodeError):
            logger.warning("Failed to extract feedback JSON")
            return None
