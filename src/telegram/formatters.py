"""Telegram message formatters — all destination/currency adaptive."""

from __future__ import annotations

from src.tools.currency import format_dual, format_local, format_usd, is_zero_decimal


def format_money(amount: float | None, symbol: str = "$", code: str = "USD") -> str:
    """Format money with proper decimals for any currency."""
    if amount is None:
        return "N/A"
    return format_local(amount, symbol, code)


def format_dual_currency(
    amount_local: float | None,
    symbol: str,
    code: str,
    rate: float,
) -> str:
    """Format as 'local (~$USD)'."""
    if amount_local is None:
        return "N/A"
    return format_dual(amount_local, symbol, code, rate)


def format_status_dashboard(state: dict) -> str:
    """Build the /status dashboard from state."""
    from src.agents.orchestrator import generate_status
    return generate_status(state)


def format_day_plan(day: dict, dest: dict) -> str:
    """Format a high-level DayPlan for Telegram."""
    symbol = dest.get("currency_symbol", "$")
    code = dest.get("currency_code", "USD")

    lines = [
        f"**Day {day.get('day', '?')}** — {day.get('city', '?')} — \"{day.get('theme', '')}\"",
        f"Vibe: {day.get('vibe', 'mixed')}",
    ]

    travel = day.get("travel")
    if travel:
        lines.append(
            f"🚆 Travel: {travel.get('mode', '?')} from {travel.get('from_city', '?')} "
            f"to {travel.get('to_city', '?')} ({travel.get('duration_hrs', '?')}h)"
        )

    activities = day.get("key_activities", [])
    if activities:
        lines.append("Key Activities:")
        for act in activities:
            tier_icon = {"must_do": "🔴", "nice_to_have": "🟡", "if_nearby": "🟢"}.get(
                act.get("tier", ""), "⚪"
            )
            lines.append(f"  {tier_icon} {act.get('name', '?')}")

    meals = day.get("meals", {})
    if meals:
        lines.append("Meals:")
        for meal_type in ("breakfast", "lunch", "dinner"):
            slot = meals.get(meal_type)
            if slot:
                name = slot.get("name") or slot.get("type", "explore")
                icon = {"breakfast": "🌅", "lunch": "☀️", "dinner": "🌙"}.get(meal_type, "🍽️")
                lines.append(f"  {icon} {meal_type.title()}: {name}")

    moment = day.get("special_moment")
    if moment:
        lines.append(f"✨ Special: {moment}")

    cost = day.get("estimated_cost_usd")
    if cost:
        lines.append(f"💰 ~{format_usd(cost)}")

    return "\n".join(lines)


def format_agenda_slot(slot: dict, dest: dict) -> str:
    """Format a single AgendaSlot for Telegram."""
    symbol = dest.get("currency_symbol", "$")
    code = dest.get("currency_code", "USD")
    rate = dest.get("exchange_rate_to_usd", 1)

    lines = [f"⏰ {slot.get('time', '?')} — {slot.get('name', '?')}"]

    address = slot.get("address")
    if address:
        lines.append(f"   📍 {address}")

    transport = slot.get("transport_from_prev")
    if transport:
        mode = transport.get("mode", "?")
        duration = transport.get("duration_min", "?")
        cost = transport.get("cost_local")
        transport_str = f"   🚶 {mode} — {duration}min"
        if cost is not None:
            transport_str += f" — {format_local(cost, symbol, code)}"
        lines.append(transport_str)

    lines.append(f"   ⏱️ {slot.get('duration_min', '?')}min")

    cost_local = slot.get("cost_local")
    if cost_local is not None:
        lines.append(f"   💰 {format_dual(cost_local, symbol, code, rate)}")

    tips = slot.get("tips")
    if tips:
        lines.append(f"   📝 {tips}")

    phrase = slot.get("local_phrase")
    if phrase:
        lines.append(f"   💬 {phrase}")

    payment = slot.get("payment")
    if payment:
        lines.append(f"   💳 {payment}")

    booking = slot.get("booking_urgency")
    if booking:
        urgency_icon = {"book_now": "🔴", "book_today": "🟡", "walk_in": "🟢"}.get(booking, "")
        legend = {"book_now": "reserve today", "book_today": "reserve this week", "walk_in": "no reservation needed"}.get(booking, "")
        lines.append(f"   {urgency_icon} {booking}" + (f" ({legend})" if legend else ""))

    rain = slot.get("rain_backup")
    if rain:
        lines.append(f"   🌧️ Backup: {rain}")

    return "\n".join(lines)


def format_budget_report(state: dict) -> str:
    """Format the full budget report for Telegram."""
    dest = state.get("destination", {})
    tracker = state.get("cost_tracker", {})
    if not tracker:
        return "No budget data yet. Costs will be tracked once your trip is planned."

    symbol = dest.get("currency_symbol", "$")
    code = dest.get("currency_code", "USD")
    totals = tracker.get("totals", {})
    budget = tracker.get("budget_total_usd", 0)
    spent = totals.get("spent_usd", 0)
    remaining = totals.get("remaining_usd", 0)
    daily_avg = totals.get("daily_avg_usd", 0)
    projected = totals.get("projected_total_usd", 0)
    status = totals.get("status", "on_track")

    pct = int((spent / budget * 100) if budget > 0 else 0)
    bar_filled = pct // 5
    bar_empty = 20 - bar_filled
    progress_bar = "█" * bar_filled + "░" * bar_empty

    status_emoji = {"under_budget": "✅", "on_track": "📊", "over_budget": "⚠️"}.get(status, "❓")

    dates = state.get("dates", {})
    current_day = state.get("current_trip_day", 1)
    total_days = dates.get("total_days", "?")

    lines = [
        f"💰 BUDGET REPORT — Day {current_day} of {total_days}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "📊 OVERALL",
        f"   Budget:    {format_usd(budget)}",
        f"   Spent:     {format_usd(spent)} ({pct}%)",
        f"   Remaining: {format_usd(remaining)}",
        f"   Daily avg: {format_usd(daily_avg)}",
        f"   On track?  {status_emoji}",
        "",
        f"   [{progress_bar}] {pct}%",
    ]

    by_category = tracker.get("by_category", {})
    if by_category:
        lines.extend(["", "📂 BY CATEGORY"])
        category_icons = {
            "accommodation": "🏨", "food": "🍽️", "activities": "🎟️",
            "transport": "🚗", "shopping": "🛍️", "wellness": "💆", "misc": "📱",
        }
        for cat, data in by_category.items():
            icon = category_icons.get(cat, "•")
            cat_spent = data.get("spent_usd", 0)
            lines.append(f"   {icon} {cat.title()}: {format_usd(cat_spent)}")

    by_city = tracker.get("by_city", {})
    if by_city:
        lines.extend(["", "📅 BY CITY"])
        for city, data in by_city.items():
            city_spent = data.get("spent_usd", 0)
            lines.append(f"   {city}: {format_usd(city_spent)}")

    tips = tracker.get("savings_tips", [])
    if tips:
        lines.extend(["", "💡 TIPS:"])
        for tip in tips[:5]:
            lines.append(f"   • {tip}")

    if projected > 0:
        over_under = int(((projected - budget) / budget * 100) if budget > 0 else 0)
        sign = "+" if over_under >= 0 else ""
        lines.extend(["", f"📈 PROJECTION: {format_usd(projected)} total ({sign}{over_under}%)"])

    return "\n".join(lines)


def split_message(text: str, max_length: int = 4096) -> list[str]:
    """Split a long message at paragraph boundaries to fit Telegram's limit."""
    if len(text) <= max_length:
        return [text]

    parts: list[str] = []
    current = ""

    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) + 2 > max_length:
            if current:
                parts.append(current.strip())
                current = ""
            # If a single paragraph exceeds limit, split at newlines or hard-split
            if len(paragraph) > max_length:
                for line in paragraph.split("\n"):
                    # Hard-split individual lines that exceed the limit
                    while len(line) > max_length:
                        chunk = line[:max_length]
                        if current:
                            parts.append(current.strip())
                            current = ""
                        parts.append(chunk)
                        line = line[max_length:]
                    if len(current) + len(line) + 1 > max_length:
                        if current:
                            parts.append(current.strip())
                        current = line
                    else:
                        current = current + "\n" + line if current else line
            else:
                current = paragraph
        else:
            current = current + "\n\n" + paragraph if current else paragraph

    if current.strip():
        parts.append(current.strip())

    if not parts:
        # No paragraph boundaries — hard-split at max_length
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    return parts
