"""Dynamic currency conversion and formatting — any world currency."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# ISO 4217 zero-decimal currencies
ZERO_DECIMAL_CURRENCIES = frozenset({
    "BIF", "CLP", "DJF", "GNF", "ISK", "JPY", "KMF", "KRW",
    "PYG", "RWF", "UGX", "VND", "VUV", "XAF", "XOF", "XPF",
})


def is_zero_decimal(code: str) -> bool:
    """Check if a currency uses zero decimal places."""
    return code.upper() in ZERO_DECIMAL_CURRENCIES


def convert(amount: float, rate: float) -> float:
    """Convert a local-currency amount to USD using the given rate (local per USD)."""
    if rate <= 0:
        return 0.0
    return amount / rate


def convert_to_local(usd_amount: float, rate: float) -> float:
    """Convert USD to local currency."""
    return usd_amount * rate


def format_local(amount: float, symbol: str, code: str) -> str:
    """Format an amount in local currency with proper decimal handling."""
    if is_zero_decimal(code):
        return f"{symbol}{amount:,.0f}"
    return f"{symbol}{amount:,.2f}"


def format_usd(amount: float) -> str:
    """Format an amount in USD."""
    return f"${amount:,.2f}"


def format_dual(
    amount_local: float,
    currency_symbol: str,
    currency_code: str,
    exchange_rate: float,
) -> str:
    """Format as 'local (~$USD)' — e.g. '¥2,000 (~$13.50)'."""
    local_str = format_local(amount_local, currency_symbol, currency_code)
    usd = convert(amount_local, exchange_rate)
    return f"{local_str} (~${usd:,.2f})"


async def refresh_exchange_rate(from_code: str, to_code: str = "USD") -> float | None:
    """Fetch a fresh exchange rate from a free API.

    Returns rate as 'from_code per to_code' (e.g. JPY per USD = ~148).
    Returns None on failure.
    """
    url = f"https://open.er-api.com/v6/latest/{to_code}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            rates = data.get("rates", {})
            rate = rates.get(from_code.upper())
            if rate:
                logger.info("Exchange rate %s/%s = %s", from_code, to_code, rate)
            return rate
    except Exception:
        logger.exception("Failed to fetch exchange rate %s/%s", from_code, to_code)
        return None


async def auto_refresh_rate(state: dict) -> dict | None:
    """Check if the exchange rate is stale (>24h) and refresh if needed.

    Args:
        state: Trip state dict with 'destination' key

    Returns:
        Updated destination dict if refreshed, None if no refresh needed.
    """
    dest = state.get("destination", {})
    currency_code = dest.get("currency_code", "")
    if not currency_code or currency_code == "USD":
        return None

    # Check staleness
    researched_at = dest.get("researched_at", "")
    if researched_at:
        from datetime import datetime, timezone
        try:
            researched_dt = datetime.fromisoformat(researched_at.replace("Z", "+00:00"))
            hours_since = (datetime.now(timezone.utc) - researched_dt).total_seconds() / 3600
            if hours_since < 24:
                return None  # Rate is fresh enough
        except (ValueError, TypeError):
            pass  # Can't parse, try to refresh

    # Refresh the rate
    new_rate = await refresh_exchange_rate(currency_code, "USD")
    if new_rate is None:
        return None

    old_rate = dest.get("exchange_rate_to_usd")
    if old_rate and abs(new_rate - old_rate) / old_rate < 0.001:
        return None  # Rate unchanged (within 0.1%)

    logger.info(
        "Exchange rate refreshed: %s/USD %s -> %s",
        currency_code, old_rate, new_rate,
    )
    updated_dest = {**dest, "exchange_rate_to_usd": new_rate}
    return updated_dest
