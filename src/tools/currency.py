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
