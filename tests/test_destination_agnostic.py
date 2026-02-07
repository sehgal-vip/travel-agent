"""Tests verifying destination-agnosticism — no hardcoded countries, currencies, or phrases."""

from __future__ import annotations

import os
import re

import pytest


def _get_source_files() -> list[str]:
    """Collect all .py files under src/."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    files = []
    for root, dirs, filenames in os.walk(src_dir):
        for f in filenames:
            if f.endswith(".py"):
                files.append(os.path.join(root, f))
    return files


def _read_all_source() -> str:
    """Read all source files into one string."""
    content = ""
    for path in _get_source_files():
        with open(path) as f:
            content += f.read() + "\n"
    return content


class TestNoHardcodedDestinations:
    """Ensure no country names, currencies, or languages are hardcoded in source."""

    def test_no_hardcoded_country_names_in_logic(self):
        """Country names should only appear in comments, docstrings, prompts — not in conditionals."""
        source = _read_all_source()
        # Look for country names used in if-statements or variable assignments (not in strings)
        # This is a heuristic — check for common patterns
        countries = ["Japan", "Morocco", "Thailand", "Italy", "Peru", "India"]
        for country in countries:
            # Find lines that use the country name outside of string literals and comments
            for line in source.split("\n"):
                stripped = line.strip()
                # Skip comments, docstrings, and string literals (prompts are ok)
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                # Check if it's in an if/elif/match/case statement
                if re.match(rf'\s*(if|elif)\s.*["\']?{country}["\']?', line):
                    # This would be a hardcoded destination check
                    pytest.fail(f"Hardcoded country '{country}' found in conditional: {line.strip()}")

    def test_no_hardcoded_currency_symbols_in_formatting(self):
        """Currency symbols should come from state, not be hardcoded."""
        source = _read_all_source()
        # Check for hardcoded currency formatting (e.g., f"¥{amount}" without reading from state)
        # Exclude test files, conftest, and string constants
        problematic_patterns = [
            r'f"¥\{',  # f"¥{amount}"
            r'f"€\{',  # f"€{amount}"
            r'f"₫\{',  # f"₫{amount}"
            r'f"₹\{',  # f"₹{amount}"
        ]
        for pattern in problematic_patterns:
            matches = re.findall(pattern, source)
            if matches:
                pytest.fail(f"Hardcoded currency symbol found: {pattern}")


class TestCurrencyFormatting:
    """Test that currency formatting handles various currencies correctly."""

    def test_zero_decimal_jpy(self):
        from src.tools.currency import format_local, is_zero_decimal
        assert is_zero_decimal("JPY")
        assert format_local(1500, "¥", "JPY") == "¥1,500"

    def test_zero_decimal_krw(self):
        from src.tools.currency import format_local, is_zero_decimal
        assert is_zero_decimal("KRW")
        assert format_local(50000, "₩", "KRW") == "₩50,000"

    def test_zero_decimal_vnd(self):
        from src.tools.currency import format_local, is_zero_decimal
        assert is_zero_decimal("VND")
        assert format_local(500000, "₫", "VND") == "₫500,000"

    def test_two_decimal_usd(self):
        from src.tools.currency import format_local, is_zero_decimal
        assert not is_zero_decimal("USD")
        assert format_local(13.5, "$", "USD") == "$13.50"

    def test_two_decimal_eur(self):
        from src.tools.currency import format_local, is_zero_decimal
        assert not is_zero_decimal("EUR")
        assert format_local(25.0, "€", "EUR") == "€25.00"

    def test_two_decimal_mad(self):
        from src.tools.currency import format_local
        assert format_local(150, "DH", "MAD") == "DH150.00"

    def test_dual_format_jpy(self):
        from src.tools.currency import format_dual
        result = format_dual(2000, "¥", "JPY", 148.0)
        assert "¥2,000" in result
        assert "$" in result

    def test_dual_format_mad(self):
        from src.tools.currency import format_dual
        result = format_dual(500, "DH", "MAD", 10.0)
        assert "DH500.00" in result
        assert "$50.00" in result

    def test_conversion(self):
        from src.tools.currency import convert, convert_to_local
        assert convert(14800, 148.0) == 100.0
        assert convert_to_local(100, 148.0) == 14800.0


class TestFormattersDynamic:
    """Test that formatters read from state, not hardcode."""

    def test_format_money_jpy(self):
        from src.telegram.formatters import format_money
        result = format_money(1500, "¥", "JPY")
        assert result == "¥1,500"

    def test_format_money_usd(self):
        from src.telegram.formatters import format_money
        result = format_money(13.5, "$", "USD")
        assert result == "$13.50"

    def test_format_money_none(self):
        from src.telegram.formatters import format_money
        assert format_money(None) == "N/A"

    def test_split_message_short(self):
        from src.telegram.formatters import split_message
        result = split_message("Hello world")
        assert result == ["Hello world"]

    def test_split_message_long(self):
        from src.telegram.formatters import split_message
        long_text = "A" * 5000
        result = split_message(long_text, max_length=4096)
        assert len(result) >= 2
        for part in result:
            assert len(part) <= 4096

    def test_budget_report_empty(self):
        from src.telegram.formatters import format_budget_report
        state = {"destination": {}, "cost_tracker": {}, "dates": {}, "current_trip_day": 1}
        result = format_budget_report(state)
        assert "No budget data" in result or "BUDGET" in result
