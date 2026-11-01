"""Static UK bank holiday schedule covering 2024 through 2027.

Source: Official UK Government bank holiday register for England and Wales:
https://www.gov.uk/bank-holidays.json (retrieved November 2026).
Committed as static data to avoid external runtime network calls or unpinned dependencies.
"""

from __future__ import annotations

from datetime import date

# Official England & Wales bank holidays covering 2024 through 2027
UK_BANK_HOLIDAYS: frozenset[date] = frozenset({
    # 2024
    date(2024, 1, 1),   # New Year's Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 4, 1),   # Easter Monday
    date(2024, 5, 6),   # Early May bank holiday
    date(2024, 5, 27),  # Spring bank holiday
    date(2024, 8, 26),  # Summer bank holiday
    date(2024, 12, 25), # Christmas Day
    date(2024, 12, 26), # Boxing Day

    # 2025
    date(2025, 1, 1),   # New Year's Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 4, 21),  # Easter Monday
    date(2025, 5, 5),   # Early May bank holiday
    date(2025, 5, 26),  # Spring bank holiday
    date(2025, 8, 25),  # Summer bank holiday
    date(2025, 12, 25), # Christmas Day
    date(2025, 12, 26), # Boxing Day

    # 2026
    date(2026, 1, 1),   # New Year's Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 4, 6),   # Easter Monday
    date(2026, 5, 4),   # Early May bank holiday
    date(2026, 5, 25),  # Spring bank holiday
    date(2026, 8, 31),  # Summer bank holiday
    date(2026, 12, 25), # Christmas Day
    date(2026, 12, 28), # Boxing Day (substitute day)

    # 2027
    date(2027, 1, 1),   # New Year's Day
    date(2027, 3, 26),  # Good Friday
    date(2027, 3, 29),  # Easter Monday
    date(2027, 5, 3),   # Early May bank holiday
    date(2027, 5, 31),  # Spring bank holiday
    date(2027, 8, 30),  # Summer bank holiday
    date(2027, 12, 27), # Christmas Day (substitute day)
    date(2027, 12, 28), # Boxing Day (substitute day)
})


def is_uk_bank_holiday(d: date) -> bool:
    """Return True if calendar date d is an official UK bank holiday."""
    return d in UK_BANK_HOLIDAYS
