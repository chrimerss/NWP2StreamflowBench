"""NOAA/NCEI U.S. climate regions (Karl & Koss 1984), keyed by state postal code."""
from __future__ import annotations

CLIMATE_REGIONS = {
    "Northeast": ["CT", "DE", "ME", "MD", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"],
    "Upper Midwest": ["IA", "MI", "MN", "WI"],
    "Ohio Valley": ["IL", "IN", "KY", "MO", "OH", "TN", "WV"],
    "Southeast": ["AL", "FL", "GA", "NC", "SC", "VA"],
    "Northern Rockies and Plains": ["MT", "NE", "ND", "SD", "WY"],
    "South": ["AR", "KS", "LA", "MS", "OK", "TX"],
    "Southwest": ["AZ", "CO", "NM", "UT"],
    "Northwest": ["ID", "OR", "WA"],
    "West": ["CA", "NV"],
}
REGION_ORDER = list(CLIMATE_REGIONS)
STATE_TO_REGION = {s: r for r, states in CLIMATE_REGIONS.items() for s in states}


def region_of_state(state: str) -> str:
    return STATE_TO_REGION.get(str(state).upper(), "Other")
