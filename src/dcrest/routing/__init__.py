"""Routing: translation of generated runoff to the outlet.

Lumped mode uses a per-unit Gamma unit hydrograph whose scale is anchored to a
geomorphic estimate from drainage area. Cell-to-cell channel routing (the
"coupled routing" half of CREST) is not implemented here yet; :mod:`dcrest`
is currently a lumped model and says so.
"""
from .uh import gamma_uh, geomorphic_lag_hours, convolve_uh

__all__ = ["gamma_uh", "geomorphic_lag_hours", "convolve_uh"]
