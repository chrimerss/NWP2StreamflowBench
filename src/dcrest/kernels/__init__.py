"""Per-step process kernels.

Every kernel is a pure function of (parameters, states, forcing, ``dt_hours``)
returning (fluxes, new states). Fluxes are **mm per step**, states are **mm**
(or degC-days of cold content where relevant). No kernel holds state, allocates
persistent buffers or reads global configuration, so they compose freely and are
safe under ``torch.compile``, ``vmap`` and gradient checkpointing.

The trailing tensor dimension is the modelling unit: a basin in lumped mode, a
grid cell in distributed mode. Kernels never couple units, so they are
embarrassingly parallel; lateral coupling belongs in :mod:`dcrest.routing`.
"""
from .snow import degree_day_step
from .canopy import interception_step
from .soil import crestphys_step
from .reservoir import linear_reservoir_step

__all__ = ["degree_day_step", "interception_step", "crestphys_step",
           "linear_reservoir_step"]
