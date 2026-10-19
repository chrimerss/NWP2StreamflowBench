---
name: regionalization
description: giving parameters to catchments that were not calibrated
---

# Parameter regionalization for ungauged catchments

Common families, each to be tested with donor-side cross-validation before use:

- **Spatial proximity** - parameters of the nearest donor(s) by centroid distance.
- **Physical similarity** - donors closest in a standardized space of climate, terrain,
  soil and vegetation attributes; choose the attribute set by cross-validation.
- **Output averaging** - run several donors' parameter sets on the target and average
  the simulated discharge; usually more robust than a single donor.
- **Regression / regional calibration** - relate parameters to attributes, or calibrate
  an attribute-to-parameter mapping jointly across catchments by gradient descent through
  dCREST (the model is differentiable end to end).

Practical points: always swap in the target's area; weight donors by similarity rather
than equally when evidence supports it; check performance by region and aridity class,
because dry and snow-dominated catchments behave differently.
