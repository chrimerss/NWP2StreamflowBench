# Tools

`python3 scripts/hydrokit.py <command>` wraps the common dCREST operations:

- `info` - data inventory, windows, parameter names and bounds.
- `simulate --set donors|targets --params P.csv --out Q.csv [--ids ...] [--start --end]`
  - run dCREST with a parameter CSV (`id` + 16 parameters, optional `area_km2`).
- `evaluate --params P.csv [--ids D0001,...] [--out per_basin.csv]` - donor NSE/KGE over
  the observation window.
- `calibrate --ids D0001,... --out P.csv [--iters 400] [--loss nse|kge] [--init released|default]`
  - gradient-descent calibration of donors over the observation window. One iteration takes
    ~3-4 s whatever the number of basins (the cost is the 12-year daily recursion), so
    calibrate all the donors you need in ONE call and keep `--iters` modest (100-300).

For anything else write Python that imports `dcrest` from the workspace root
(`dcrest.simulate(P, T, PET, params, config)` with `(days, basins)` tensors). Use
`scripts/hydrokit.py` as a reference for loading data and building the model config.
