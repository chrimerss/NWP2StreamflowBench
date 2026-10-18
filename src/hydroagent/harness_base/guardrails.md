- Use only the files in this workspace. Do not search for, download or request
  streamflow, gauge identities or any other data from the internet; the targets are
  deliberately anonymous.
- Treat `task_folder/` as read-only.
- Never use target-side information you do not have (there is no target discharge);
  every modelling choice must be justified by donor-side evidence.
- Write a first complete, valid `output/target_discharge.csv` before any long
  experiment, then overwrite it when you have something better.
- Keep long computations within a few minutes each: subsample donors for exploratory
  cross-validation and calibrate in batches.
- When choosing or comparing parameter-transfer methods for ungauged catchments, load the `regionalization` skill.
