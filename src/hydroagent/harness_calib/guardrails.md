- Use only the files in this workspace. Do not search for, download or request
  streamflow, gauge identities or any other data from the internet; the targets are
  deliberately anonymous.
- Treat `task_folder/` as read-only.
- Never use target-side information you do not have (there is no target discharge);
  every modelling choice must be justified by donor-side evidence.
- Write a first complete, valid `output/target_discharge.csv` before any long
  experiment, then overwrite it when you have something better.
- Run calibrations in the background with a log and keep working while they run; never
  let a single blocking command exceed ten minutes.
- When choosing how to give parameters to catchments that were not calibrated, load the `regionalization` skill.
