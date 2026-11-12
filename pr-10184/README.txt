PR #10184 performance charts

Source: https://github.com/InferMatrixLab/InferRSI/pull/394
Evidence snapshot: e4773063f1c16fdbd5897b772be12262fe5035d5

Both charts are deterministic plots of the published evidence, not new benchmark runs.
69 median points and 207 raw timing samples are in curve-data.csv.
The immutable per-point source URLs are included in the CSV.
Error bars show observed min/max, not confidence intervals.
Historical revisions and independent launch pairs are shown separately.
Graph on/off also changed internal-format settings; these are coupled configuration observations.
All three latency targets remain unmet. Final ABBA and full visual acceptance remain incomplete.

Rebuild: install matplotlib and numpy, then run python plot_curves.py.
The source directory contains JSON/Markdown evidence; original model images are not bundled.
AI-assisted plotting and documentation: Codex.
