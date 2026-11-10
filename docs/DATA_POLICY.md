# Data policy

This repository publishes **code, tests and synthetic demo data only**.

- Do not commit `.f1cache/`, `.fastf1_cache/`, `.openf1_cache/` or `web_data/` (they are in `.gitignore`).
  They hold data downloaded from F1's timing services or OpenF1.
- Formula 1 timing data and F1 TV content are subject to F1's terms. A personal F1 TV subscription is for personal use;
  it does not by itself give a right to republish downloaded data or to build a public dataset from it. Check the
  current F1 TV and F1 data terms (or ask F1) before sharing any dataset.
- OpenF1 has its own terms and now requires an account for some data. Credit it, and check its terms before
  redistributing anything from it.
- Safe to share: your code, your written analysis and conclusions, charts you made yourself (credit the source),
  and the SYNTHETIC demo (`--demo`), which is labelled as synthetic.
- Anyone cloning the repo builds their own local copy of the data with their own access.
