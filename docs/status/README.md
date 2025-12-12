### Status snapshots (tracked in Git)

This folder contains **lightweight, human-readable status exports** that are safe to commit to Git.

What goes here:
- `run_registry_export.json`: latest N runs (ids, status, timestamps, paths)
- `run_registry_export.md`: quick Markdown table view
- `run_cards/`: one Markdown file per `run_id` (links to Drive results/artifacts, key metrics)

What should NOT go here:
- Heavy artifacts (checkpoints, plots) → keep on Drive under `.../artifacts/runs/<run_id>/`
- Full `results/` trees → keep on Drive
- The SQLite DB itself (changes frequently; merge-unfriendly)

How to refresh:
- `python -m src.tools.export_status --db <path-to-run_registry.sqlite> --limit 50`
- `python -m src.tools.export_run_cards --db <path-to-run_registry.sqlite> --limit 50`


