"""
Data-related utilities for the C-MAPSS RUL/HI pipelines.

This subpackage intentionally keeps feature-engineering and target-construction
logic separate from model code so that:

- training and diagnostics can share the same helpers, and
- new physics-informed targets (e.g. HI_phys) can be added without touching
  core model implementations.
"""


