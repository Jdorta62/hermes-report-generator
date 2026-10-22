"""Rendering of a run record as the self-contained ``report.html``."""

from hermes_report.render.compare import render_compare, write_compare
from hermes_report.render.run import render_run, write_report

__all__ = ["render_compare", "render_run", "write_compare", "write_report"]
