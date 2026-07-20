"""evoviz: static, self-contained run reports (viz_design.md, minimal cut)."""

from .report import generate, load_run, render_html

__all__ = ["generate", "load_run", "render_html"]
