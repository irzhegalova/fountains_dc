"""Reusable, explicit-state fountain analysis."""
from .core import (
    Settings, Thresholds, AnalysisResult, calculate_thresholds, load_mask,
    run_dataset, filter_calls, score_snippets, add_peak_prominence,
    read_intervals, match_manual_peaks, select_examples,
)

__version__ = "2.0.0"
