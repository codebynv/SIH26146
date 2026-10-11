"""Step 10 evaluation: deterministic, offline functional testing of BitTrace.

Nothing in this package trains, downloads or calls out. It reads three small
synthetic files from ``evaluation/scenarios/``, pushes them through the shipped
pipeline, and reports what came out.

Entry point::

    python -m evaluation.run_evaluation
"""

__all__ = ["canonical", "corpus", "harness", "run_evaluation"]