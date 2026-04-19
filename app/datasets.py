"""Backward-compatible shim for dataset helpers.

New flat-directory Colab workflows should use ``hp_datasets.py`` instead of
``datasets.py`` to avoid colliding with Hugging Face's package name.
"""

from .hp_datasets import *  # noqa: F401,F403
