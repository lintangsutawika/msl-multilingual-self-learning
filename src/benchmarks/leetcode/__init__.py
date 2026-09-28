"""LeetCode multilingual benchmark: generate Harbor tasks from per-language templates."""

from .adapter import LANGUAGES
from .main import main

__all__ = [
    "LANGUAGES",
    "main",
]