"""
benchmark.py
------------
Load conjectures from the benchmark spreadsheet.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from .conjecture import Conjecture


def load_benchmark(path: str) -> List[Conjecture]:
    """
    Load all conjectures and fail loudly on malformed rows.
    """
    df = pd.read_excel(path)
    conjectures: List[Conjecture] = []
    errors = []
    for _, row in df.iterrows():
        try:
            conjectures.append(Conjecture(row))
        except Exception as e:
            errors.append(f"ID={row.get('Conjecture ID', '?')}: {e}")

    if errors:
        details = "\n".join(errors[:10])
        raise RuntimeError(
            f"{len(errors)} conjecture(s) invalide(s) dans le benchmark.\n{details}"
        )
    return conjectures
