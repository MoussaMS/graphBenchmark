"""
benchmark.py
────────────
Charge le benchmark depuis un fichier Excel et retourne la liste des conjectures.
"""

from __future__ import annotations
import pandas as pd
from typing import List
from .conjecture import Conjecture


def load_benchmark(path: str) -> List[Conjecture]:
    """
    Charge toutes les conjectures depuis le fichier Excel.
    Retourne une liste de Conjecture.
    """
    df = pd.read_excel(path)
    conjectures = []
    for _, row in df.iterrows():
        try:
            c = Conjecture(row)
            conjectures.append(c)
        except Exception as e:
            print(f"[WARN] Conjecture ignorée (ID={row.get('Conjecture ID', '?')}): {e}")
    return conjectures
