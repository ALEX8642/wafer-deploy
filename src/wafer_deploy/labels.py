"""labels.py — the 8 basic defect labels, in the canonical column order.

Every logit column, probability, temperature and threshold vector in this repo
indexes by this order. It MUST equal wafer_mixed.data.LABEL_NAMES — the bridge
(see bridge.py) asserts that equality loudly, because a silent drift would
scramble every downstream array while staying self-consistent in isolation.
"""
from __future__ import annotations

LABELS: list[str] = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Scratch", "Random",
]
NUM_LABELS = len(LABELS)
