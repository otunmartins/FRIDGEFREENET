#!/usr/bin/env python3
"""
Scalar discovery score for autoresearch-style keep/discard decisions.

Maps MD / PropertyExtractor feedback to a single number (higher = better).
"""

from typing import Any, Mapping


def discovery_score(
    feedback: Mapping[str, Any],
    high_performer_weight: float = 2.0,
    mechanism_weight: float = 0.5,
    problematic_weight: float = 1.0,
) -> float:
    """
    Compute a scalar score from evaluation feedback (higher is better).

    Args:
        feedback: Dict with optional keys high_performers, effective_mechanisms,
            problematic_features (lists or similar).
        high_performer_weight: Points per high performer.
        mechanism_weight: Points per effective mechanism.
        problematic_weight: Penalty per problematic feature.

    Returns:
        Composite score (float). Non-list values are coerced safely.
    """
    hp = _len_safe(feedback.get("high_performers"))
    mech = _len_safe(feedback.get("effective_mechanisms"))
    bad = _len_safe(feedback.get("problematic_features"))
    return hp * high_performer_weight + mech * mechanism_weight - bad * problematic_weight


def _len_safe(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, str):
        return 1 if value.strip() else 0
    return 1
