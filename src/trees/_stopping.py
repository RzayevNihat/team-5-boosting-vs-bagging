"""Stopping-criteria checks for CART tree growth.

Kept as a standalone pure function (independent of split search) so it
can be unit-tested in isolation.
"""

from __future__ import annotations

import numpy as np


def should_stop(
    depth: int,
    n_samples: int,
    y: np.ndarray,
    max_depth: int | None,
    min_samples_split: int,
) -> bool:
    """Return True if node growth should stop at this node.

    Checks (any one being true stops growth):
        - max_depth reached (if max_depth is not None).
        - fewer than min_samples_split samples reached this node.
        - the node is already pure (single class).

    Note: a fourth condition -- "no valid split exists" (e.g. all
    feature vectors are identical) -- is not checked here, since it can
    only be discovered by actually running find_best_split. That case
    is handled by DecisionTree._build_tree treating a None result from
    find_best_split as an implicit leaf.

    Args:
        depth: Current depth of the node being considered (root = 0).
        n_samples: Number of samples at the current node.
        y: Class labels at the current node.
        max_depth: Maximum allowed depth, or None for unbounded.
        min_samples_split: Minimum samples required to split further.

    Returns:
        True if growth should stop at this node.
    """
    if max_depth is not None and depth >= max_depth:
        return True
    if n_samples < min_samples_split:
        return True
    if len(np.unique(y)) == 1:
        return True  # pure node
    return False
