"""Node data structure for the CART decision tree.

Kept as a standalone dataclass (rather than nested inside DecisionTree)
so it can be imported and unit-tested independently, and so that
_splitter.py / decision_tree.py don't need to know about each other's
internals to construct or traverse nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """A single node in the decision tree.

    Leaf nodes have feature_index/threshold/left/right all left at
    their default (None). Internal nodes have feature_index, threshold,
    left, and right all populated.

    Attributes:
        samples: Number of samples reaching this node (unweighted count).
        value: Class distribution at this node, stored as raw (weighted)
            counts aligned to the tree's sorted unique classes -- NOT
            normalized proportions. Normalization happens only at the
            point of use (predict_proba, feature_importances), so this
            value is always the ground truth for both.
        impurity_value: Impurity of this node itself (i.e. the "parent"
            impurity used when computing this node's split gain, if any).
        depth: Depth of this node in the tree (root = 0).
        feature_index: Index of the feature used to split at this node,
            or None if this is a leaf.
        threshold: Split threshold (samples with feature <= threshold go
            left), or None if this is a leaf.
        left: Left child Node, or None if this is a leaf.
        right: Right child Node, or None if this is a leaf.
    """

    samples: int
    value: np.ndarray
    impurity_value: float
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        """A node is a leaf iff it has no children (both must be None)."""
        return self.left is None and self.right is None
