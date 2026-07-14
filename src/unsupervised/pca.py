"""
Principal Component Analysis (PCA) implementation.
"""

import numpy as np


class PCA:
    """Principal Component Analysis implemented from scratch."""

    def __init__(self, n_components):
        self.n_components = n_components

        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None