"""
Preprocessing utilities for the project.
"""

import numpy as np

class StandardScaler:
    """Standardize features using mean and standard deviation."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """Compute mean and standard deviation."""
        pass