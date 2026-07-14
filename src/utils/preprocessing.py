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
       """
       Compute the mean and standard deviation of each feature.
       """
       X = np.asarray(X, dtype=float)

       self.mean_ = np.mean(X, axis=0)
       self.scale_ = np.std(X, axis=0)

       return self
    
    def transform(self, X):
       """Scale features using the fitted statistics."""
       X = np.asarray(X, dtype=float)

       return (X - self.mean_) / self.scale_