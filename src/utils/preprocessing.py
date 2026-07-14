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

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # Prevent division by zero for constant features.
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)

        return self
    
    def transform(self, X):
        """Scale features using the fitted statistics."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "StandardScaler must be fitted before transform."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                "X must have the same number of features as the fitted data."
            )

        return (X - self.mean_) / self.scale_
    

    def fit_transform(self, X):
        """
        Fit the scaler and return the transformed data.
        """
        return self.fit(X).transform(X)
    
