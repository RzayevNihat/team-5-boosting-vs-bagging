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

    def fit(self, X):
        """
        Fit PCA by computing the covariance matrix,
        eigenvalues, and eigenvectors.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        n_samples, n_features = X.shape

        if self.n_components <= 0:
            raise ValueError("n_components must be greater than zero.")

        if self.n_components > n_features:
            raise ValueError(
                "n_components cannot be greater than the number of features."
            )

        if n_samples < 2:
            raise ValueError("PCA requires at least two samples.")

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        covariance_matrix = np.cov(X_centered, rowvar=False)
        covariance_matrix = np.atleast_2d(covariance_matrix)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        eigenvalues = np.maximum(eigenvalues, 0.0)

        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]

        total_variance = np.sum(eigenvalues)

        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(
                self.n_components
            )
        else:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )

        return self
    
    def transform(self, X):
            """
            Project data onto the fitted principal components.
            """
            if self.mean_ is None or self.components_ is None:
                raise RuntimeError(
                    "PCA must be fitted before transform."
                )

            X = np.asarray(X, dtype=float)

            if X.ndim != 2:
                raise ValueError("X must be a two-dimensional array.")

            if X.shape[1] != self.mean_.shape[0]:
                raise ValueError(
                    "X must have the same number of features as the fitted data."
                )

            X_centered = X - self.mean_

            return X_centered @ self.components_.T

    def fit_transform(self, X):
            """
            Fit PCA and return the projected data.
            """
            return self.fit(X).transform(X)