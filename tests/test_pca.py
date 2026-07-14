import numpy as np

from src.unsupervised.pca import PCA


def test_pca_output_shape():
    X = np.array(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ]
    )

    pca = PCA(n_components=1)
    X_transformed = pca.fit_transform(X)

    assert X_transformed.shape == (10, 1)
    assert pca.components_.shape == (1, 2)


def test_pca_explained_variance_ratio():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))

    pca = PCA(n_components=4)
    pca.fit(X)

    assert np.all(pca.explained_variance_ratio_ >= 0)
    assert np.isclose(
        np.sum(pca.explained_variance_ratio_),
        1.0,
    )


def test_pca_transformed_data_is_centered():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))

    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    assert np.allclose(
        np.mean(X_transformed, axis=0),
        0.0,
        atol=1e-10,
    )