# utils.py

import numpy as np
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalise les vecteurs pour avoir une norme unitaire.

    :param vectors: Tableau numpy des vecteurs à normaliser.
    :return: Tableau numpy des vecteurs normalisés.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def compute_clustering_metrics(vectors: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calcule diverses métriques de clustering.

    :param vectors: Tableau numpy des vecteurs encodés.
    :param labels: Tableau numpy des labels de cluster prédits.
    :return: Dictionnaire contenant les scores des métriques.
    """
    silhouette = silhouette_score(vectors, labels)
    davies_bouldin = davies_bouldin_score(vectors, labels)
    calinski_harabasz = calinski_harabasz_score(vectors, labels)
    return {
        'Silhouette Score': silhouette,
        'Davies-Bouldin Index': davies_bouldin,
        'Calinski-Harabasz Index': calinski_harabasz
    }
