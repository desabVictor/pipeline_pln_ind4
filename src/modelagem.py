"""Módulo de modelagem de tópicos e clusterização (LDA, NMF e K-Means com SVD)."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD


def treinar_lda(
    count_matrix,
    n_topics: int = 5,
    learning_method: str = "batch",
    random_state: int = 42,
) -> LatentDirichletAllocation:
    """Ajusta um modelo LDA sobre a matriz de contagem."""
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method=learning_method,
        random_state=random_state,
    )
    lda.fit(count_matrix)
    print(f"LDA ajustado com {n_topics} tópicos.")
    return lda


def treinar_nmf(
    tfidf_matrix,
    n_topics: int = 5,
    init: str = "nndsvd",
    random_state: int = 42,
) -> tuple[NMF, np.ndarray, np.ndarray]:
    """Ajusta um modelo NMF sobre a matriz TF-IDF e retorna (modelo, W, H)."""
    nmf = NMF(
        n_components=n_topics,
        random_state=random_state,
        init=init,
    )
    W = nmf.fit_transform(tfidf_matrix)
    H = nmf.components_

    print(f"NMF ajustado com {n_topics} tópicos.")
    print(f"Shape de W (documentos x tópicos): {W.shape}")
    print(f"Shape de H (tópicos x termos): {H.shape}")
    print(f"Erro de reconstrução: {nmf.reconstruction_err_:.3f}")
    return nmf, W, H


def treinar_kmeans_svd(
    tfidf_matrix,
    n_clusters: int = 5,
    n_components_svd: int = 100,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[TruncatedSVD, np.ndarray, KMeans]:
    """Reduz dimensionalidade da matriz TF-IDF via TruncatedSVD e ajusta K-Means."""
    n_comp = min(n_components_svd, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(
        n_components=n_comp,
        random_state=random_state,
    )
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    variancia_explicada = svd.explained_variance_ratio_.sum()

    print(f"Shape após SVD: {tfidf_reduced.shape}")
    print(f"Variância explicada acumulada: {variancia_explicada:.3f}")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    kmeans.fit(tfidf_reduced)
    print(f"K-Means ajustado com {n_clusters} clusters.")
    return svd, tfidf_reduced, kmeans


def extrair_top_termos(
    components: np.ndarray,
    feature_names: np.ndarray,
    n_top_words: int = 10,
    prefixo: str = "Tópico",
) -> pd.DataFrame:
    """Extrai os termos mais representativos de cada tópico/cluster."""
    linhas = []
    for idx, comp in enumerate(components):
        top_indices = comp.argsort()[: -n_top_words - 1 : -1]
        top_terms = [feature_names[i] for i in top_indices]
        linhas.append(
            {
                "grupo": f"{prefixo} {idx + 1}",
                "top_termos": ", ".join(top_terms),
            }
        )
    return pd.DataFrame(linhas)
