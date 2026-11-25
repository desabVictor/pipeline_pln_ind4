"""Módulo de vetorização (CountVectorizer e TF-IDF) e ranking de termos."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def carregar_corpus(
    input_path: Path | str,
    column_id: str = "id",
    column_text: str = "clean_text",
) -> pd.DataFrame:
    """Carrega o corpus pré-processado e valida as colunas obrigatórias."""
    df_corpus = pd.read_csv(input_path, encoding="utf-8")

    if column_id not in df_corpus.columns:
        raise ValueError(
            f"Coluna '{column_id}' não encontrada. Colunas disponíveis: {list(df_corpus.columns)}"
        )
    if column_text not in df_corpus.columns:
        raise ValueError(
            f"Coluna '{column_text}' não encontrada. Colunas disponíveis: {list(df_corpus.columns)}"
        )

    df_corpus = df_corpus.dropna(subset=[column_text]).reset_index(drop=True)
    print(f"Total de documentos carregados: {len(df_corpus)}")
    return df_corpus


def _extrair_ranking(matrix, feature_names, score_col: str) -> pd.DataFrame:
    """Calcula a soma dos scores por termo e retorna um DataFrame ordenado."""
    scores = np.asarray(matrix.sum(axis=0)).flatten()
    return (
        pd.DataFrame({"term": feature_names, score_col: scores})
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )


def vetorizar_count(
    texts: pd.Series,
    stop_words: set[str] | list[str] | None = None,
    ngram_range: tuple[int, int] = (1, 1),
    max_df: float = 0.85,
    min_df: int = 5,
):
    """Aplica CountVectorizer e retorna (vectorizer, matrix, ranking_df)."""
    sw = list(stop_words) if stop_words else None
    vectorizer = CountVectorizer(
        stop_words=sw,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Dimensão CountVectorizer (documentos x termos): {matrix.shape}")
    print(f"Tamanho do vocabulário: {len(feature_names)}")

    ranking_df = _extrair_ranking(matrix, feature_names, score_col="countv_scores")
    return vectorizer, matrix, ranking_df


def vetorizar_tfidf(
    texts: pd.Series,
    stop_words: set[str] | list[str] | None = None,
    ngram_range: tuple[int, int] = (1, 1),
    max_df: float = 0.85,
    min_df: int = 5,
):
    """Aplica TfidfVectorizer e retorna (vectorizer, matrix, ranking_df)."""
    sw = list(stop_words) if stop_words else None
    vectorizer = TfidfVectorizer(
        stop_words=sw,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Dimensão TF-IDF (documentos x termos): {matrix.shape}")
    print(f"Tamanho do vocabulário: {len(feature_names)}")

    ranking_df = _extrair_ranking(matrix, feature_names, score_col="sum_tfidf")
    return vectorizer, matrix, ranking_df
