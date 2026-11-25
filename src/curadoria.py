"""Módulo de curadoria de stopwords baseada em Document Frequency (DF)."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def calcular_document_frequency(
    texts: pd.Series,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 5,
) -> pd.DataFrame:
    """Calcula a frequência documental (DF) absoluta e percentual de cada termo."""
    vectorizer_df = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        binary=True,
    )
    binary_matrix = vectorizer_df.fit_transform(texts)
    doc_frequency = binary_matrix.sum(axis=0).A1
    terms = vectorizer_df.get_feature_names_out()
    total_docs = len(texts)

    print(f"Total de termos únicos (min_df={min_df}): {len(terms)}")

    table_df = (
        pd.DataFrame(
            {
                "term": terms,
                "doc_frequency": doc_frequency,
                "doc_frequency_pct": (doc_frequency / total_docs * 100).round(2),
            }
        )
        .sort_values("doc_frequency", ascending=False)
        .reset_index(drop=True)
    )
    return table_df


def filtrar_candidatos_ruido(
    table_df: pd.DataFrame,
    edge_review_pct: float = 15.0,
) -> pd.DataFrame:
    """Seleciona termos com DF percentual acima do limiar para revisão manual."""
    targets_review = table_df[table_df["doc_frequency_pct"] >= edge_review_pct].copy()
    targets_review["decision"] = ""
    print(f"Termos com DF >= {edge_review_pct}%: {len(targets_review)}")
    return targets_review


def exportar_curadoria(targets_review: pd.DataFrame, output_path: Path | str) -> Path:
    """Exporta a tabela de termos candidatos para revisão manual em CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    targets_review.to_csv(path, index=False, encoding="utf-8")
    print(f"Arquivo de curadoria exportado: '{path}' ({len(targets_review)} termos).")
    return path


def carregar_stopwords_curadas(
    curadoria_path: Path | str,
    decision_col: str = "decision",
    exclude_value: str = "excluir",
) -> set[str]:
    """Lê o CSV de curadoria revisado e retorna o conjunto de stopwords marcadas para exclusão."""
    final_selection = pd.read_csv(curadoria_path, encoding="utf-8")
    custom_stopwords = set(
        final_selection.loc[
            final_selection[decision_col].astype(str).str.strip().str.lower() == exclude_value.lower(),
            "term",
        ]
    )
    print(f"CUSTOM_STOPWORDS construída com {len(custom_stopwords)} termos.")
    return custom_stopwords
