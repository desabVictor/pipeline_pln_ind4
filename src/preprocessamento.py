"""Módulo de pré-processamento linguístico (lematização e remoção de stopwords) com spaCy."""

import pandas as pd
import spacy


def carregar_modelo_spacy(modelo: str = "en_core_web_sm"):
    """Carrega o modelo spaCy especificado, orientando a instalação caso não exista."""
    try:
        return spacy.load(modelo)
    except OSError:
        print(f"Modelo spaCy '{modelo}' não encontrado. Execute: python -m spacy download {modelo}")
        raise


def limpar_texto(
    text: str,
    nlp,
    stopwords_extra: set[str] | None = None,
) -> str:
    """Normaliza, lematiza e remove stopwords, pontuação, números e tokens curtos."""
    if not isinstance(text, str) or text.strip() == "":
        return ""

    extra = stopwords_extra or set()
    doc = nlp(text.lower())

    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and token.lemma_.lower() not in extra
        and not token.is_punct
        and not token.is_space
        and not token.like_num
        and len(token.lemma_) > 2
    ]
    return " ".join(tokens)


def preprocessar_dataset(
    df_clean: pd.DataFrame,
    nlp,
    stopwords_extra: set[str] | None = None,
) -> pd.DataFrame:
    """Aplica o pré-processamento textual na coluna 'abstract' e remove textos vazios."""
    print("Iniciando pré-processamento textual sobre a coluna 'abstract'...")
    clean_text = df_clean["abstract"].apply(lambda t: limpar_texto(t, nlp, stopwords_extra))

    df_custom = pd.DataFrame(
        {
            "id": df_clean["id"],
            "year": df_clean["year"],
            "clean_text": clean_text,
            "all_areas": df_clean["all_areas"],
        }
    )

    count_before = len(df_custom)
    df_custom = df_custom[df_custom["clean_text"].str.strip() != ""].reset_index(drop=True)
    count_final = len(df_custom)

    print(f"Total antes do descarte por texto vazio: {count_before}")
    print(f"Total com 'clean_text' válido: {count_final}")
    print(f"Artigos descartados: {count_before - count_final}")
    return df_custom
