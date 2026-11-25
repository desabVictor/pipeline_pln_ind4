"""Módulo de coleta de dados via API OpenAlex (pyalex) e construção do dataset."""

import pandas as pd
import pyalex
from pyalex import Concepts, Works


def configurar_pyalex(email: str = "", api_key: str = "") -> None:
    """Configura e-mail e chave de API opcionais para requisições no OpenAlex."""
    if email:
        pyalex.config.email = email
    if api_key:
        pyalex.config.api_key = api_key


def buscar_conceito_openalex(termo_busca: str = "Industry 4.0") -> dict | None:
    """Busca um conceito no OpenAlex pelo nome e retorna o resultado mais relevante."""
    resultados = Concepts().search(termo_busca).get()
    if not resultados:
        return None
    return resultados[0]


def coletar_artigos_openalex(
    concept_id: str,
    year_start: int,
    year_end: int,
    keywords_areas: dict[str, list[str]],
    language: str = "en",
    doc_type: str = "article",
    per_page: int = 200,
) -> tuple[dict, dict[str, set[str]]]:
    """Coleta artigos únicos no OpenAlex filtrados por conceito, período e palavras-chave."""
    unique_works: dict = {}
    id_to_categories: dict[str, set[str]] = {}

    for category, terms in keywords_areas.items():
        print(f"Categoria: {category}")
        for term in terms:
            print(f"  > Buscando termo: {term}...", end=" ")
            try:
                query = (
                    Works()
                    .filter(
                        concepts={"id": concept_id},
                        from_publication_date=f"{year_start}-01-01",
                        to_publication_date=f"{year_end}-12-31",
                        type=doc_type,
                        language=language,
                    )
                    .filter(title_and_abstract={"search": term})
                )

                count_termo = 0
                for page in query.paginate(per_page=per_page):
                    for work in page:
                        w_id = work["id"]
                        if w_id not in unique_works:
                            unique_works[w_id] = work
                        if w_id not in id_to_categories:
                            id_to_categories[w_id] = set()
                        id_to_categories[w_id].add(category)
                        count_termo += 1

                print(f"Encontrados: {count_termo}")
            except Exception as exc:
                print(f"Erro ao buscar '{term}': {exc}")

    print(f"\nExtração concluída: {len(unique_works)} artigos únicos encontrados.")
    return unique_works, id_to_categories


def rebuild_abstract(inverted: dict | None) -> str:
    """Reconstrói o texto do resumo a partir do índice invertido retornado pelo OpenAlex."""
    if not inverted or not isinstance(inverted, dict):
        return ""
    pos_map = {pos: word for word, positions in inverted.items() for pos in positions}
    return " ".join(pos_map[i] for i in sorted(pos_map.keys()))


def construir_dataframe(
    unique_works: dict,
    id_to_categories: dict[str, set[str]],
) -> pd.DataFrame:
    """Converte os artigos coletados em um DataFrame estruturado."""
    data_list = []
    for w_id, work in unique_works.items():
        data_list.append(
            {
                "id": w_id,
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "doi": work.get("doi"),
                "cited_by": work.get("cited_by_count"),
                "abstract": rebuild_abstract(work.get("abstract_inverted_index")),
                "all_areas": ", ".join(sorted(list(id_to_categories.get(w_id, set())))),
            }
        )
    return pd.DataFrame(data_list)
