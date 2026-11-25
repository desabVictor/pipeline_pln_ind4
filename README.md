# pipeline_pln_ind4

Repositório dedicado às ferramentas e códigos utilizados no trabalho **"PROCESSAMENTO DE LINGUAGEM NATURAL APLICADO À IDENTIFICAÇÃO DE TENDÊNCIAS DA LITERATURA SOBRE INDÚSTRIA 4.0"**, de autoria de **Victor Souza Silva** em parceria com o **Grupo de Pesquisa NLP**.

---

## Estrutura do Repositório

```text
pipeline_pln_ind4/
├── data/
│   ├── raw/                                  # Dados brutos coletados da API OpenAlex
│   └── processed/                            # Datasets limpos, pré-processados e curadoria
├── notebooks/                                # Notebooks autocontidos em ordem de execução
│   ├── 01_coleta_e_construcao_dataset.ipynb
│   ├── 02_limpeza_e_filtragem.ipynb
│   ├── 03_preprocessamento_texto.ipynb
│   ├── 04_curadoria_stopwords.ipynb
│   ├── 05_vetorizacao_e_ranking.ipynb
│   └── 06_modelagem_topicos.ipynb
├── results/                                  # Rankings de termos e tópicos/clusters gerados
├── src/                                      # Módulos Python reutilizáveis do pipeline
│   ├── __init__.py
│   ├── coleta.py
│   ├── limpeza.py
│   ├── preprocessamento.py
│   ├── curadoria.py
│   ├── vetorizacao.py
│   └── modelagem.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Pré-requisitos e Instalação

1. **Clone o repositório:**
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd pipeline_pln_ind4
   ```

2. **Crie e ative um ambiente virtual:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Baixe o modelo de linguagem do spaCy:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## Como Replicar o Pipeline

Execute os notebooks na pasta `notebooks/` na ordem numérica abaixo. Cada notebook possui uma seção inicial de **Parâmetros de Configuração** caso deseje ajustar período, palavras-chave, limiares ou número de tópicos.

| Ordem | Notebook | Descrição | Entrada | Saída Principal |
| :---: | :--- | :--- | :--- | :--- |
| **01** | `01_coleta_e_construcao_dataset.ipynb` | Conecta à API OpenAlex (`pyalex`), valida o conceito alvo, coleta artigos e reconstrói os resumos. | API OpenAlex | `data/raw/dataset_bruto.csv` |
| **02** | `02_limpeza_e_filtragem.ipynb` | Remove resumos vazios, tags HTML/JATS, *mojibake*, textos fora do idioma alvo e transcrições de vídeo. | `data/raw/dataset_bruto.csv` | `data/processed/dataset_limpo.csv` |
| **03** | `03_preprocessamento_texto.ipynb` | Aplica lematização e remoção de stopwords, números e pontuação com `spaCy`. | `data/processed/dataset_limpo.csv` | `data/processed/dataset_preprocessado.csv` |
| **04** | `04_curadoria_stopwords.ipynb` | Calcula a frequência documental (*DF*) e exporta termos candidatos a ruído para revisão manual. | `data/processed/dataset_preprocessado.csv` | `data/processed/curadoria_stopwords.csv` |
| **05** | `05_vetorizacao_e_ranking.ipynb` | Gera as matrizes `CountVectorizer` e `TF-IDF` aplicando as stopwords curadas e exporta rankings de termos. | `dataset_preprocessado.csv` + `curadoria_stopwords.csv` | `results/ranking_countvectorizer.csv` e `results/ranking_tfidf.csv` |
| **06** | `06_modelagem_topicos.ipynb` | Ajusta os modelos **LDA**, **NMF** e **K-Means (com TruncatedSVD)** e extrai os termos principais. | `dataset_preprocessado.csv` + `curadoria_stopwords.csv` | `results/topicos_lda.csv`, `results/topicos_nmf.csv` e `results/clusters_kmeans.csv` |

> [!NOTE]
> **Etapa de Curadoria Manual (Notebook 04):** Após rodar a exportação em `04_curadoria_stopwords.ipynb`, abra o arquivo `data/processed/curadoria_stopwords.csv` e preencha a coluna `decision` com `excluir` nos termos que devem ser tratados como stopwords adicionais nas etapas **05** e **06**.

---

## Autoria e Parceria

- **Autor:** Victor Souza Silva
- **Grupo de Pesquisa:** Grupo de Pesquisa NLP — IFAM CMDI
