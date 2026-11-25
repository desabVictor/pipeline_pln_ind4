"""Módulo de limpeza e filtragem de qualidade dos resumos coletados."""

import html
import re

import pandas as pd
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0  # Garante determinismo na detecção de idioma


class MojibakeDetector:
    """Detector de texto com problemas de codificação (mojibake)."""

    PATTERN_SMART_QUOTES_BROKEN = re.compile(r"â€")
    PATTERN_LATIN1_SCRIPT_MIXED = re.compile(r"[\xC0-\xFF]{2,}")
    PATTERN_UTF8_CONTROL_BYTES = re.compile(
        r"Ã[\x80-\xbf]|Â[\x80-\xbf]|Ð[\x80-\xbf]|Ñ[\x80-\xbf]"
    )

    @classmethod
    def has_mojibake(cls, text: str) -> bool:
        if not isinstance(text, str) or len(text) == 0:
            return False
        if cls.PATTERN_SMART_QUOTES_BROKEN.search(text):
            return True
        if cls.PATTERN_LATIN1_SCRIPT_MIXED.search(text):
            return True
        if cls.PATTERN_UTF8_CONTROL_BYTES.search(text):
            return True
        return False


def detectar_idioma(text: str) -> str | None:
    """Detecta o idioma predominante de um texto usando langdetect."""
    if not isinstance(text, str) or len(text) < 10:
        return None
    try:
        return detect(text)
    except Exception:
        return None


def transcricao_video(text: str) -> bool:
    """Identifica se o texto apresenta padrão típico de transcrição de vídeo (timestamps)."""
    if not isinstance(text, str) or len(text) < 50:
        return False

    timestamp_pattern = re.compile(r"\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2}")
    timestamps = timestamp_pattern.findall(text)

    if not timestamps:
        return False

    densidade = len(timestamps) / (len(text) / 1000)
    if densidade > 2:
        return True

    if text.lstrip().startswith(("0:0", "0:", "1:", "2:", "3:", "4:", "5:", "6:", "7:", "8:", "9:")):
        if len(timestamps) >= 5:
            return True

    return False


def remover_ruido_html(text: str) -> str:
    """Remove entidades e tags HTML/JATS residuais."""
    if pd.isna(text):
        return text
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def limpar_dataset(df_init: pd.DataFrame, target_lang: str = "en") -> pd.DataFrame:
    """Executa o pipeline completo de limpeza e filtragem de qualidade sobre o DataFrame."""
    print("INICIANDO PIPELINE DE LIMPEZA E FILTRAGEM...")
    print("=" * 80)
    print(f"Total de artigos iniciais: {len(df_init)}")

    # 1. Resumos vazios
    df_clean = df_init.dropna(subset=["abstract"]).copy()
    discard_nan = len(df_init) - len(df_clean)
    print(f"[1/5] Resumos vazios descartados: {discard_nan} | Restantes: {len(df_clean)}")

    # 2. Tags HTML/JATS
    count_html = df_clean["abstract"].str.contains(r"<[^>]+>", regex=True, na=False).sum()
    df_clean["abstract"] = df_clean["abstract"].apply(remover_ruido_html)
    print(f"[2/5] Artigos com tags HTML limpas: {count_html}")

    # 3. Mojibake
    mask_mojibake = df_clean["abstract"].apply(MojibakeDetector.has_mojibake)
    count_mojibake = int(mask_mojibake.sum())
    df_clean = df_clean[~mask_mojibake].copy()
    print(f"[3/5] Artigos com mojibake removidos: {count_mojibake} | Restantes: {len(df_clean)}")

    # 4. Idioma
    language_detection = df_clean["abstract"].apply(detectar_idioma)
    mask_other_lang = language_detection != target_lang
    count_other_lang = int(mask_other_lang.sum())
    df_clean = df_clean[~mask_other_lang].copy()
    print(f"[4/5] Artigos fora do idioma '{target_lang}' removidos: {count_other_lang} | Restantes: {len(df_clean)}")

    # 5. Transcrições de vídeo
    mask_video = df_clean["abstract"].apply(transcricao_video)
    count_video = int(mask_video.sum())
    df_clean = df_clean[~mask_video].copy()
    df_clean = df_clean.reset_index(drop=True)
    print(f"[5/5] Transcrições de vídeo removidas: {count_video} | Restantes: {len(df_clean)}")

    taxa = (len(df_clean) / len(df_init) * 100) if len(df_init) > 0 else 0.0
    print("=" * 80)
    print(f"Artigos finais: {len(df_clean)} (Taxa de retenção: {taxa:.2f}%)")
    return df_clean
