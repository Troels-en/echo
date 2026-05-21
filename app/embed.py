"""Local multilingual embeddings via sentence-transformers."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    name = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    log.info("loading embedding model: %s", name)
    m = SentenceTransformer(name)
    log.info("embed model ready, dim=%d", m.get_embedding_dimension())
    return m


def dim() -> int:
    return _model().get_embedding_dimension()


def embed(text: str | list[str]) -> np.ndarray:
    single = isinstance(text, str)
    texts = [text] if single else text
    vecs = _model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs[0] if single else vecs
