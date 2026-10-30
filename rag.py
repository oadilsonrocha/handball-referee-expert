"""Busca dos trechos relevantes (RAG) usando embeddings do Gemini."""
import json
import re
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors, types

ROOT = Path(__file__).resolve().parent
# Aceita os arquivos dentro de data/ ou soltos na raiz (versão simplificada para celular).
_DATA = ROOT / "data" if (ROOT / "data" / "chunks.json").exists() else ROOT
CHUNKS_PATH = _DATA / "chunks.json"
EMB_PATH = _DATA / "embeddings.npy"

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH = 20


def load_chunks():
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


def _embed(client: genai.Client, texts, task_type: str, retries: int = 10):
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM)
    for attempt in range(retries):
        try:
            res = client.models.embed_content(model=EMBED_MODEL, contents=texts, config=cfg)
            return np.array([e.values for e in res.embeddings], dtype=np.float32)
        except errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(min(15 * (attempt + 1), 60))  # limite por minuto ou instabilidade
                continue
            raise


def build_embeddings(client: genai.Client, chunks, progress=None) -> np.ndarray:
    """Gera os embeddings de todos os trechos (uma vez só) e salva em disco."""
    parts = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [f"{c['title']}\n{c['text']}" for c in batch]
        parts.append(_embed(client, texts, "RETRIEVAL_DOCUMENT"))
        if progress:
            progress(min(i + BATCH, len(chunks)) / len(chunks))
    emb = _normalize(np.vstack(parts))
    EMB_PATH.parent.mkdir(exist_ok=True)
    np.save(EMB_PATH, emb)
    return emb


def load_embeddings(n_chunks: int):
    if EMB_PATH.exists():
        emb = np.load(EMB_PATH)
        if emb.shape[0] == n_chunks:
            return emb
    return None


RULE_REF = re.compile(r"\b(\d{1,2}:\d{1,2}[a-z]?)\b")


def search(client, question, chunks, emb, k=6, modalities=None):
    """Retorna os k trechos mais parecidos com a pergunta."""
    q = _normalize(_embed(client, [question], "RETRIEVAL_QUERY"))[0]
    scores = emb @ q

    # Pequeno reforço quando a pergunta cita uma regra pelo número (ex.: "8:5").
    refs = RULE_REF.findall(question)
    if refs:
        for i, c in enumerate(chunks):
            if any(r in c["text"] for r in refs):
                scores[i] += 0.05

    if modalities:
        mask = np.array([c["modality"] in modalities for c in chunks])
        scores = np.where(mask, scores, -1.0)

    top = np.argsort(-scores)[:k]
    return [chunks[i] for i in top if scores[i] > 0]
