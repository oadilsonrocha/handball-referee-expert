"""Busca dos trechos relevantes (RAG) por palavras-chave (BM25), sem serviços externos."""
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_DATA = ROOT / "data" if (ROOT / "data" / "chunks.json").exists() else ROOT
CHUNKS_PATH = _DATA / "chunks.json"

STOP = set("""a an the and or of to in on at by for with from as is are be been being was were it its
this that these those which who whom what when where how why if then than so not no can may must shall
should will would do does did has have had there their they them he she his her i you we our your
any all each other such only also into out up down over under more most very""".split())

TOKEN = re.compile(r"\d{1,2}:\d{1,2}[a-z]?|[a-zà-ú0-9]+")


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str):
    return [_stem(t) for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


class Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c['title']} {c['text']}")) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k=5, modalities=None):
        q = Counter(tokenize(query))
        refs = set(re.findall(r"\b\d{1,2}:\d{1,2}[a-z]?\b", query))
        scored = []
        for i, d in enumerate(self.docs):
            c = self.chunks[i]
            if modalities and c["modality"] not in modalities:
                continue
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            for t, qf in q.items():
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if refs and any(r in c["text"] for r in refs):
                s += 3.0
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:k]]


def load_index():
    return Index(json.loads(CHUNKS_PATH.read_text(encoding="utf-8")))
