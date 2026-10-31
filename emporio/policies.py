"""BM25 search over the store policy manual (data/politicas.md).

The manual is split into one chunk per heading (``##`` and ``###``), each chunk
keeping its heading path so the model can cite the section. BM25 is implemented
here (no extra dependency): the manual is tiny and lexical matching on
accent-stripped Portuguese tokens works well for it.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from .data import DATA_DIR, normalize

K1, B = 1.5, 0.75
_TOKEN = re.compile(r"[a-z0-9]+")
# Very small stop list; BM25 IDF already down-weights common words.
_STOP = {"a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na", "um", "uma",
         "para", "por", "com", "que", "se", "ao", "ou", "eu", "meu", "minha", "qual", "quais", "como"}


def tokenize(text: str) -> list[str]:
    toks = [t for t in _TOKEN.findall(normalize(text)) if t not in _STOP]
    # Crude plural/suffix folding so "devolucoes"/"devolucao" and "parcelas"/"parcela" match.
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in toks]


def load_chunks(path: Path | None = None) -> list[dict]:
    text = (path or DATA_DIR / "politicas.md").read_text(encoding="utf-8")
    chunks: list[dict] = []
    h2 = h3 = ""
    body: list[str] = []

    def flush():
        content = "\n".join(body).strip()
        if content:
            title = " > ".join(x for x in (h2, h3) if x)
            chunks.append({"section": title, "text": content})
        body.clear()

    for line in text.splitlines():
        if line.startswith("## "):
            flush(); h2, h3 = line[3:].strip(), ""
        elif line.startswith("### "):
            flush(); h3 = line[4:].strip()
        elif line.startswith("# "):
            continue
        else:
            body.append(line)
    flush()
    return chunks


class PolicyIndex:
    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        # Headings count double: they carry the topic of the chunk.
        self._docs = [Counter(tokenize(c["section"]) * 2 + tokenize(c["text"])) for c in self.chunks]
        self._len = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._len) / max(1, len(self._len))
        df: Counter = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q = tokenize(query)
        scored = []
        for i, doc in enumerate(self._docs):
            score = 0.0
            for t in q:
                f = doc.get(t, 0)
                if f:
                    score += self._idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * self._len[i] / self._avg))
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [{**self.chunks[i], "score": round(s, 3)} for s, i in scored[:top_k]]
