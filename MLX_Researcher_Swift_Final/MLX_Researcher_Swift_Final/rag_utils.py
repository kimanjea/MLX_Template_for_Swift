#
//  rag_utils.py
//  MLX_Researcher_Swift_Final
//
//  Created by AVLA Student on 8/26/25.
//


# rag_utils.py
import re
import numpy as np
from typing import List, Tuple

CODEY_LINE = re.compile(r"(```|pd\.|df\[|\.plot\(|=\s*[^=]|:\s*$|\)\s*$|^import\s+|^from\s+)", re.I)

def keyword_score(q: str, sent: str) -> int:
    q_tokens = {w.lower() for w in re.findall(r"[a-z0-9]+", q)}
    s_tokens = {w.lower() for w in re.findall(r"[a-z0-9]+", sent)}
    return len(q_tokens & s_tokens)

def split_sentences(text: str) -> List[str]:
    # cheap splitter that also handles line-based chunks
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def compress_context(question: str, chunks: List[str], is_scaffold: bool, max_chars: int = 800) -> str:
    kept_sents: List[str] = []

    for ch in chunks:
        # drop code-like lines for non-scaffold Q&A
        if not is_scaffold:
            lines = [ln for ln in ch.splitlines() if not CODEY_LINE.search(ln) and "?" not in ln]
            ch = " ".join(lines).strip()
            if not ch:
                continue

        # sentence score by keyword overlap; keep top few
        sents = split_sentences(ch)
        scored = sorted(((keyword_score(question, s), s) for s in sents), key=lambda x: x[0], reverse=True)
        # take the best 2 sentences per chunk
        for score, s in scored[:2]:
            if score > 0:
                kept_sents.append(s)

        if len(" ".join(kept_sents)) >= max_chars:
            break

    # final trim
    ctx = " ".join(kept_sents)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars].rsplit(" ", 1)[0] + "…"
    return ctx
