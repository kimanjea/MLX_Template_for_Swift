# temp.py
import os
import sys
import csv
import re
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -------------------------------
# Load documents & embeddings
# -------------------------------
pdf_path = "Final_Activity_v1.pdf"
text_pages: List[str] = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            text_pages.append(text)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
docs = splitter.create_documents(text_pages)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_texts = [d.page_content for d in docs]
# cosine-ready (L2-normalized) embeddings
embeddings = embedder.encode(doc_texts, normalize_embeddings=True)

def retrieve_context(question: str, top_k: int = 6) -> List[Tuple[str, float]]:
    q_emb = embedder.encode([question], normalize_embeddings=True)[0]
    sims = np.dot(embeddings, q_emb)  # cosine similarity
    top_k_idx = sims.argsort()[-top_k:][::-1]
    return [(docs[i].page_content, float(sims[i])) for i in top_k_idx]

# -------------------------------
# Classifier
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
classifier_model_path = os.path.join(BASE_DIR, "data_activism_classifier")
clf_tokenizer = AutoTokenizer.from_pretrained(classifier_model_path)
clf_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_path)
clf = pipeline("text-classification", model=clf_model, tokenizer=clf_tokenizer)

def classify(text: str) -> str:
    out = clf(text)[0]
    label_id = int(out["label"].split("_")[-1]) if out["label"].startswith("LABEL_") else int(out["label"])
    # NOTE: flip to (0, "0") if your dataset uses 0 = on-topic
    topic = "on-topic" if label_id in (1, ) else "off-topic"
    print(topic, file=sys.stdout, flush=True)
    return topic

# -------------------------------
# CSV logging (optional)
# -------------------------------
csv_log_path = os.path.join(BASE_DIR, "conversation_log.csv")

def log_conversation(question, response, topic):
    file_exists = os.path.isfile(csv_log_path)
    with open(csv_log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['user', 'assistant', 'topic', 'timestamp'])
        writer.writerow([question, response, topic, datetime.now().isoformat()])

# -------------------------------
# RAG guards & compression
# -------------------------------
GENERIC_PAT = re.compile(r"^(what is|what's|define|explain|tell me about)\b", re.I)
CODEY_LINE = re.compile(r"(```|^from\s+|^import\s+|pd\.|df\[|\.plot\(|=\s*[^=]|:\s*$|\)\s*$)", re.I)

def is_generic_definition(q: str) -> bool:
    # Short, broad questions: "what is python", "define list", etc.
    return bool(GENERIC_PAT.search(q.strip())) and len(q.split()) <= 8

def is_scaffold(q: str) -> bool:
    # Only true if the USER MESSAGE contains literal ? placeholders AND looks like code
    looks_like_code = bool(re.search(r"(df\[|pd\.|\.plot|\=|```)", q)) or "\n" in q
    return ("?" in q) and looks_like_code

def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def keyword_score(q: str, sent: str) -> int:
    q_tokens = {w.lower() for w in re.findall(r"[a-z0-9]+", q)}
    s_tokens = {w.lower() for w in re.findall(r"[a-z0-9]+", sent)}
    return len(q_tokens & s_tokens)

def compress_context(question: str, raw_chunks: List[str], scaffold: bool, max_chars: int = 800) -> str:
    kept: List[str] = []
    for ch in raw_chunks:
        # For non-scaffold Q&A, strip out code-like lines and any "?" lines
        if not scaffold:
            lines = [ln for ln in ch.splitlines() if not CODEY_LINE.search(ln) and "?" not in ln]
            ch = " ".join(lines).strip()
            if not ch:
                continue

        sents = split_sentences(ch)
        # keep top 2 sentences per chunk by keyword overlap
        scored = sorted(((keyword_score(question, s), s) for s in sents), key=lambda x: x[0], reverse=True)[:2]
        for score, s in scored:
            if score > 0:
                kept.append(s)

        if len(" ".join(kept)) >= max_chars:
            break

    ctx = " ".join(kept).strip()
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars].rsplit(" ", 1)[0] + "…"
    return ctx

# -------------------------------
# Main RAG entry
# -------------------------------
def classify_and_context(question: str,
                         top_k: int = 6,
                         min_sim: float = 0.40) -> tuple[str, str, list[str]]:
    """
    Returns:
      topic: "on-topic" | "off-topic"
      context_text: compressed, non-codey support text ('' if none)
      chunks: [context_text] or []   (Swift expects a list of strings)
    """
    topic = classify(question)
    scaffold = is_scaffold(question)

    context_text = ""
    chunks: List[str] = []

    if topic == "on-topic" and not is_generic_definition(question):
        results = retrieve_context(question, top_k=top_k)  # [(text, score)]
        kept = [(t, s) for (t, s) in results if s >= min_sim]

        # For very short questions, demand a stronger match for RAG
        if len(question.split()) <= 8 and kept and kept[0][1] < 0.50:
            kept = []

        raw_chunks = [t for (t, _) in kept]

        # Final compression & cleaning so the context is supportive, not a dump
        context_text = compress_context(question, raw_chunks, scaffold, max_chars=800)
        if context_text:
            chunks = [context_text]  # single supportive blob

        print(f"[RAG] q={question!r} retrieved={len(results)} kept={len(kept)} inject={'yes' if chunks else 'no'}")
        for i, (_t, s) in enumerate(kept[:3]):  # log top-3 scores
            print(f"[RAG] kept#{i+1} score={s:.3f}")
    else:
        print(f"[RAG] q={question!r} -> no context (topic={topic}, generic={is_generic_definition(question)})")

    return topic, context_text, chunks
