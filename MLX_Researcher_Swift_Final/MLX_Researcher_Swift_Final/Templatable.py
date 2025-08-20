# temp.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
import csv
import sys

# -------------------------------
# Load documents & embeddings
# -------------------------------
import pdfplumber

pdf_path = "Final_Activity_v1.pdf"
text_chunks = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            text_chunks.append(text)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
docs = splitter.create_documents(text_chunks)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_texts = [doc.page_content for doc in docs]
embeddings = embedder.encode(doc_texts)

def retrieve_context(question, top_k=1):
    q_emb = embedder.encode([question])[0]
    sims = np.dot(embeddings, q_emb)
    top_k_idx = sims.argsort()[-top_k:][::-1]
    return [docs[i].page_content for i in top_k_idx]

# -------------------------------
# Load classifier
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
classifier_model_path = os.path.join(BASE_DIR, "data_activism_classifier")
clf_tokenizer = AutoTokenizer.from_pretrained(classifier_model_path)
clf_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_path)

clf = pipeline("text-classification", model=clf_model, tokenizer=clf_tokenizer)

# -------------------------------
# Classify function
# -------------------------------
def classify(text: str) -> str:
    out = clf(text)[0]
    label_id = int(out["label"].split("_")[-1]) if out["label"].startswith("LABEL_") else out["label"]
    topic = "on-topic" if label_id in (1, "1") else "off-topic"

    # ✅ print like the old ask() did
    print(topic, file=sys.stdout, flush=True)

    return topic

# -------------------------------
# Optional: CSV logging
# -------------------------------
csv_log_path = os.path.join(BASE_DIR, "conversation_log.csv")

def log_conversation(question, response, topic):
    file_exists = os.path.isfile(csv_log_path)
    with open(csv_log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['user', 'assistant', 'topic', 'timestamp'])
        writer.writerow([question, response, topic, datetime.now().isoformat()])

def is_scaffold(q: str) -> bool:
    return ("?" in q) and ("df[" in q or "groupby" in q.lower())

def classify_and_context(question: str, top_k: int = 6):
    """
    - classify
    - if on-topic: retrieve top_k chunks; if NOT scaffold, drop any chunk with '?'
    - join chunks into context_text
    """
    topic = classify(question)  # (prints "on-topic"/"off-topic")
    if topic == "on-topic":
        chunks = retrieve_context(question, top_k=top_k)
        if not is_scaffold(question):
            chunks = [c for c in chunks if "?" not in c]
        context_text = "\n".join(chunks)
    else:
        chunks = []
        context_text = ""
    return topic, context_text, chunks
