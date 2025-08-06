

from mlx_lm import load, generate
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import stream_generate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import accelerate
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import pipeline
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from huggingface_hub import login
import textwrap
import sys
import os
from typing import List


pdf_path = "Data_Activisim_Piechart_Activity.pdf"

text_chunks = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            text_chunks.append(text)

###########################################
# STEP 2: Split text into manageable chunks
###########################################

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50
)

docs = splitter.create_documents(text_chunks)

###########################################
# STEP 3: Embed chunks for retrieval
###########################################

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_texts = [doc.page_content for doc in docs]
embeddings = embedder.encode(doc_texts)

###########################################
# STEP 4: Define retrieval function
###########################################

def retrieve_context(question, docs, embeddings, embedder, top_k=1):
    q_emb = embedder.encode([question])[0]
    similarities = np.dot(embeddings, q_emb)
    top_k_idx = similarities.argsort()[-top_k:][::-1]
    return [docs[i].page_content for i in top_k_idx]


model, tokenizer = load ("ShukraJaliya/BLUECOMPUTER.2")

BASE_DIR = os.path.dirname(__file__)
cache_file = os.path.join(BASE_DIR, "mistral_prompt.safetensors")
if os.path.exists(cache_file):
    prompt_cache = load_prompt_cache(cache_file)
else:
    prompt_cache = make_prompt_cache(model)

#question = f""" how can i make a data visualization """


classifier_model_path = os.path.join(BASE_DIR, "data_activism_classifier")
clf_tokenizer = AutoTokenizer.from_pretrained(classifier_model_path)
clf_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_path)

clf = pipeline(
    "text-classification",
    model=clf_model,
    tokenizer=clf_tokenizer,
    return_all_scores=False
)

# ✅ Define classify before ask()
def classify(text):
    out = clf(text)[0]
    label_id = int(out["label"].split("_")[-1]) if out["label"].startswith("LABEL_") else out["label"]
    return "on-topic" if label_id in (1, "1") else "off-topic"

# ✅ Now define ask
def ask(question: str) -> str:
    if not question:
        return "Please provide a question."

    is_scaffold = "?" in question and ("df[" in question or "groupby" in question)
    is_on_topic = classify(question) == "on-topic"

    if is_on_topic:
        print("on-topic")
        context_chunks = retrieve_context(question, docs, embeddings, embedder)
        if not is_scaffold:
            context_chunks = [c for c in context_chunks if "?" not in c]
        context_text = "\n".join(context_chunks)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert who only teaches data activism and Python programming to K–12 students. "
                    "You explain concepts step by step using clear, scaffolded language. "
                    "You never provide exact code solutions. "
                    "If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps. "
                    "For general programming questions (like \"What is a function?\"), give a full explanation with a short example, but do not solve specific problems. "
                    "If a student asks something unrelated or off-topic, politely redirect them to focus on data activism or Python programming.\n\n"
                    f"Context:\n{context_text}"
                ),
            },
            {"role": "user", "content": question},
        ]

    else:
        print("off-topic")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert who only teaches data activism and Python programming to K–12 students. "
                    "You explain concepts step by step using clear, scaffolded language. "
                    "You never provide exact code solutions. "
                    "If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps. "
                    "For general programming questions (like \"What is a function?\"), give a full explanation with a short example, but do not solve specific problems. "
                    "If a student asks something unrelated or off-topic, politely redirect them to focus on data activism or Python programming."
                ),
            },
            {"role": "user", "content": question},
        ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    response_text = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=1024,
        prompt_cache=None,
    ):
        response_text += response.text

    save_prompt_cache(cache_file, prompt_cache)
    return response_text



