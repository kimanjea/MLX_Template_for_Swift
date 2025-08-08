

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


pdf_path = "Final_Activity.pdf"

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
tokenizer = AutoTokenizer.from_pretrained(
    "ShukraJaliya/BLUECOMPUTER.2",
    trust_remote_code=True,
)

BASE_DIR = os.path.dirname(__file__)
cache_file = os.path.join(BASE_DIR, "mistral_prompt.safetensors")
if os.path.exists(cache_file):
    prompt_cache = load_prompt_cache(cache_file)
else:
    prompt_cache = make_prompt_cache(model)



classifier_model_path = os.path.join(BASE_DIR, "data_activism_classifier")
clf_tokenizer = AutoTokenizer.from_pretrained(classifier_model_path)
clf_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_path)

clf = pipeline(
    "text-classification",
    model=clf_model,
    tokenizer=clf_tokenizer,
    return_all_scores=False
)


def classify(text):
    out = clf(text)[0]
    label_id = int(out["label"].split("_")[-1]) if out["label"].startswith("LABEL_") else out["label"]
    return "on-topic" if label_id in (1, "1") else "off topic"

max_tokens = 1000

def ask(question: str) -> str:
    if(question):
        if(classify(question)=="on-topic"):
            print("on-topic")
            if question:
                context_chunks = retrieve_context(question, docs, embeddings, embedder)
                context_text = "\n".join(context_chunks)
                print(context_text)
                
                tokenizer.add_special_tokens({
                    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
                })
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            """You are an expert who only teaches data activism and Python programming to K–12 students. "
                            You explain concepts step by step using clear, scaffolded language.
                            You never provide exact code solutions.
                            If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps.
                            For general programming questions (like \"What is a function?\"), give a full explanation with a short example, but do not solve specific problems.
                            If a student asks something unrelated or off-topic, politely redirect them to focus on data activism or Python programming.\n\n"""
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Here is the context:\n\n{context_text}\n\n Answer the task: {question}"
                    }
                ]


                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

                ###########################################
                # STEP 7: Generate response using MLX
                ###########################################
                
                response = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    verbose=True,
                    prompt_cache=prompt_cache,
                    max_tokens=max_tokens,
                )
                    
                save_prompt_cache(cache_file, prompt_cache)
                return response
            
        else:
                print("off-topic")
                tokenizer.add_special_tokens({
                    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
                })
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in data activism and Python programming for K–12 students.\n"
                    "If a student asks something off-topic or requests unrelated content, politely redirect them back to data activism or Python by asking:\n"
                    "When you redirect or answer follow-ups, keep your response to two concise sentences.\n"
                    "Explain the answer using the chat history to tie back their questions."
                        ),
                    },
                    {"role": "user",
                    "content": (
                    f"Student just asked an off-topic question here: {question}. Guide the student back on the topic of data activism using the system instructions and the on-topic conversation history"
                    )}
                ]

                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

                ###########################################
                # STEP 7: Generate response using MLX
                ###########################################
                
                response = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    verbose=True,
                    prompt_cache=prompt_cache,
                    max_tokens=max_tokens,
                )
                    
                return response






