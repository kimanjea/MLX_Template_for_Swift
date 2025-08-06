

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


def classify(text):
    out = clf(text)[0]
    label_id = int(out["label"].split("_")[-1]) if out["label"].startswith("LABEL_") else out["label"]
    return "on-topic" if label_id in (1, "1") else "off topic"


def ask(question: str) -> str:
    if(question):
        if(classify(question)=="on-topic"):
            print("on-topic")
            if question:
                context_chunks = retrieve_context(question, docs, embeddings, embedder)
                context_text = "\n".join(context_chunks)

                messages = [
                {
                    "role": "user",
                    "content": f"""You are an expert data activism and programming tutor for high school students.

                    Your role is to provide step-by-step hints to guide students in completing coding and data visualization tasks. You do not write full solutions or complete the code for them. Instead, you offer clear, concise, and detailed hints for only the next step they should take, focusing on replacing any placeholder question marks (?) in their code.

                    Always ensure:
                    - Your responses are limited to topics related to data activism, coding, and data visualization.
                    - You avoid answering any questions unrelated to these topics.
                    - Your hints are detailed enough for the student to take confident action but never include the full solution.
                    - If a student asks about definitions or programming concepts related to data activism, provide direct and clear explanations.
                    - Encourage students to ask more specific questions about data activism if they ask general or unrelated questions.
                    Respond using no more than 5 clear and precise sentences per hint. Do not include any code snippets, code blocks, or print statements in your response unless explicitly asked for code.

                    Context:
                    {context_text}

                    Task:
                    Question: {question}"""
                        }
                    ]

                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

                ###########################################
                # STEP 7: Generate response using MLX
                ###########################################


                response_text = ""
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=1024,
                    #sampler=make_sampler(temp=0.0, top_p=1.0),
                    prompt_cache=prompt_cache
                ):
                    response_text += response.text
                save_prompt_cache(cache_file, prompt_cache)
                return response_text
            
        else:
                print("off-topic")
                messages = [
                {
                    "role": "user",
                    "content": f"""You are an expert data‑activism and programming tutor for high‑school students.
                        You only discuss coding, data visualization and other topics directly related to data activism.
                        If the student asks anything off topic, respond in exactly two sentences:
                        1) remind them that we’re focusing on data‑activism tasks;
                        2) invite them to ask a new question about data activism.
                        Use the conversation history to guide the student back to asking questions more on topic questions as they previously talked about.
                    Task:
                    Question: {"student just asked an off topic question, guide them back to data activism?"}"""
                        }
                    ]

                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

                ###########################################
                # STEP 7: Generate response using MLX
                ###########################################


                response_text = ""
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=1024,
                    prompt_cache=prompt_cache
                    #sampler=make_sampler(temp=0.0, top_p=1.0),
                    #prompt_cache=prompt_cache,
                ):
                    response_text += response.text
                return response_text






