# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from Templatable import classify_and_context

class ClassifyRequest(BaseModel):
    question: str

class ClassifyResponse(BaseModel):
    topic: str
    context_chunks: list[str]

app = FastAPI(title="MLX Classifier API", version="1.0")

@app.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(req: ClassifyRequest):
    topic, ctx, chunks = classify_and_context(req.question, top_k=6)
    chunks = [c for c in chunks if c and c.strip()]
    return ClassifyResponse(topic=topic, context_chunks=chunks)
