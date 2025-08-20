from fastapi import FastAPI
from pydantic import BaseModel
from Templatable import classify_and_context  # <- import the new helper
import uvicorn

class ClassifyRequest(BaseModel):
    question: str

class ClassifyResponse(BaseModel):
    topic: str
    context_text: str   # NEW

app = FastAPI(title="MLX Classifier API", version="1.0")

@app.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(req: ClassifyRequest):
    topic, ctx, _ = classify_and_context(req.question, top_k=6)
    return ClassifyResponse(topic=topic, context_text=ctx)

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
