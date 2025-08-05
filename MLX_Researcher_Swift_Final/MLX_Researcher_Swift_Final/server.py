#
# server.py
#  MLX Researcher
#
#  Created by xrlead on 8/4/25.
#

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn
from Templatable import ask  # or import ask if you have a full RAG ask() function

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

app = FastAPI(
    title="MLX Template API",
    version="1.1"
)

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    return AskResponse(answer=ask(req.question))
    
    
if __name__ == "__main__":
    # Use 0.0.0.0 to allow network access; port 8000 matches your Swift client
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
