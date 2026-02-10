import sys

try:
    import pysqlite3 as sqlite3  # type: ignore
    sys.modules["sqlite3"] = sqlite3
except Exception:
    pass

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from RAG.rag_pipeline import rag_only
from RAG.rag_default import setChromaClient as set_rag_chroma
from Agentic.agentic_pipeline import agentic
from Agentic.agentic_default import setChromaClient as set_agentic_chroma

app = FastAPI(title="HS Code Classification API")

class Request(BaseModel):
    description: str
    k: int = 20

@app.on_event("startup")
def startup_event():
    load_dotenv()
    chroma_path = os.getenv("CHROMA_PATH", "./vector_db/chroma_hs_codes")
    set_rag_chroma(chroma_path)
    set_agentic_chroma(chroma_path)

@app.post("/classify/rag")
def classify_rag(req: Request):
    return rag_only(req.description, k=req.k)

@app.post("/classify/agentic")
def classify_agentic(req: Request):
    return agentic(req.description, k=req.k)
