from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import os
import re
from rag_default import classify_code_without_llm,classify_with_llm,setChromaClient
load_dotenv()
client = OpenAI()

def rag_only(code_desc):
    setChromaClient(r"C:\Users\Asus\Desktop\Study\Msc DA\Research\Artifact\Agentic_AI_Compliance_Project\vector_db\chroma_hs_codes")
    return classify_code_without_llm(code_desc,collection="hs_codes")

def rag_with_llm(code_desc):
    setChromaClient(r"C:\Users\Asus\Desktop\Study\Msc DA\Research\Artifact\Agentic_AI_Compliance_Project\vector_db\chroma_hs_codes_hierarchy")
    return classify_with_llm(code_desc,collection="hs_codes_hierarchy")
