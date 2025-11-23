import json
import time
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from agentic_default import setChromaClient,agentic_classify


def agentic(code_desc):
    setChromaClient(r"C:\Users\Asus\Desktop\Study\Msc DA\Research\Artifact\Agentic_AI_Compliance_Project\vector_db\chroma_hs_codes")
    return agentic_classify(code_desc,collection="hs_codes")

def agentic_hier(code_desc):
    setChromaClient(r"C:\Users\Asus\Desktop\Study\Msc DA\Research\Artifact\Agentic_AI_Compliance_Project\vector_db\chroma_hs_codes_hierarchy")
    return agentic_classify(code_desc,collection="hs_codes_hierarchy")