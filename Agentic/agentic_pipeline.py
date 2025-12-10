from openai import OpenAI
from dotenv import load_dotenv
from agentic_default import setChromaClient,agentic_classify

def agentic(code_desc,K):
    setChromaClient("./vector_db/chroma_hs_codes")
    return agentic_classify(code_desc,collection="hs_codes",k=K)

# def agentic_hier(code_desc,K):
#     setChromaClient("./vector_db/chroma_hs_codes_hierarchy")
#     return agentic_classify(code_desc,collection="hs_codes_hierarchy",k=K)