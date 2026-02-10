from dotenv import load_dotenv
from rag_default import classify_code_without_llm, classify_with_llm
load_dotenv()


def rag_only(code_desc, K):
    return classify_code_without_llm(code_desc, collection="hs_codes", k=K)


def rag_only_hierarchy(code_desc, K):
    return classify_code_without_llm(code_desc, collection="hs_codes_hierarchy", k=K)


def rag_with_llm(code_desc, K):
    return classify_with_llm(code_desc, collection="hs_codes", k=K)


def rag_with_llm_hierarchy(code_desc, K):
    return classify_with_llm(code_desc, collection="hs_codes_hierarchy", k=K)
