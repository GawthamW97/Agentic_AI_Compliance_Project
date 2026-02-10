from agentic_default import agentic_classify


def agentic(code_desc, k):
    return agentic_classify(code_desc, collection="hs_codes", k=k)

# def agentic_hier(code_desc,K):
#     setChromaClient("./vector_db/chroma_hs_codes_hierarchy")
#     return agentic_classify(code_desc,collection="hs_codes_hierarchy",k=K)
