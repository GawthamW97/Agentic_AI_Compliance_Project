import json
import time
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
global chroma_client

def setChromaClient(path):
    global chroma_client
    chroma_client = chromadb.PersistentClient(path)

def flat_rag_retrieve(product_desc, k_retrieval=30):
    """
    Flat RAG retrieval:
      - Single collection (hs_codes)
      - Semantic similarity only (Chroma)
      - Returns ordered candidates with similarity scores
    """
    setChromaClient(r"C:\Users\Asus\Desktop\Study\Msc DA\Research\Artifact\Agentic_AI_Compliance_Project\vector_db\chroma_hs_codes")
    hs_collection = chroma_client.get_collection("hs_codes")
    
    query_vector = client.embeddings.create(
        model="text-embedding-3-large",
        input=product_desc
    ).data[0].embedding
    
    res = hs_collection.query(
        query_embeddings=[query_vector],
        n_results=k_retrieval,
        include=["documents", "metadatas", "distances"]
    )

    # res = hs_collection.query(
    #     query_texts=[product_desc],
    #     n_results=k_retrieval,
    #     include=["documents", "metadatas", "distances"]
    # )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    candidates = []
    for i, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
        sim = 1 - dist  # convert distance -> similarity
        candidates.append({
            "rank": i + 1,
            "hs_code": meta.get("hscode", cid),
            "description": meta.get("description", doc),
            "similarity": sim
        })

    return candidates


def agent_reason_flat(product_desc, candidates, max_candidates_in_prompt=15):
    """
    LLM reasoning on top of flat RAG candidates.
    """

    # Trim candidate list to keep prompt reasonable
    trimmed = candidates[:max_candidates_in_prompt]

    cstr = "\n".join([
        f"{c['rank']}. {c['hs_code']} (sim={c['similarity']:.3f}): {c['description']}"
        for c in trimmed
    ])

    prompt = f"""
You are a senior customs classification expert.

A product needs to be classified into a 6-digit HS code.

PRODUCT DESCRIPTION:
"{product_desc}"

Here are the top candidate HS codes retrieved by a semantic search
(you MUST choose one of these codes):

CANDIDATES:
{cstr}

TASK:
1. Select the single best 6-digit HS code from the list.
2. Briefly justify your choice.
3. Provide a confidence score between 0 and 1 (higher = more confident).

Respond in JSON ONLY, in this exact format:

{{
  "final_code": "XXXXXX",
  "justification": "…",
  "confidence": 0.0
}}
"""

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.2
    )

    try:
        return json.loads(resp.output_text)
    except Exception:
        return {
            "final_code": None,
            "justification": "Invalid JSON or parsing error.",
            "confidence": 0.0
        }

def reflect_and_correct_flat(
    product_desc,
    candidates,
    first_answer,
    confidence_threshold=0.75,
    max_reflections=1
):
    """
    Reflection for flat RAG:
      - If confidence < threshold, reflect once.
      - Must choose from the same candidate hs_codes.
    """

    answer = first_answer

    if answer.get("final_code") is None:
        low_conf = True
    else:
        low_conf = answer.get("confidence", 0.0) < confidence_threshold

    if not low_conf or max_reflections <= 0:
        return answer

    candidate_codes = [c["hs_code"] for c in candidates]

    reflect_prompt = f"""
You previously classified this product with low confidence.

PRODUCT:
"{product_desc}"

YOUR PREVIOUS ANSWER:
{json.dumps(answer)}

VALID CANDIDATE HS CODES (you MUST pick from this list only):
{candidate_codes}

Reflect on your decision and either:
- confirm your previous HS code, OR
- choose a better HS code from the candidate list.

Return JSON ONLY:

{{
  "final_code": "XXXXXX",
  "justification": "…",
  "confidence": 0.0
}}
"""

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=reflect_prompt,
        temperature=0.1
    )

    try:
        new_answer = json.loads(resp.output_text)
        return new_answer
    except Exception:
        return answer

def agentic_hs_classifier_flat(
    product_desc,
    k_retrieval=30,
    confidence_threshold=0.75
):
    """
    Full agentic HS classifier using flat RAG:
      1. Retrieve top-K candidates from hs_codes collection.
      2. LLM selects best HS code.
      3. Optional reflection if confidence is low.
    """

    candidates = flat_rag_retrieve(product_desc, k_retrieval=k_retrieval)
    initial_answer = agent_reason_flat(product_desc, candidates)
    final_answer = reflect_and_correct_flat(
        product_desc,
        candidates,
        initial_answer,
        confidence_threshold=confidence_threshold,
        max_reflections=1
    )

    return {
        "product": product_desc,
        "candidates": candidates,
        "initial_answer": initial_answer,
        "final_answer": final_answer
    }
