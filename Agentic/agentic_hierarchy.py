import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
import json
import time
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./vector_db")

embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

collections = {
    2: chroma_client.get_collection("hs_codes_2d", embedding_function=embed_fn),
    4: chroma_client.get_collection("hs_codes_4d", embedding_function=embed_fn),
    6: chroma_client.get_collection("hs_codes_6d", embedding_function=embed_fn),
}

def hierarchical_retrieval(
    product_desc,
    k2=3,   # chapters
    k4=5,   # headings per chapter
    k6=8    # subheadings per heading
):
    """
    Retrieve HS candidates hierarchically:
      Level 1: 2-digit chapters
      Level 2: 4-digit headings (filtered by parent_code)
      Level 3: 6-digit subheadings (filtered by parent_code)

    Returns a dict with:
      - chapters:   [(code, desc)]
      - headings:   [(code, desc, parent_chapter)]
      - subheadings:[(code, desc, parent_heading, parent_chapter)]
    """

    # --- Level 1: 2-digit chapters ---
    res2 = collections[2].query(
        query_texts=[product_desc],
        n_results=max(k2, 2)
    )
    chapters = [
        (cid, meta["description"])
        for cid, meta in zip(res2["ids"][0], res2["metadatas"][0])
    ]

    # --- Level 2: 4-digit headings ---
    headings = []
    for ch_id, _ in chapters:
        res4 = collections[4].query(
            query_texts=[product_desc],
            where={"parent_code": ch_id},
            n_results=max(k4, 2)
        )
        for hid, meta in zip(res4["ids"][0], res4["metadatas"][0]):
            headings.append((hid, meta["description"], ch_id))

    # --- Level 3: 6-digit subheadings ---
    subheadings = []
    # limit branching: only expand first k4 headings total
    for hd_id, _, parent_ch in headings[:k4]:
        res6 = collections[6].query(
            query_texts=[product_desc],
            where={"parent_code": hd_id},
            n_results=max(k6, 2)
        )
        for sid, meta in zip(res6["ids"][0], res6["metadatas"][0]):
            subheadings.append((sid, meta["description"], hd_id, parent_ch))

    return {
        "chapters": chapters,
        "headings": headings,
        "subheadings": subheadings
    }

def agent_reason(product_desc, tree):
    """
    LLM reasoning on top of hierarchical retrieval.
    Prompt is trimmed to avoid huge contexts.
    """

    # Limit candidates passed to the model
    chapters = tree["chapters"][:3]
    headings = tree["headings"][:5]

    # Deduplicate subheading codes while preserving order
    seen = set()
    sub_clean = []
    for sid, desc, hd, ch in tree["subheadings"]:
        if sid not in seen:
            sub_clean.append((sid, desc, hd, ch))
            seen.add(sid)
    subheadings = sub_clean[:10]

    chapters_str = "\n".join([f"- {cid}: {desc}" for cid, desc in chapters])
    headings_str = "\n".join([f"- {hid}: {desc}" for hid, desc, ch in headings])
    subs_str = "\n".join([f"- {sid}: {desc}" for sid, desc, hd, ch in subheadings])

    prompt = f"""
You are a senior customs classification expert.

Classify the following product into the most appropriate 6-digit HS code.

PRODUCT:
"{product_desc}"

CANDIDATE HIERARCHY:

CHAPTERS (2-digit):
{chapters_str}

HEADINGS (4-digit):
{headings_str}

SUBHEADINGS (6-digit):
{subs_str}

TASK:
1. Choose a single best 6-digit HS code from the subheading list above.
2. Justify your choice based on the hierarchy (chapter → heading → subheading).
3. Provide a confidence score between 0 and 1.

Respond in JSON ONLY:

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
        return {"final_code": None, "justification": "Invalid JSON", "confidence": 0.0}

def reflect_and_correct(
    product_desc,
    tree,
    first_answer,
    confidence_threshold=0.75,
    max_reflections=1
):
    """
    Simple reflection loop:
      - If confidence < threshold, reflect once using the same candidates.
      - Otherwise, accept the initial answer.
    """

    answer = first_answer

    if answer.get("final_code") is None:
        # Try at least once if initial parsing failed
        low_conf = True
    else:
        low_conf = answer.get("confidence", 0.0) < confidence_threshold

    if not low_conf or max_reflections <= 0:
        return answer

    # Build candidate code list
    sub_codes = []
    for sid, _, _, _ in tree["subheadings"]:
        if sid not in sub_codes:
            sub_codes.append(sid)

    if not sub_codes:
        return answer

    # --- Single reflection attempt ---
    reflect_prompt = f"""
You previously classified this product with low confidence.

PRODUCT:
"{product_desc}"

YOUR PREVIOUS ANSWER:
{json.dumps(answer)}

VALID 6-DIGIT CANDIDATES:
{sub_codes}

Reflect on your previous reasoning and choose the BEST code from the candidate list ONLY.
Return JSON ONLY in the format:

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

def agentic_hs_classifier(
    product_desc,
    k2=3,
    k4=5,
    k6=8,
    confidence_threshold=0.75
):
    """
    Full Agentic HS classification:
      1. Hierarchical retrieval
      2. LLM reasoning
      3. Optional reflection
    """

    tree = hierarchical_retrieval(product_desc, k2=k2, k4=k4, k6=k6)
    initial_answer = agent_reason(product_desc, tree)
    final_answer = reflect_and_correct(
        product_desc,
        tree,
        initial_answer,
        confidence_threshold=confidence_threshold,
        max_reflections=1  # keep latency under control
    )

    return {
        "product": product_desc,
        "retrieved_tree": tree,
        "initial_answer": initial_answer,
        "final_answer": final_answer
    }

