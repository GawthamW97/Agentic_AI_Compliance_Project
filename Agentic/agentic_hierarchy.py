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

def retrieve_6digit_first(product_desc, k6=20):
    # --- Step 1: Retrieve 6-digit candidates ---
    res6 = collections[6].query(
        query_texts=[product_desc],
        n_results=k6,
        include=["documents", "metadatas", "distances"]
    )

    six_candidates = []
    for sid, doc, meta, dist in zip(
        res6["ids"][0],
        res6["documents"][0],
        res6["metadatas"][0],
        res6["distances"][0]
    ):
        six_candidates.append({
            "hs6": sid,
            "desc6": meta.get("description", doc),
            "sim6": 1 - dist
        })

    # --- Step 2: Derive 4-digit parents ---
    unique_4 = sorted({c["hs6"][:4] for c in six_candidates})
    four_candidates = []
    for code4 in unique_4:
        try:
            res4 = collections[4].get([code4])
            if res4["metadatas"]:
                four_candidates.append({
                    "hs4": code4,
                    "desc4": res4["metadatas"][0]["description"]
                })
        except:
            pass  # ignore missing headings

    # --- Step 3: Derive 2-digit parents (chapters) ---
    unique_2 = sorted({c["hs6"][:2] for c in six_candidates})
    two_candidates = []
    for code2 in unique_2:
        try:
            res2 = collections[2].get([code2])
            if res2["metadatas"]:
                two_candidates.append({
                    "hs2": code2,
                    "desc2": res2["metadatas"][0]["description"]
                })
        except:
            pass

    return {
        "chapters": two_candidates,
        "headings": four_candidates,
        "subheadings": six_candidates
    }

def agent_reason(product_desc, tree, max6=10, max4=5, max2=3):
    """
    LLM reasoning on top of 6-digit-first hierarchical retrieval.
    """

    # trim lists for prompt safety
    subs = tree["subheadings"][:max6]
    heads = tree["headings"][:max4]
    chaps = tree["chapters"][:max2]

    chap_str = "\n".join([f"- {c['hs2']}: {c['desc2']}" for c in chaps])
    head_str = "\n".join([f"- {h['hs4']}: {h['desc4']}" for h in heads])
    sub_str = "\n".join([f"{i+1}. {s['hs6']} (sim={s['sim6']:.3f}): {s['desc6']}" 
                         for i, s in enumerate(subs)])

    prompt = f"""
            You are a senior customs classification expert.

            PRODUCT:
            \"{product_desc}\"

            We retrieved the TOP HS code candidates first (highest granularity).
            Based on hierarchical structure, their parent HEADINGS (4-digit) and CHAPTERS (2-digit)
            are derived below.

            2-DIGIT CHAPTERS:
            {chap_str}

            4-DIGIT HEADINGS:
            {head_str}

            6-DIGIT SUBHEADINGS (candidates):
            {sub_str}

            TASK:
            1. Choose the code from the list above. The code can only be of length 5 or 6.
            2. Justify the choice using: product meaning, heading consistency, and chapter relevance.
            3. Provide a confidence score between 0 and 1.
            4. Also, use your own semantic understandings and knowledge of Harmonized System Codes to
            assign the correct code. 
            5. Use the 2-DIGIT CHAPTERS and 4-DIGIT HEADINGS as a context to enrich the relevance 
            of the HS code for the description that required HS Code prediction.
            6. If there are any punctuations in the HS Code, remove the punctuation and persist the entire code.

            Respond in JSON only:

            {{
            "final_code": "XXXXXX",
            "justification": "…",
            "confidence": 0.0
            }}
            """

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.1
    )

    try:
        ans = json.loads(resp.output_text)
        final_code = ans.get("final_code")
        
        # If LLM returned invalid code → fallback
        if final_code is None or not final_code.isdigit():
            fallback_code = tree["subheadings"][0]["hs6"]
            return {
                "final_code": fallback_code,
                "justification": "Fallback to highest-ranked retrieval (invalid model output).",
                "confidence": 0.0,
            }
        
        return ans

    except Exception:
        # Hard fallback
        fallback_code = tree["subheadings"][0]["hs6"]
        return {
            "final_code": fallback_code,
            "justification": "Fallback due to invalid JSON.",
            "confidence": 0.0
        }
    

def reflect_and_correct(
    product_desc,
    tree,
    first_answer,
    confidence_threshold=0.75,
    max_reflections=1
):
    answer = first_answer

    def fallback():
        return {
            "final_code": tree["subheadings"][0]["hs6"],
            "justification": "Fallback after low-confidence reflection.",
            "confidence": 0.0
        }

    # If first answer has no code → fallback immediately
    if answer.get("final_code") is None or not str(answer["final_code"]).isdigit():
        return fallback()
    
    low_conf = (
        answer.get("final_code") is None or
        answer.get("confidence", 0.0) < confidence_threshold
    )

    if not low_conf or max_reflections <= 0:
        return answer

    sub_codes = [s["hs6"] for s in tree["subheadings"]]

    reflect_prompt = f"""
            You are an expert in classifying the Harmonized System Codes.
            You classified with low confidence.

            PRODUCT:
            \"{product_desc}\"

            PREVIOUS ANSWER:
            {json.dumps(answer)}

            VALID 6-DIGIT CANDIDATES:
            {sub_codes}

            1. Use your own semantic understandings and knowledge of Harmonized System Codes to
            assign the correct code. 
            2. Use the 2-DIGIT CHAPTERS and 4-DIGIT HEADINGS as a context to enrich the relevance 
            of the HS code for the description that required HS Code prediction.
            3. If there are any punctuations in the HS Code, remove the punctuation and persist the entire code.

            Reflect and choose the best option.
            Return JSON:

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
        new = json.loads(resp.output_text)
        code = new.get("final_code")

        # validate parsed code
        if code is None or not code.isdigit():
            return fallback()

        return new

    except:
        return fallback()
    

def agentic_hs_classifier(product_desc, k6=20, confidence_threshold=0.75):
    tree = retrieve_6digit_first(product_desc, k6=k6)

    initial_answer = agent_reason(product_desc, tree)
    
    final_answer = reflect_and_correct(
        product_desc,
        tree,
        initial_answer,
        confidence_threshold=confidence_threshold,
        max_reflections=1
    )

    # FINAL SAFETY
    if final_answer["final_code"] is None or not str(final_answer["final_code"]).isdigit():
        final_answer["final_code"] = tree["subheadings"][0]["hs6"]
        final_answer["confidence"] = 0.0
        final_answer["justification"] += " | Final fallback used."

    return {
        "product": product_desc,
        "retrieved_tree": tree,
        "initial_answer": initial_answer,
        "final_answer": final_answer
    }

