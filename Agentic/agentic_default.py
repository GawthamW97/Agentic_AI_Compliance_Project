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

def retrieve_candidates(product_desc,collection, k=10):
    hs_collection = chroma_client.get_collection(collection)
    query_vector = client.embeddings.create(
        model="text-embedding-3-large",
        input=product_desc
    ).data[0].embedding

    result = hs_collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    hs_ids = result["ids"][0]
    docs = result["documents"][0]
    distances = result["distances"][0]
    similarities = [1 - d for d in distances]
 
    return [{"rank": i+1, "hs_code": hs_ids[i], "description": docs[i], "similarity": similarities[i]} 
            for i in range(len(hs_ids))]

def reasoning_module(product_desc, candidates):
    context = "\n".join([f"{c['rank']}. {c['hs_code']}: {c['description']} (similarity: {c['similarity']:.4f}" for i, c in enumerate(candidates)])
    prompt = f"""
    You are a customs classification expert.
    Product description: "{product_desc}"
    Candidate HS codes:
    {context}
    
    Also, use your own semantic understandings and knowledge of Harmonized System Codes to
    assign the correct code.

    Select the single most appropriate HS code and briefly explain why.
    Respond with plain JSON only, no code blocks or additional text:
    {{ "selected_code": "xxxxx", "justification": "..." }}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        output = json.loads(resp.choices[0].message.content)
    except:
        output = {"selected_code": "UNKNOWN", "justification": resp.choices[0].message.content}
    return output

def validate_classification(result, candidates):
    """
    Deterministic checks (you can expand with real compliance rules).
    Example rules:
      - HS code must exist in candidates.
      - Check that product semantics align with HS section (Animals, Machines, etc.).
    """
    code = result["selected_code"]
    if code not in [c["hs_code"] for c in candidates]:
        return False, "Code not in retrieved candidates."
    # if code.startswith("95") and "animal" in result["justification"].lower():
    #     return False, "Conflict between 'animal' product and 'toy' HS section."
    return True, "Valid classification."

def reflective_correction(product_desc, candidates, prev_result, error_msg):
    """
    Ask the LLM to self-reflect and correct if validation failed.
    """
    reflection_prompt = f"""
    The previous classification attempt failed validation:
    Error: {error_msg}
    Product description: "{product_desc}"
    Candidates:
    {json.dumps(candidates, indent=2)}
    Previous justification: "{prev_result['justification']}"

    Also, use your own semantic understandings and knowledge of Harmonized System Codes to
    assign the correct code. 

    Reflect on your reasoning and provide a corrected HS code and justification.
    Respond with plain JSON only, no code blocks or additional text:
    {{ "selected_code": "xxxxx", "justification": "..." }}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": reflection_prompt}],
        temperature=0.2
    )
    try:
        output = json.loads(resp.choices[0].message.content)
    except:
        output = {"selected_code": "UNKNOWN", "justification": resp.choices[0].message.content}
    return output

def agentic_classify(product_desc,collection):
    start = time.time()
    candidates = retrieve_candidates(product_desc,collection)
    reasoning = reasoning_module(product_desc, candidates)
    is_valid, validation_msg = validate_classification(reasoning, candidates)

    if not is_valid:
        print(f"Validation failed: {validation_msg}. Triggering reflection...")
        reasoning = reflective_correction(product_desc, candidates, reasoning, validation_msg)
        is_valid, validation_msg = validate_classification(reasoning, candidates)

    end = time.time()

    return {
        "product_desc": product_desc,
        "final_code": reasoning["selected_code"],
        "justification": reasoning["justification"],
        "validation": validation_msg,
        "elapsed_sec": round(end - start, 2)
    }
