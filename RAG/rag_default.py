from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import os
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
global chroma_client

def setChromaClient(path):
    global chroma_client
    chroma_client = chromadb.PersistentClient(path)

def get_hs_candidates(code_desc, k=5,collection="hs_codes"):
    hs_collection = chroma_client.get_collection(collection)
    query_vector = client.embeddings.create(
        model="text-embedding-3-large",
        input=code_desc
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

def classify_with_llm(code_desc, collection,model="gpt-4o-mini",k=5):
    candidates = get_hs_candidates(code_desc,k,collection=collection)
    context = "\n".join([
        f"{c['rank']}. {c['hs_code']}: {c['description']} (similarity: {c['similarity']:.4f})"
        for c in candidates
    ])

    prompt = f"""
    You are a customs classification assistant.
    A product is described as: "{code_desc}".

    Based on the following HS code candidates:
    {context}

    Select the single most appropriate HS code that best matches the product description.
    Respond with ONLY the HS code number, nothing else.
    """

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.1
    )

    raw_output = response.output_text.strip()
    match = re.search(r'\d+', raw_output)
    final_code = match.group(0) if match else "UNCERTAIN"

    return final_code, raw_output,candidates

def classify_code_without_llm(code_desc, k=5, collection="hs_codes"):
    candidates = get_hs_candidates(code_desc, k, collection)
    results = []
    for c in candidates:
        results.append({
            "rank": c["rank"],
            "hs_code": c["hs_code"],
            "hs_description": c["description"],
            "similarity": c["similarity"]
        })
    return results

