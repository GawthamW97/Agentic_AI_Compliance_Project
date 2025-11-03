from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import os
import re

load_dotenv()
client = OpenAI()
chroma_client = chromadb.PersistentClient(r"C:\Users\Asus\Desktop\Study\Msc DA\Research\Artifact\Agentic_AI_Compliance_Project\vector_db\chroma_hs_codes_hierarchy")

def get_hs_from_db(code_desc, k=5, collection="hs_codes_hierarchy"):
    hs_collection = chroma_client.get_collection(collection)
    query_vector = client.embeddings.create(
        model="text-embedding-3-large",
        input=code_desc
    ).data[0].embedding

    result = hs_collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    return result["ids"][0], result["metadatas"][0], result["documents"][0], result["distances"][0]

def classify_with_llm(code_desc, candidates, model="gpt-4o-mini"):
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
        temperature=0.0
    )

    raw_output = response.output_text.strip()
    match = re.search(r'\d+', raw_output)
    final_code = match.group(0) if match else "UNCERTAIN"

    return final_code, raw_output

def classify_code(code_desc):
    hs_ids, metas, docs,distances = get_hs_from_db(code_desc)
    print("Code Count - ",len(metas))

    context  ="\n".join([f"{i+1}.{id}: {desc} - distance{distance}" for i,(id,desc,distance) 
                         in enumerate (zip(hs_ids,docs,distances))])
    prompt = f"""
    you are a customs classification assistant. 
    A Harmonized System code has been described for a product,
    Code Description : "{code_desc}"

    Based on the following HS codes and descriptions:
    {context}

    Return the most probable HS code and brief justification along with the distance.
    """

    completion = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.2
    )

    return completion.output_text

def classify_code_without_llm(code_desc, k=5, confidence_threshold=0.8, collection="hs_codes"):
    hs_ids, metas, docs, distances = get_hs_from_db(code_desc, k, collection)
    results = []
    for i in range(len(hs_ids)):
        similarity = 1 - distances[i]
        results.append({
            "rank": i + 1,
            "hs_code": hs_ids[i],
            "hs_description": docs[i],
            "similarity": round(similarity, 4)
        })

    return results

# print(classify_code("Hop cones; ground, powdered or in the form of pellets; lupulin"))
# print(classify_code_without_llm("XI - Singlets and other vests, bathrobes, dressing gowns and similar articles; men's or boys', of textile materials other than cotton (not knitted or crocheted) (HS Code: 620799)"))

