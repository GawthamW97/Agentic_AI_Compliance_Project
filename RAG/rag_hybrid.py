from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = OpenAI()

chroma_client = chromadb.PersistentClient(r"C:\Users\Asus\Desktop\Study\Msc DA\Research\Artifact\Agentic_AI_Compliance_Project\vector_db\chroma_hs_codes_hierarchy")

hs_collection = chroma_client.get_collection("hs_codes_hierarchy")

all_docs = pd.DataFrame({
    "id": hs_collection.get(include=["documents"])["ids"],
    "doc": hs_collection.get(include=["documents"])["documents"]
})
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(all_docs["doc"])


def hybrid_retrieve(query_text, k=5, alpha=0.7):
    query_emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=query_text
    ).data[0].embedding

    semantic_result = hs_collection.query(
        query_embeddings=[query_emb],
        n_results=min(100, len(all_docs))
    )
    sem_ids = semantic_result["ids"][0]
    sem_docs = semantic_result["documents"][0]
    sem_scores = [1 - d for d in semantic_result["distances"][0]]

    sem_df = pd.DataFrame({
        "id": sem_ids,
        "semantic_score": sem_scores,
        "doc": sem_docs
    })

    query_tfidf = tfidf.transform([query_text])
    lexical_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    lex_df = pd.DataFrame({
        "id": all_docs["id"],
        "lexical_score": lexical_scores
    })

    combined = pd.merge(sem_df, lex_df, on="id", how="outer").fillna(0)
    combined["hybrid_score"] = alpha * combined["semantic_score"] + (1 - alpha) * combined["lexical_score"]
    combined = combined.sort_values("hybrid_score", ascending=False).head(k)

    combined.reset_index(drop=True, inplace=True)
    combined["rank"] = combined.index + 1

    results = combined[["rank","id", "doc", "semantic_score", "lexical_score", "hybrid_score"]].to_dict(orient="records")
    return results


