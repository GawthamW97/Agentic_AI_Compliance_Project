from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()
# Load HS dataset
df = pd.read_csv("./dataset/harmonized-system.csv")

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./vector_db/")

# Set up embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-3-large",
        api_key = os.environ["OPENAI_API_KEY"]
)

# Create collections for each level
collections = {
    2: chroma_client.get_or_create_collection(name="hs_codes_2d", embedding_function=openai_ef),
    4: chroma_client.get_or_create_collection(name="hs_codes_4d", embedding_function=openai_ef),
    6: chroma_client.get_or_create_collection(name="hs_codes_6d", embedding_function=openai_ef)
}

# Helper to build metadata with hierarchy
def build_metadata(row):
    return {
        "hscode": row["hscode"],
        "description": row["description"],
        "level": row["level"],
        "section": row["section"],
        "parent_code": str(row["parent"]) if "parent" in row else None,
        "full_path": f"{row['section']} → {row['parent']} → {row['hscode']}"
    }

# Embed and insert per level
for level in [2, 4, 6]:
    subset = df[df["level"] == level]
    col = collections[level]
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Embedding level {level}"):
        text = f"HS Code {row['hscode']}: {row['description']}"
        metadata = build_metadata(row)
        col.add(
            documents=[text],
            metadatas=[metadata],
            ids=[str(row["hscode"])]
        )

print("Hierarchical embeddings stored in Chroma.")
