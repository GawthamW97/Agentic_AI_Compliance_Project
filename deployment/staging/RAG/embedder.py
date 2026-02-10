import chromadb
import os
from openai import OpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

load_dotenv()
client  = OpenAI()

'''Create the embeddings of the HS Code and store it in ChromaDB'''
def create_embeddings(collection_name):

    client = chromadb.PersistentClient(path=f"./vector_db/chroma_{collection_name}")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-3-large",
        api_key=os.environ["OPENAI_API_KEY"]
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function = openai_ef
    )

    return collection

def build_hierarchy(row):
    hscode_clean = str(row['hscode']).strip("'")
    return f"Section {row['section']} - Chapter - {hscode_clean[:2]} - {row['description']}"

def create_hs_db(df,hierarchy = True):
    collection_name = "hs_codes_hierarchy"
    if hierarchy:
         df["text"] = df.apply(build_hierarchy,axis=1)
    else:
         df["text"] = df["description"]
         collection_name = "hs_codes"

    collection = create_embeddings(collection_name)

    if collection != None:
        for _,row in tqdm(df.iterrows(), total=len(df)):
                collection.add(
                    documents = [row["text"]],
                    metadatas = [row.to_dict()],
                    ids = [str(row["hscode"]).strip("'")]
                )

# read dataset that will be embedded completly into the Chroma database
df_hs = pd.read_csv("./dataset/harmonized-system.csv")

create_hs_db(df_hs,False)
# create_hs_db(df_hs,True)

