from openai import OpenAI
from dotenv import load_dotenv

import chromadb
import pandas as pd


load_dotenv()

client = OpenAI()

chroma_client = chromadb.PersistentClient(r"C:\Users\Asus\Desktop\Study\Msc DA\Research\Artifact\vector_db\chroma_hs_codes")
hs_collection = chroma_client.get_collection("hs_codes")

df_hs = pd.read_csv("./dataset/hs_code_data.csv")

df_hs_clean = df_hs["comm_code","commodity"]

df_hs_test = df_hs.sample(n = 2000, random_state = 42)

df_hs_remain = df_hs_clean.drop(df_hs_test.index)

df_hs_validate = df_hs_remain.sample(n = 2000,random_state = 42)

df_hs_remain = df_hs_remain.drop(df_hs_validate.index)