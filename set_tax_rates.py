# imports

import os
import glob
from dotenv import load_dotenv
import gradio as gr
import pandas as pd

# imports for langchain
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


MODEL = "gpt-4o-mini"
db_name = "vector_db"


load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

loader1 = CSVLoader(file_path="dataset/VAT Rates in Europe  EU VAT Rates by Country.csv")
documents = loader1.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(len(chunks))

embeddings = OpenAIEmbeddings()

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()


vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")


collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

def get_vat_rate(country: str):
    question = f"What is the VAT rate for {country}?"
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

sales_df = pd.read_csv("dataset\estat_sts_trtu_m_en.csv")

sales_df = sales_df.head(100)

vat_rates = []
for country in sales_df["geo"]:
    answer = get_vat_rate(country)
    
    import re
    match = re.search(r"(\d+(\.\d+)?)", answer)
    vat_rate = float(match.group(1)) if match else None
    vat_rates.append(vat_rate)

sales_df["vat_rate"] = vat_rates
sales_df["amount_with_vat"] = sales_df["OBS_VALUE"] * (1 + sales_df["vat_rate"]/100)

sales_df.to_csv("sales_with_vat.csv", index=False)
print("Sales dataset enriched with VAT and saved as sales_with_vat.csv")