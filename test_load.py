from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

loader = CSVLoader(file_path="dataset/sales_and_customer_insights.csv")
documents = loader.load()

print(f"Loaded {len(documents)} documents")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")

