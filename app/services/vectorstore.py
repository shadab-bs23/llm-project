from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = Chroma(collection_name="knowledge_base", embedding_function=embeddings)

def store_in_vectorstore(chunks, metadata):
    # Add metadata to each chunk
    for chunk in chunks:
        chunk.metadata.update(metadata)
    
    vectorstore.add_documents(chunks)
