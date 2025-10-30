from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

embeddings = OpenAIEmbeddings(
    openai_api_key=api_key,
    model="text-embedding-3-small"  # Faster and cheaper than default
)
vectorstore = Chroma(collection_name="knowledge_base", embedding_function=embeddings)

def store_in_vectorstore(chunks, metadata, batch_size=50):
    # Add metadata to each chunk
    for chunk in chunks:
        chunk.metadata.update(metadata)
    
    # Process in batches for better performance with large documents
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
