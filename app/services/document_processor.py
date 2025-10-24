# app/services/document_processor.py
import tempfile
import os
from app.services.summarizer import summarize_docs
from app.services.vectorstore import store_in_vectorstore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
import asyncio

async def process_document(file, metadata):
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Load document
        if tmp_path.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif tmp_path.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Process summarization and vectorstore storage in parallel
        async def summarize_task():
            return summarize_docs(docs)
        
        async def store_task():
            store_in_vectorstore(chunks, metadata)
            return True

        # Run both tasks concurrently
        summary, _ = await asyncio.gather(summarize_task(), store_task())

        return {"summary": summary}
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass