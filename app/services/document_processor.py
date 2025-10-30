# app/services/document_processor.py
import tempfile
import os
import time
from app.services.summarizer import summarize_docs
from app.services.vectorstore import store_in_vectorstore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_document(file, metadata):
    start_time = time.time()
    
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

        # Optimize chunking - larger chunks for better performance
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased from 1000
            chunk_overlap=100,  # Reduced from 200
            separators=["\n\n", "\n", " ", ""]  # Better separators
        )
        chunks = splitter.split_documents(docs)

        # Optimize summarization - use only first few pages for large documents
        def get_summary_docs(docs, max_pages=5):
            if len(docs) <= max_pages:
                return docs
            return docs[:max_pages]  # Only summarize first 5 pages
        
        summary_docs = get_summary_docs(docs)

        # Run tasks in parallel with thread pool for CPU-bound operations
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks to thread pool
            summary_task = loop.run_in_executor(executor, summarize_docs, summary_docs)
            store_task = loop.run_in_executor(executor, store_in_vectorstore, chunks, metadata)
            
            # Wait for both to complete
            summary, _ = await asyncio.gather(summary_task, store_task)

        end_time = time.time()
        print(f"Document processing took {end_time - start_time:.2f} seconds")
        
        return {"summary": summary}
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass