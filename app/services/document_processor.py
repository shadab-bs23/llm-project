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

# Shared thread pool for all document processing (reuse threads)
_executor = ThreadPoolExecutor(max_workers=4)

async def process_document(file, metadata):
    start_time = time.time()
    
    # Stream file to disk in chunks (reduces memory usage)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp_path = tmp.name
            # Stream in 64KB chunks instead of loading entire file
            while chunk := await file.read(65536):
                tmp.write(chunk)

        # Load document
        if tmp_path.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif tmp_path.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
        docs = loader.load()

        # Larger chunks = fewer embeddings = faster processing
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,  # Increased from 2000
            chunk_overlap=50,  # Reduced from 100
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # Limit pages for summarization
        def get_summary_docs(docs, max_pages=5):
            return docs[:max_pages] if len(docs) > max_pages else docs
        
        summary_docs = get_summary_docs(docs)

        # Run summarization and vector storage in parallel using shared executor
        loop = asyncio.get_event_loop()
        summary_task = loop.run_in_executor(_executor, summarize_docs, summary_docs)
        store_task = loop.run_in_executor(_executor, store_in_vectorstore, chunks, metadata)
        
        # Wait for both to complete
        summary, _ = await asyncio.gather(summary_task, store_task)

        end_time = time.time()
        print(f"Document '{file.filename}' processed in {end_time - start_time:.2f}s")
        
        return {"summary": summary}
    
    finally:
        # Clean up temporary file
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass