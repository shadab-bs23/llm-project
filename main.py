import os
from typing import List
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from app.services.document_processor import process_document
from app.services.vectorstore import vectorstore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
import asyncio


app = FastAPI()


@app.post("/process-document")
async def process_document_api(
    file: UploadFile,
    uploaded_by: str = Form(...),
):
    result = await process_document(file, {"uploaded_by": uploaded_by})
    return {"status": "success", "summary": result["summary"]}


# Semaphore to limit concurrent file processing (prevent CPU thrashing)
_concurrency_limit = asyncio.Semaphore(3)

@app.post("/process-documents")
async def process_documents_api(
    files: List[UploadFile] = File(...),
    uploaded_by: str = Form(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided. Use the 'files' field and select one or more.")

    async def process_with_limit(file):
        async with _concurrency_limit:
            try:
                # Add per-file timeout (60 seconds)
                result = await asyncio.wait_for(
                    process_document(file, {"uploaded_by": uploaded_by}),
                    timeout=60.0
                )
                return {"filename": file.filename, "summary": result["summary"], "status": "success"}
            except asyncio.TimeoutError:
                return {"filename": file.filename, "error": "Processing timeout (>60s)", "status": "error"}
            except Exception as e:
                return {"filename": file.filename, "error": str(e), "status": "error"}

    # Process files with bounded concurrency and error handling
    results = await asyncio.gather(*[process_with_limit(f) for f in files])
    
    return {"status": "success", "results": results}


@app.get("/query")
def query_knowledge_base(query: str):
    # Check if vectorstore has any data by searching with the actual query
    try:
        # Search for documents similar to the user's question
        # This uses semantic similarity to find relevant content
        relevant_docs = vectorstore.similarity_search(query, k=5)
        if not relevant_docs:
            return {"answer": "I don't know"}
            
    except Exception:
        # If there's an error accessing the vectorstore, it's likely empty
        return {"answer": "I don't know"}
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
        retriever=retriever
    )
    return {"answer": qa_chain.run(query)}
