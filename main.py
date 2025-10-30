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


@app.post("/process-documents")
async def process_documents_api(
    files: List[UploadFile] = File(...),
    uploaded_by: str = Form(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided. Use the 'files' field and select one or more.")

    tasks = [
        process_document(f, {"uploaded_by": uploaded_by})
        for f in files
    ]
    results = await asyncio.gather(*tasks)
    response_items = [
        {"filename": f.filename, "summary": r["summary"]}
        for f, r in zip(files, results)
    ]
    return {"status": "success", "results": response_items}


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
