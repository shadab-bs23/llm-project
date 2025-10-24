import os
from fastapi import FastAPI, UploadFile, Form
from app.services.document_processor import process_document
from app.services.vectorstore import vectorstore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA


app = FastAPI()


@app.post("/process-document")
async def process_document_api(
    file: UploadFile,
    uploaded_by: str = Form(...),
    project: str = Form(...)
):
    result = await process_document(file, {"uploaded_by": uploaded_by, "project": project})
    return {"status": "success", "summary": result["summary"]}


@app.get("/query")
def query_knowledge_base(query: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        retriever=retriever
    )
    return {"answer": qa_chain.run(query)}
