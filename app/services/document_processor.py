import tempfile
from app.services.summarizer import summarize_docs
from app.services.vectorstore import store_in_vectorstore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

async def process_document(file, metadata):
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

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

    # Summarize
    summary = summarize_docs(docs)

    # Store embeddings
    store_in_vectorstore(chunks, metadata)

    return {"summary": summary}
