from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
import os

def summarize_docs(docs):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)
