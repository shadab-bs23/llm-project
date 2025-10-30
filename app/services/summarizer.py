from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

# Summarize the documents
# @param docs: The documents to summarize
# @return: The summary of the documents
def summarize_docs(docs):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    # map_reduce is a chain type that summarizes the documents using the map and reduce functions
    # map function is used to summarize the documents
    # reduce function is used to combine the summaries of the documents
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    # run the chain to get the summary of the documents
    return chain.run(docs)
