from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama     # NEW

def make_qa_chain(vectorstore):
    llm = Ollama(
        base_url = "http://localhost:11434",
        model    = "gemma3:1b",
        temperature = 0.3,
    )
    return RetrievalQA.from_chain_type(
        llm        = llm,
        chain_type = "stuff",
        retriever  = vectorstore.as_retriever(search_kwargs={"k": 5}),
    )

