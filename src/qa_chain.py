from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.vectorstores.base import VectorStore



def get_conversational_chain(vectorstore: VectorStore) -> RetrievalQA:
    """
    Creates a retrieval chain from OpenAI's retrieval API.
    Args:
        vectorstore: The OpenAI retrieval API.


    Returns:
        RetrievalQA: A retrieval chain from OpenAI's retrieval API.
    """
    llm = OpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever= vectorstore.as_retriever(),
        return_source_documents = True
    )
    return chain