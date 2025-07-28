from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def get_text_chunks(text: str, chunk_size: int =1000, chunk_overlap: int=200) -> list[str]:
    """
    Splits a given text  into smaller overlapping chunks for processing.

    :param text: The full input text that will be split into chunks.
    :param chunk_size: Maximum number of characters per chunk. Defaults to 1000.
    :param chunk_overlap: Number of characters that overlap between chunks. Defaults to 200.
    :return: A list of text chunks suitable for embedding or vector storage
    """
    splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)


def get_vectorstore(text_chunks: list[str]):
    """
    Converts a list of text chunks into a FAISS vector store using OpenAI embeddings.
    Args:
        text_chunks: The text chunks to embed and store.

    Returns:
        FAISS: A FAISS vector store containing embedded representations of the text chunks.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore