from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
from pathlib import Path
from convert_to_txt import convert_to_txt
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
from typing import Tuple, List
from pathlib import Path
from convert_to_txt import convert_to_txt
from loguru import logger


def create_vector_store(vector_db_path, embeddings, chunks):
    """
    Create a vector store Chroma.
    Returns: vectordb, persist_path
    """
    persist_path = vector_db_path
    os.makedirs(persist_path, exist_ok=True)

    vectordb = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings
    )
    vectordb.add_documents(chunks)
    vectordb.persist()
    logger.debug(f"Vector store created at {persist_path} with {len(chunks)} chunks.")
    return vectordb, persist_path

def load_vectorstore(db_path, embeddings):
    if not os.path.isdir(db_path):
        raise ValueError(f"Vector store not found at {db_path}")
    return Chroma(persist_directory=db_path, embedding_function=embeddings)

def embedd_document(document_path, vector_db_path, embeddings):
    """
    Convert document to text, split into chunks, and embed into Chroma vector store.
    """

    # Extract raw text from document
    docs_text = convert_to_txt(document_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    text_chunks = splitter.split_text(docs_text)

    # Wrap chunks in Document objects for proper storage in vector store
    chunks = [Document(page_content=chunk) for chunk in text_chunks]

    vectordb, persist_path = create_vector_store(vector_db_path, embeddings, chunks)
    logger.debug(f"Document embedded and vector store created successfully at {vectordb}.")

from langchain_core.documents import Document


def get_rag_context(query: str, vectorstore, k: int = 3) -> Tuple[List[str], List[float]]:
    """
    Retrieve top-k relevant chunks with similarity scores.
    SAFE: handles None vectorstore + unexpected formats.
    """

    # 🔴 Safety check (prevents your crash)
    if vectorstore is None:
        logger.warning("[RAG ERROR] Vectorstore is None")
        return [], []

    try:
        results = vectorstore.similarity_search_with_score(query, k=k)

        if not results:
            return [], []

        docs = []
        scores = []

        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item

                # handle LangChain Document safely
                content = getattr(doc, "page_content", None)
                if content is None:
                    content = str(doc)

                docs.append(content)
                scores.append(float(score))
            else:
                logger.warning(f"[RAG WARN] Unexpected result format: {item}")

        return docs, scores

    except Exception as e:
        logger.warning(f"[RAG ERROR] {e}")
        return [], []