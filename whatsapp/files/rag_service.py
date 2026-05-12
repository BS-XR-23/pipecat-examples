from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
from pathlib import Path
from convert_to_txt import convert_to_txt
from loguru import logger


def create_vector_store(vector_db_path, embeddings, chunks):
    """Create and persist a Chroma vector store."""
    persist_path = vector_db_path
    os.makedirs(persist_path, exist_ok=True)

    vectordb = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings,
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
    """Convert document to text, split into chunks, and embed into Chroma vector store."""
    docs_text   = convert_to_txt(document_path)
    splitter    = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    text_chunks = splitter.split_text(docs_text)
    chunks      = [Document(page_content=chunk) for chunk in text_chunks]

    vectordb, persist_path = create_vector_store(vector_db_path, embeddings, chunks)
    logger.debug(f"Document embedded successfully at {persist_path}.")


def get_rag_context(question: str, vectordb, k: int = 3):
    """Perform similarity search and return context for RAG."""
    docs_found = vectordb.similarity_search(question, k=k)
    if not docs_found:
        return [], []

    context_parts = []
    metadata_list = []

    for d in docs_found:
        if isinstance(d, Document):
            context_parts.append(d.page_content)
            metadata_list.append(d.metadata if hasattr(d, "metadata") else {})
        else:
            context_parts.append(str(d))
            metadata_list.append({})

    return context_parts, metadata_list
