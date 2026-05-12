from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
import os


def embedd_document(document_path: str, vectorstore_path: str, embeddings) -> None:
    """Load a text document, split it, and embed into Chroma."""
    logger.info(f"[RAG] Embedding document: {document_path}")

    loader = TextLoader(document_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vectorstore_path
    )

    logger.info(f"[RAG] Embedded {len(chunks)} chunks into Chroma at {vectorstore_path}")


def load_vectorstore(vectorstore_path: str, embeddings) -> Chroma:
    """Load an existing Chroma vectorstore from disk."""
    logger.info(f"[RAG] Loading vectorstore from: {vectorstore_path}")
    return Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embeddings
    )


def get_rag_context(query: str, vectorstore: Chroma, k: int = 4):
    """Retrieve top-k relevant chunks for a query with similarity scores."""
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        docs   = [doc.page_content for doc, _ in results]
        scores = [score           for _,   score in results]
        return docs, scores
    except Exception as e:
        logger.warning(f"[RAG ERROR] {e}")
        return [], []