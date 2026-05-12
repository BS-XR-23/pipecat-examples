from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from typing import Tuple, List
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