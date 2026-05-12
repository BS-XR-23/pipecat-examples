from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
from pathlib import Path
from convert_to_txt import convert_to_txt
from loguru import logger
from typing import Tuple, List


# -------------------------
# TEXT CLEANING (NEW)
# -------------------------
def clean_text(text: str) -> str:
    """
    Removes noisy lines before embedding.
    Prevents brochure/website tone in RAG.
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if len(line) < 3:
            continue

        if "copyright" in line.lower():
            continue
        if "all rights reserved" in line.lower():
            continue

        cleaned.append(line)

    return "\n".join(cleaned)



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

    logger.debug(
        f"Vector store created at {persist_path} with {len(chunks)} chunks."
    )

    return vectordb, persist_path



def load_vectorstore(db_path, embeddings):
    if not os.path.isdir(db_path):
        raise ValueError(f"Vector store not found at {db_path}")

    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )



def embedd_document(document_path, vector_db_path, embeddings):
    """
    Convert document to text, clean it, split into chunks,
    and embed into Chroma vector store.
    """

    # Extract raw text
    docs_text = convert_to_txt(document_path)

    # ✅ CLEAN TEXT BEFORE CHUNKING
    docs_text = clean_text(docs_text)

    # -------------------------
    # BETTER CHUNKING (same structure, improved quality)
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=40,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! "
        ]
    )

    text_chunks = splitter.split_text(docs_text)

    # -------------------------
    # ADD METADATA TO CHUNKS (NEW)
    # -------------------------
    chunks = []
    for i, chunk in enumerate(text_chunks):
        chunks.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": str(document_path),
                    "chunk_id": i,
                    "type": "bank_knowledge"
                }
            )
        )

    vectordb, persist_path = create_vector_store(
        vector_db_path,
        embeddings,
        chunks
    )

    logger.debug(
        f"Document embedded successfully at {persist_path}"
    )

    return vectordb, persist_path



def get_rag_context(query: str, vectorstore, k: int = 3) -> Tuple[List[str], List[float]]:
    """
    Retrieve top-k relevant chunks with similarity scores.
    SAFE: handles None vectorstore + unexpected formats.
    """

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

                content = getattr(doc, "page_content", None)
                if content is None:
                    content = str(doc)

                content = content.strip()

                if len(content) < 5:
                    continue

                docs.append(content)
                scores.append(float(score))
            else:
                logger.warning(f"[RAG WARN] Unexpected result format: {item}")

        return docs, scores

    except Exception as e:
        logger.warning(f"[RAG ERROR] {e}")
        return [], []