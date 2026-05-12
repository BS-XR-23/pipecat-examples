import os
import hashlib
from rag_service import OllamaEmbeddings, embedd_document, load_vectorstore


def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def should_embed(document_path: str, hash_file: str) -> bool:
    current_hash = get_file_hash(document_path)

    if not os.path.exists(hash_file):
        return True

    try:
        with open(hash_file, "r") as f:
            saved_hash = f.read().strip()
    except Exception:
        return True

    return current_hash != saved_hash


def save_hash(document_path: str, hash_file: str):
    os.makedirs(os.path.dirname(hash_file), exist_ok=True)
    current_hash = get_file_hash(document_path)
    with open(hash_file, "w") as f:
        f.write(current_hash)


# -------------------- VECTORSTORE BUILDER --------------------
def build_vectorstore(vector_db_path: str, document_path: str):
    embeddings = OllamaEmbeddings(model="embeddinggemma")

    hash_file = os.path.join(vector_db_path, "hash.txt")

    if os.path.exists(document_path):
        if should_embed(document_path, hash_file):
            embedd_document(document_path, vector_db_path, embeddings)
            save_hash(document_path, hash_file)

        return load_vectorstore(vector_db_path, embeddings)

    return None


# -------------------- PROMPT LOADER --------------------
def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()