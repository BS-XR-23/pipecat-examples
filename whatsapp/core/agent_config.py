import os
from utils.vector_store_helpers import load_vectorstore, load_system_prompt
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_BASE_PATH = os.getenv("VECTORSTORE_BASE_PATH")
class AgentConfig:
    def __init__(self, config: dict):
        self.name = config["name"]
        self.tools = config.get("tools")
        self.graph_builder = config.get("graph")
        self.memory_template = config.get("memory")

        self.vector_db_id = config.get("vector_db")

        self.document_path = config.get("document")
        self.system_prompt_path = config.get("prompt")

        # runtime overrides
        self.system_prompt_raw = config.get("system_prompt_raw")
        self.hash_address = config.get("hash_address")
        self.llm_model = config.get("llm_model")
        self.embedding_model = config.get("embedding_model")

        # caches
        self._vectorstore = None
        self._system_prompt = None
        self._compiled_graph = None

    # ─────────────────────────────────────────────
    # HOT RELOAD
    # ─────────────────────────────────────────────
    def reload_paths(
        self,
        vector_db_id: str = None,
        document_path: str = None,
        system_prompt_raw: str = None,
        hash_address: str = None,
        llm_model: str = None,
        embedding_model: str = None,
    ):
        if vector_db_id:
            self.vector_db_id = vector_db_id

        if document_path:
            self.document_path = document_path

        if system_prompt_raw:
            self.system_prompt_raw = system_prompt_raw

        if hash_address:
            self.hash_address = hash_address

        if llm_model:
            self.llm_model = llm_model

        if embedding_model:
            self.embedding_model = embedding_model

        # clear caches
        self._vectorstore = None
        self._system_prompt = None

    # ─────────────────────────────────────────────
    # GRAPH
    # ─────────────────────────────────────────────
    def get_graph(self):
        if self.graph_builder is None:
            raise ValueError(f"[{self.name}] graph_builder is not configured")
        if self._compiled_graph is None:
            self._compiled_graph = self.graph_builder()
        return self._compiled_graph

    # ─────────────────────────────────────────────
    # SYSTEM PROMPT
    # ─────────────────────────────────────────────
    def get_system_prompt(self):
        if self._system_prompt is None:
            if self.system_prompt_raw:
                self._system_prompt = self.system_prompt_raw
            elif self.system_prompt_path:
                self._system_prompt = load_system_prompt(self.system_prompt_path)
            else:
                raise ValueError(f"[{self.name}] No system prompt found")
        return self._system_prompt

    # ─────────────────────────────────────────────
    # 🔥 VECTOR STORE (FINAL CLEAN VERSION)
    # ─────────────────────────────────────────────
    def get_vectorstore(self):
        """
        Resolves:
        chatbot_2 → /vectorstore/chroma/chatbot_2
        """

        if not self.vector_db_id:
            raise ValueError(f"[{self.name}] vector_db_id is missing")

        # clean safety
        vector_db_id = self.vector_db_id.replace("\\", "/").strip()

        full_path = os.path.join(VECTORSTORE_BASE_PATH, vector_db_id)

        if not os.path.exists(full_path):
            raise ValueError(
                f"[{self.name}] Vector DB not found at: {full_path}"
            )

        if self._vectorstore is None:
            embeddings = OllamaEmbeddings(
                model=self.embedding_model or "embeddinggemma"
            )

            self._vectorstore = load_vectorstore(
                full_path,
                embeddings
            )

        return self._vectorstore