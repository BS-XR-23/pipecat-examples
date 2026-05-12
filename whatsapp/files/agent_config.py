from utils.vector_store_helpers import build_vectorstore, load_system_prompt


class AgentConfig:
    def __init__(self, config: dict):
        self.name = config["name"]
        self.tools = config.get("tools")
        self.graph_builder = config.get("graph")
        self.memory_template = config.get("memory")
        self.vector_db_path = config.get("vector_db")
        self.document_path = config.get("document")
        self.system_prompt_path = config.get("prompt")

        # 🔥 CACHED OBJECTS
        self._compiled_graph = None
        self._vectorstore = None
        self._system_prompt = None

    # -------------------- GRAPH --------------------
    def get_graph(self):
        if self.graph_builder is None:
            raise ValueError(f"[{self.name}] graph_builder is not configured")
        if self._compiled_graph is None:
            self._compiled_graph = self.graph_builder()
        return self._compiled_graph

    # -------------------- VECTORSTORE --------------------
    def get_vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = build_vectorstore(
                self.vector_db_path,
                self.document_path
            )
        return self._vectorstore

    # -------------------- SYSTEM PROMPT --------------------
    def get_system_prompt(self):
        if self._system_prompt is None:
            self._system_prompt = load_system_prompt(
                self.system_prompt_path
            )
        return self._system_prompt