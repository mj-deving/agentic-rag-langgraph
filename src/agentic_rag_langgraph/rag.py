"""The agent: a vector store, one search tool, and a two-node LangGraph state graph.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

`agent` is the chat model with the search tool bound; `tools` is LangGraph's prebuilt ToolNode.
The tool returns the chunks twice: as JSON content the model reads, and as a structured artifact
the grader reads. The citation gate takes the retrieved ids from those artifacts in the same run.
"""

import json
import sys
import uuid
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from . import config
from .instructions import INSTRUCTIONS

SEARCH_TOOL_NAME = "search-regulation"
SEARCH_TOOL_DESCRIPTION = (
    "Durchsucht wörtliche Auszüge der KI-Verordnung (EU) 2024/1689 (Artikel 4, 5, 6, 26, 50, 99, 113). "
    "Liefert Textstellen mit chunkId. Für Fragen mit mehreren Teilen einmal pro Teilfrage aufrufen."
)


class SearchInput(BaseModel):
    query: str = Field(min_length=3, description="Suchanfrage auf Deutsch, möglichst nah am Gesetzeswortlaut")


def embeddings() -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)


def chat_model() -> BaseChatModel:
    from langchain_ollama import ChatOllama

    # reasoning=False sends Ollama `think: false`, the same switch the Mastra repo sets.
    return ChatOllama(
        model=config.CHAT_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0, reasoning=False
    )


def ingest(emb: Embeddings, chunks_path=config.ROOT / "corpus" / "chunks.json", out=config.INDEX_PATH) -> int:
    """Embeds every chunk and writes the store to disk. Chunk id = document id, so reruns replace."""
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    store = InMemoryVectorStore(emb)
    docs = [
        Document(page_content=c["text"], id=c["id"], metadata={"chunkId": c["id"], "article": c["article"]})
        for c in chunks
    ]
    store.add_documents(docs, ids=[c["id"] for c in chunks])
    dims = {len(v["vector"]) for v in store.store.values()}
    if dims != {config.EMBEDDING_DIMENSION}:
        raise RuntimeError(f"expected {config.EMBEDDING_DIMENSION} dimensions, got {sorted(dims)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(out))
    return len(docs)


def load_store(emb: Embeddings, path=config.INDEX_PATH) -> InMemoryVectorStore:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run ingest` first")
    return InMemoryVectorStore.load(str(path), emb)


def search(store: InMemoryVectorStore, query: str, k: int) -> list[dict]:
    hits = store.similarity_search_with_score(query, k=k)
    return [
        {
            "chunkId": doc.metadata["chunkId"],
            "article": doc.metadata["article"],
            "score": round(float(score), 4),
            "text": doc.page_content,
        }
        for doc, score in hits
    ]


def make_search_tool(store: InMemoryVectorStore, k: int):
    @tool(SEARCH_TOOL_NAME, description=SEARCH_TOOL_DESCRIPTION, args_schema=SearchInput,
          response_format="content_and_artifact")
    def search_regulation(query: str) -> tuple[str, list[dict]]:
        chunks = search(store, query, k)
        return json.dumps({"chunks": chunks}, ensure_ascii=False), chunks

    return search_regulation


def build_graph(model: BaseChatModel, store: InMemoryVectorStore, *, prompt: str, k: int, checkpointer=None):
    search_tool = make_search_tool(store, k)
    llm = model.bind_tools([search_tool])

    def agent(state: MessagesState):
        return {"messages": [llm.invoke([SystemMessage(prompt), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([search_tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


@dataclass
class RunResult:
    answer: str
    retrieved_ids: set[str]
    queries: list[str] = field(default_factory=list)
    steps: int = 0


def run(graph, question: str) -> RunResult:
    """One question on a fresh thread, so retrieval from other runs can never satisfy the gate."""
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": config.RECURSION_LIMIT}
    messages = graph.invoke({"messages": [HumanMessage(question)]}, cfg)["messages"]
    retrieved, queries = set(), []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == SEARCH_TOOL_NAME and m.status == "success":
            retrieved.update(c["chunkId"] for c in m.artifact or [])
        if isinstance(m, AIMessage):
            queries += [tc["args"].get("query", "") for tc in m.tool_calls]
    final = messages[-1]
    answer = final.text if isinstance(final, AIMessage) else ""
    return RunResult(answer, retrieved, queries, sum(isinstance(m, AIMessage) for m in messages))


def default_graph(prompt_version: str = config.PROMPT_VERSION):
    return build_graph(
        chat_model(), load_store(embeddings()), prompt=INSTRUCTIONS[prompt_version], k=config.TOP_K,
        checkpointer=InMemorySaver(),
    )


def ingest_main() -> None:
    print(f"indexed {ingest(embeddings())} chunks into {config.INDEX_PATH}")


def ask_main() -> None:
    if len(sys.argv) < 2:
        sys.exit('usage: uv run ask "<Frage>"')
    result = run(default_graph(), " ".join(sys.argv[1:]))
    print(result.answer)
    print(f"\nretrieved: {', '.join(sorted(result.retrieved_ids))}")
