"""The real LangGraph graph, offline: a scripted chat model and hash embeddings, no Ollama."""

import json

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag_langgraph.config import ROOT
from agentic_rag_langgraph.grading import REFUSAL_MARKER, citation_validity
from agentic_rag_langgraph.rag import SEARCH_TOOL_NAME, build_graph, ingest, load_store, run, search

CHUNKS = json.loads((ROOT / "corpus/chunks.json").read_text(encoding="utf-8"))
EMB = DeterministicFakeEmbedding(size=1024)


class ScriptedModel(GenericFakeChatModel):
    """Replays fixed AI messages; tool binding is a no-op because the script already holds the calls."""

    def bind_tools(self, tools, **kwargs):
        return self


def store() -> InMemoryVectorStore:
    s = InMemoryVectorStore(EMB)
    s.add_documents(
        [Document(c["text"], id=c["id"], metadata={"chunkId": c["id"], "article": c["article"]}) for c in CHUNKS],
        ids=[c["id"] for c in CHUNKS],
    )
    return s


def graph_for(*replies: AIMessage, k: int = 5):
    model = ScriptedModel(messages=iter(replies))
    return build_graph(model, store(), prompt="test", k=k, checkpointer=InMemorySaver())


def tool_call(query: str, call_id: str = "c1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": SEARCH_TOOL_NAME, "args": {"query": query}, "id": call_id}])


def test_retrieved_ids_come_from_the_tool_artifact_of_this_run():
    s = store()
    expected = {h["chunkId"] for h in search(s, "Geldbuße", 5)}
    cited = sorted(expected)[0]
    g = graph_for(tool_call("Geldbuße"), AIMessage(f"Antwort [chunk:{cited}]"))
    result = run(g, "Welche Geldbuße droht?")
    assert result.retrieved_ids == expected
    assert result.queries == ["Geldbuße"]
    assert result.answer == f"Antwort [chunk:{cited}]"
    assert citation_validity(result.answer, result.retrieved_ids) == 1


def test_citation_of_an_unretrieved_chunk_fails_the_gate():
    g = graph_for(tool_call("Geldbuße"), AIMessage("Antwort [chunk:does-not-exist]"))
    result = run(g, "Welche Geldbuße droht?")
    assert citation_validity(result.answer, result.retrieved_ids) == 0


def test_answer_without_a_tool_call_has_no_retrieval():
    g = graph_for(AIMessage("Antwort [chunk:art50-p1]"))
    result = run(g, "Frage")
    assert result.retrieved_ids == set()
    assert citation_validity(result.answer, result.retrieved_ids) == 0


def test_runs_do_not_share_retrieval_through_the_checkpointer():
    first = tool_call("Geldbuße")
    g = graph_for(first, AIMessage("A"), AIMessage(f"{REFUSAL_MARKER} nichts"))
    assert run(g, "eins").retrieved_ids
    assert run(g, "zwei").retrieved_ids == set()


def test_two_tool_calls_union_their_chunks_and_k_is_respected():
    g = graph_for(tool_call("Geldbuße", "c1"), tool_call("Deepfake", "c2"), AIMessage("A"), k=3)
    result = run(g, "zwei Teile")
    assert 3 < len(result.retrieved_ids) <= 6
    assert result.queries == ["Geldbuße", "Deepfake"]


def test_tool_message_carries_json_for_the_model_and_chunks_for_the_grader():
    g = graph_for(tool_call("Geldbuße"), AIMessage("A"))
    cfg = {"configurable": {"thread_id": "t"}}
    msgs = g.invoke({"messages": [("user", "q")]}, cfg)["messages"]
    tm = next(m for m in msgs if isinstance(m, ToolMessage))
    payload = json.loads(tm.content)
    assert [c["chunkId"] for c in payload["chunks"]] == [c["chunkId"] for c in tm.artifact]
    assert set(payload["chunks"][0]) == {"chunkId", "article", "score", "text"}


def test_too_short_query_is_returned_to_the_model_as_an_error_not_a_crash():
    g = graph_for(tool_call("ab"), AIMessage(f"{REFUSAL_MARKER} x"))
    result = run(g, "q")
    assert result.retrieved_ids == set()
    assert result.answer.startswith(REFUSAL_MARKER)


def test_ingest_round_trip(tmp_path):
    out = tmp_path / "index.json"
    assert ingest(EMB, out=out) == len(CHUNKS)
    loaded = load_store(EMB, path=out)
    assert len(loaded.store) == len(CHUNKS)
