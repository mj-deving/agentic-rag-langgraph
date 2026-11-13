"""Compares this repo with its Mastra twin: code lines and the eval runs of both.

    uv run python scripts/compare.py <path-to-agentic-rag-mastra> [<commit>]

The Mastra side is read with `git show <commit>:<file>`, so uncommitted work there is not counted.
Corpus building (fetch-corpus.ts, build-chunks.ts) is left out on the Mastra side because this
repo copies the corpus instead of rebuilding it. Code lines exclude blanks, comments and
docstrings. Mastra result files are read at the same commit. Writes results/comparison.json.
"""

import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each file is assigned to one role so the two repos can be compared row by row.
MASTRA = {
    "src/lib/grading.ts": "grading",
    "src/lib/golden.ts": "golden loader",
    "src/mastra/agents/instructions.ts": "prompt",
    "src/lib/config.ts": "config",
    "src/lib/models.ts": "agent + tool + store",
    "src/mastra/agents/regulation-agent.ts": "agent + tool + store",
    "src/mastra/tools/search-regulation.ts": "agent + tool + store",
    "src/mastra/lib/vector-store.ts": "agent + tool + store",
    "src/mastra/index.ts": "agent + tool + store",
    "src/mastra/workflows/ingest-corpus.ts": "ingest + ask",
    "scripts/ingest.ts": "ingest + ask",
    "scripts/ask.ts": "ingest + ask",
    "src/mastra/scorers/index.ts": "eval runner + scorers",
    "scripts/eval.ts": "eval runner + scorers",
}
LANGGRAPH = {
    "src/agentic_rag_langgraph/grading.py": "grading",
    "src/agentic_rag_langgraph/golden.py": "golden loader",
    "src/agentic_rag_langgraph/instructions.py": "prompt",
    "src/agentic_rag_langgraph/config.py": "config",
    "src/agentic_rag_langgraph/rag.py": "agent + tool + store",  # also holds ingest + ask (16 lines)
    "src/agentic_rag_langgraph/eval.py": "eval runner + scorers",
}


def ts_code_lines(src: str) -> int:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return sum(1 for line in src.splitlines() if line.strip() and not line.strip().startswith("//"))


def py_code_lines(src: str) -> int:
    # Lines that hold at least one token other than comments, docstrings and layout tokens.
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    lines: set[int] = set()
    prev = tokenize.NEWLINE
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        is_docstring = tok.type == tokenize.STRING and prev in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
        if tok.type not in skip and not is_docstring:
            lines.update(range(tok.start[0], tok.end[0] + 1))
        if tok.type not in (tokenize.COMMENT, tokenize.NL):
            prev = tok.type
    return len(lines)


def by_role(files: dict[str, str], read, count) -> dict:
    per_file = {f: count(read(f)) for f in files}
    roles: dict[str, int] = {}
    for f, role in files.items():
        roles[role] = roles.get(role, 0) + per_file[f]
    return {"files": per_file, "roles": roles, "total": sum(per_file.values())}


def main() -> None:
    mastra_root = Path(sys.argv[1]).resolve()
    commit = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    sha = subprocess.run(["git", "-C", mastra_root, "rev-parse", commit], capture_output=True, text=True, check=True).stdout.strip()

    def git_show(path: str) -> str:
        return subprocess.run(["git", "-C", mastra_root, "show", f"{sha}:{path}"], capture_output=True, text=True, check=True).stdout

    loc = {
        "rule": "code lines without blanks, comments or docstrings",
        "mastra": {"commit": sha, **by_role(MASTRA, git_show, ts_code_lines)},
        "langgraph": by_role(LANGGRAPH, lambda f: (ROOT / f).read_text(encoding="utf-8"), py_code_lines),
    }
    runs = {}
    for name in RUNS:
        mastra = json.loads(git_show(f"results/{name}.json"))
        langgraph = json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))
        runs[name] = {
            "mastra": summary(mastra["gateResults"], mastra["scores"], mastra),
            "langgraph": summary(langgraph["aggregate"]["gateResults"], langgraph["aggregate"]["scores"], langgraph)
            | {"answeredWithoutSearch": [i["id"] for i in langgraph["items"] if not i["queries"]]},
            "perCase": per_case(mastra["items"], langgraph["items"]),
        }
    out = ROOT / "results" / "comparison.json"
    out.write_text(json.dumps({"loc": loc, "runs": runs}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: loc[k]["roles"] | {"total": loc[k]["total"]} for k in ("mastra", "langgraph")}, indent=2))
    print(json.dumps(runs, indent=2, ensure_ascii=False))


# Same prompt (v2), model, embeddings and top-k in both repos; file names match by construction.
RUNS = ("2026-11-09-qwen3-8b-v2-v1", "2026-11-09-qwen3-8b-v2-heldout")


def summary(gates: list[dict], scores: dict, run: dict) -> dict:
    items = run["items"]
    ooc = [i for i in items if "out-of-corpus" in i["tags"]]
    return {
        "verdict": run["verdict"],
        "citationValidity": next(g["score"] for g in gates if g["id"] == "citation-validity"),
        "refusalCorrectness": scores["refusal-correctness"],
        "outOfCorpusRefused": f"{sum(i['scores']['refusal-correctness'] == 1 for i in ooc)}/{len(ooc)}",
        "citationRecall": scores.get("citation-recall"),
        "factCoverage": scores.get("fact-coverage"),
        "retrievalRecallAtK": run["retrieval"]["recallAtK"],
    }


def per_case(mastra: list[dict], langgraph: list[dict]) -> dict:
    """Where the two agents decided differently on the same question."""
    from agentic_rag_langgraph.grading import is_refusal

    m = {i["id"]: i for i in mastra}
    same_decision = [i["id"] for i in langgraph if is_refusal(i["answer"]) == is_refusal(m[i["id"]]["answer"])]
    gate_fail = lambda items: sorted(i["id"] for i in items if i["scores"]["citation-validity"] == 0)  # noqa: E731
    return {
        "cases": len(langgraph),
        "sameAnswerOrRefuseDecision": len(same_decision),
        "differentDecision": sorted(set(m) - set(same_decision)),
        "gateFailures": {"mastra": gate_fail(mastra), "langgraph": gate_fail(langgraph)},
    }


if __name__ == "__main__":
    main()
