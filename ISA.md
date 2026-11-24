---
task: "LangGraph-Port des Agentic-RAG-Vertrags aus agentic-rag-mastra, gleicher Korpus, gleiche Regeln"
slug: 20261109_agentic-rag-langgraph
project: agentic-rag-langgraph
effort: advanced
effort_source: inferred
phase: complete
progress: 15/15
mode: interactive
started: 2026-11-09T16:00:00+01:00
updated: 2026-11-09T16:00:00+01:00
absence_review: skipped — Anti-Kriterien aus den Projektgrenzen abgeleitet
---

> **Über diese Datei.** Das ist die ISA (Ideal State Artifact), gegen die dieses Repo gebaut wurde:
> was „fertig“ hieß, mit welcher Probe jedes Kriterium widerlegt würde und was bewusst als Grenze
> stehen blieb. Es ist der Arbeitsstand aus dem Bau, nur um private Bezüge bereinigt; Hinweise wie
> „kein Remote“ oder „ISA.md gitignored“ beschreiben die Zeit vor der Veröffentlichung. Commit-Hashes
> stammen aus der lokalen Vorgeschichte, öffentlich wurde das Repo als neuer Verlauf exportiert. Die
> Leak-Probes lesen ihre Begriffe aus einer lokalen, nicht versionierten Datei `.private-terms`.

## Ideal State

### L3 · One sentence

Ein LangGraph-Repo erfüllt denselben zitierenden, verweigernden RAG-Vertrag wie das Mastra-Repo auf denselben Daten und Regeln, sodass beide Frameworks mit gleichen Zahlen verglichen werden können.

### L2 · Summary

Wer zwischen Mastra und LangGraph wählt, legt beide Repos nebeneinander und sieht dieselbe Aufgabe, dieselben Gates und ehrliche Zahlen.

- Korpus und Golden Sets byte-gleich kopiert, Hash-Test.
- Grading 1:1 aus TypeScript portiert, inklusive Mastra-Verdict-Arithmetik.
- LangGraph-Primitive: StateGraph, ToolNode, tools_condition, InMemorySaver, InMemoryVectorStore.
- README-Tabelle Mastra vs. LangGraph aus beiden `results/`.

### L1 · Full articulation

Der Vergleich ist nur fair, wenn sich ausschließlich das Framework unterscheidet: gleiche Chunks, gleiche Fälle, gleicher Prompt v2 wörtlich, gleiches Modell (qwen3:8b, think off, temperature 0), gleiche Embeddings (bge-m3), top-k 5, gleiche Bewertungsregeln. Die Eval schreibt dieselben Metriken und endet ≠ 0, solange Gate oder Schwellen nicht halten. Das README trennt, was LangGraph geschenkt hat, von dem, was von Hand dazukam.

## Constraints

- Python mit uv.
- In `../agentic-rag-mastra` nur lesen.
- Ollama-Lock aus den Projektnotizen: ein Eval-/Ingest-Lauf zur Zeit.
- Lokal, privat, kein Remote. autoreview `--engine claude` vor jedem Commit.

## Criteria

- [x] ISC-1: Korpus (`corpus/chunks.json`, `corpus/sources/*`) und Golden Sets (`golden/*.jsonl`) sind byte-gleich mit der Quelle, Hashes und Quell-Commit in `corpus/PROVENANCE.json`.
- [x] ISC-2: Jeder Chunk ist wörtlicher Teilstring seines Snapshots, jeder Snapshot-Hash stimmt mit seiner Provenienz.
- [x] ISC-3: Grading-Port deckt jeden Fall aus `test/grading.test.ts` ab, inklusive „zitiert, aber nicht abgerufen → 0“.
- [x] ISC-4: Verdict-Arithmetik entspricht Mastras `runEvals` (nicht bewertbar fällt raus, Gate bei Mittel 1, failed/scored/passed).
- [x] ISC-5: Prompt v1 und v2 sind hash-gleich mit dem gerenderten TypeScript-Original.
- [x] ISC-6: Der Graph sammelt abgerufene IDs nur aus Tool-Artefakten des eigenen Laufs; offline mit Fake-Modell geprüft.
- [x] ISC-7: `uv run pytest` grün ohne Ollama.
- [x] ISC-8: `uv run eval` schreibt `results/*.json|md` mit denselben Metriken und endet ≠ 0 außer bei `passed`.
- [x] ISC-9: Eval-Läufe golden v1 und heldout mit Prompt v2 sind committet, Fehlschläge inklusive.
- [x] ISC-10: README mit Quickstart, Architecture, Results und „Mastra vs. LangGraph“-Tabelle (Zahlen beider Repos, Glue-Code-Zeilen, geschenkt vs. handgebaut).
- [x] ISC-11: Anti: keine README-Zahl, die nicht in einer Ergebnisdatei steht.
- [x] ISC-12: Anti: keine privaten Bezüge, Secrets oder absoluten Nutzerpfade im Repo.
- [x] ISC-13: Anti: keine Datei in `../agentic-rag-mastra` verändert.
- [x] ISC-14: Jeder Commit hat einen autoreview-Lauf mit `--engine claude` ohne offenen P1.
- [x] ISC-15: `scripts/compare.py` erzeugt `results/comparison.json` reproduzierbar aus beiden Repos: Codezeilen je Rolle und Lauf-für-Lauf-Metriken, Mastra-Seite per `git show` an einem gepinnten Commit. Probe: `uv run python scripts/compare.py ../agentic-rag-mastra d19dfcd` reproduziert die committete Datei (Mastra 485 / LangGraph 372 Zeilen, 39/40 gleiche Entscheidungen); autoreview vor 9822458 hat Script und Zahlen nachgerechnet, scoped-clean.

## Test Strategy

| isc | type | check | threshold | tool | anchors_to |
|-----|------|-------|-----------|------|------------|
| ISC-1 | bash | SHA-256 aller Kopien gegen PROVENANCE.json | 18/18 | uv run pytest tests/test_corpus.py | |
| ISC-2 | bash | Chunk ⊂ Snapshot, Snapshot-Hash | 100 % | uv run pytest tests/test_corpus.py | |
| ISC-3 | bash | Grading-Grenzfälle | alle grün | uv run pytest tests/test_grading.py | test/grading.test.ts |
| ISC-4 | bash | Verdict-Fälle | alle grün | uv run pytest tests/test_grading.py | Mastra runEvals |
| ISC-5 | bash | Prompt-Hashes | 2/2 | uv run pytest tests/test_instructions.py | instructions.ts |
| ISC-6 | bash | Graph offline mit Fake-Modell | alle grün | uv run pytest tests/test_graph.py | |
| ISC-7 | bash | Suite ohne Ollama | exit 0 | `OLLAMA_BASE_URL=http://127.0.0.1:9 uv run pytest` | |
| ISC-8 | bash | Exit-Code der Eval | ≠ 0 bei failed/scored | `uv run eval; echo $?` | |
| ISC-9 | bash | Ergebnisse committet | 4 Dateien | `git ls-files results/` | |
| ISC-10 | bash | README-Abschnitte | 4 Überschriften | `rg "^## (Quickstart\|Architecture\|Results\|Mastra vs)" README.md` | |
| ISC-11 | manual | README-Zahlen gegen results | jede gefunden | Abgleich per Skript, README ist Prosa | |
| ISC-12 | bash | Leak-Scan | 0 Treffer | `rg -i "/Users/\|\.claude\|api[_-]?key"` über git ls-files | |
| ISC-13 | bash | Mastra-Repo unverändert durch mich | kein Diff von mir | `git -C ../agentic-rag-mastra status` vor/nach | |
| ISC-14 | manual | autoreview-Ausgabe | 0 P1 | autoreview `--engine claude` | |

## Decisions

- 2026-11-09: InMemoryVectorStore (langchain-core) statt LibSQL: nativ, 66 Vektoren, als JSON persistiert; braucht numpy.
- 2026-11-09: Graph explizit als StateGraph mit ToolNode statt `create_agent`, damit jede Kante sichtbar ist.
- 2026-11-09: Rekursionslimit 41 = 20 Modellschritte, gespiegelt am `stepCountIs(20)` im Mastra-Bundle (gelesen, nicht als Default verifiziert).

## Evidence
- 2026-11-09 ISC-1..7: `OLLAMA_BASE_URL=http://127.0.0.1:9 uv run pytest -q` → 79 passed (Hash-Kopien 18/18, Grading-Grenzfälle inkl. `normalize`-Erwartungen aus dem TS-Original per bun erzeugt, Verdict-Fälle, Prompt-Hashes v1/v2, Graph offline mit Fake-Modell).
- 2026-11-09 Glue-Code: `uv run python scripts/count_loc.py ../agentic-rag-mastra d19dfcd` → Mastra 485, LangGraph 372 Codezeilen (`results/loc.json`).
- 2026-11-09 ISC-8: Eval-Job unter Ollama-Lock (16:02–16:08): golden v1 → `verdict: failed`, exit 1; heldout → `verdict: scored`, exit 1. Ergebnisdateien `results/2026-11-09-qwen3-8b-v2-{v1,heldout}.{json,md}`.
- 2026-11-09 ISC-9: Commit 9822458 enthält beide Läufe (`git ls-files results/` → 5 Dateien inkl. comparison.json).
- 2026-11-09 ISC-10/11: README mit 4 Pflichtabschnitten; autoreview (claude, P1) hat jede README-Zahl gegen results/*.json nachgerechnet, scoped-clean. „12 lines“ als einzige unbelegte Zahl entfernt.
- 2026-11-09 ISC-12: `git ls-files -z | xargs -0 rg -il -f .private-terms` plus Secret-Muster → 0 Treffer.
- 2026-11-09 ISC-13: Im Mastra-Repo nur gelesen (git show, bun -e Imports). Dortige Änderungen stammen aus einem parallelen Arbeitsstrang (Commit 69f2782).
- 2026-11-09 ISC-14: autoreview `--engine claude --max-priority P1` vor 025024a und vor 9822458, beide `scoped-clean`.
- Befund: 8/40 golden und 2/8 heldout ohne Tool-Aufruf beantwortet; 3 davon falsche Verweigerungen. Tool-Zwang im ersten Schritt nicht gebaut (Held-out-Disziplin).
