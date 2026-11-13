# Eval run 2026-11-09T15:07:59.253Z

Framework LangGraph, model `qwen3:8b`, prompt `v2`, golden `golden/heldout.jsonl`, embeddings `bge-m3`, top-k 5, 8 cases. Verdict: **scored**.

## Aggregate

| metric | score | rule |
|---|---|---|
| retrieval recall@k (no LLM) | 100.0 % | tracked |
| citation-validity | 100.0 % | gate: must be 100 % |
| refusal-correctness | 62.5 % | ≥ 85.0 % |
| citation-recall | 100.0 % | ≥ 80.0 % |
| fact-coverage | 100.0 % | ≥ 70.0 % |

Out-of-corpus refused: 2 / 4.

## By tag

| tag | n | validity | refusal | recall | facts |
|---|---|---|---|---|---|
| numerical | 2 | 100.0 % | 50.0 % | 100.0 % | 100.0 % |
| out-of-corpus | 4 | 100.0 % | 50.0 % | – | – |
| single | 2 | 100.0 % | 100.0 % | 100.0 % | 100.0 % |

## Cases below 1.0 (3)

### h-a99-midcap (answered)

{"citation-validity": 1, "refusal-correctness": 0, "citation-recall": null, "fact-coverage": null}

> [KEINE_GRUNDLAGE] Die abgerufenen Auszüge enthalten dazu keine Grundlage.

### h-ooc-conformity (refused)

{"citation-validity": 1, "refusal-correctness": 0, "citation-recall": null, "fact-coverage": null}

> Das Konformitätsbewertungsverfahren für Hochrisiko-KI-Systeme ist in Artikel 6 der KI-Verordnung geregelt. Hochrisiko-KI-Systeme sind Systeme, die als Sicherheitsbauteil eines Produkts unter harmonisierendem Recht der Union verwendet werden oder selbst solch ein Produkt sind, und das Produkt muss einer Konformitätsbewertung durch Dritte im Hinblick auf das Inverkehrbringen oder die Inbetriebnahme unterzogen werden [chunk:art6-p1]. Zusätzlich arbeiten die Betreiber von Hochrisiko-KI-Systemen mit den zuständigen Behörden zusammen, um Maßnahmen zur Umsetzung der Verordnung zu ergreifen [chunk:art

### h-ooc-market-surveillance (refused)

{"citation-validity": 1, "refusal-correctness": 0, "citation-recall": null, "fact-coverage": null}

> Marktüberwachungsbehörden haben das Recht, von Betreibern von Hochrisiko-KI-Systemen Informationen zu erhalten, wenn diese den Verdacht haben, dass das System ein Risiko im Sinne von Artikel 79 Absatz 1 birgt. Sie müssen unverzüglich informiert werden, wenn Betreiber den Verdacht haben, dass die Verwendung des Systems ein solches Risiko birgt, und die Verwendung des Systems aussetzen. Zudem müssen Marktüberwachungsbehörden von schwerwiegenden Vorfällen unverzüglich informiert werden. [chunk:art26-p5]
