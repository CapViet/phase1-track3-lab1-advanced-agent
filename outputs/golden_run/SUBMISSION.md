# Lab 16 — Golden Test Set Submission

**Dataset:** `data/hotpot_golden.json` (20 multi-hop QA examples, unseen)
**Model:** Ollama `qwen2.5:3b` (local, real LLM mode)
**Reflexion budget:** up to 3 attempts
**Records:** 40 (20 examples × 2 agents)
**Reproduce:**
```bash
python run_benchmark.py --dataset data/hotpot_golden.json --out-dir outputs/golden_run --mode real --provider ollama
python autograde.py --report-path outputs/golden_run/report.json
```

## Headline result

| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| **Exact match (EM)** | **0.70** | **0.80** | **+0.10** |
| Avg attempts | 1.00 | 1.50 | +0.50 |
| Avg tokens | 627.1 | 1194.3 | +567.2 |

**Reflexion improved exact-match accuracy by 10 points on unseen data.** This is a larger gain than on the development set (+1.8 pts), as expected: the golden set is harder (ReAct baseline 0.70 vs 0.955 on dev), leaving more headroom for self-reflection to recover answers.

> **Note on latency:** wall-clock timings this run were heavily inflated by machine/Ollama load and are **not** representative — they are omitted here. EM and token counts are unaffected and valid.

## Failure modes (count by mode → per agent)

| Mode | ReAct | Reflexion |
|---|---:|---:|
| none (correct) | 14 | 16 |
| incomplete_multi_hop | 5 | 2 |
| looping | 0 | 2 |
| entity_drift | 1 | 0 |

**Where Reflexion helped:** it recovered **3 of 5 `incomplete_multi_hop`** cases (the actor stopped at the intermediate hop) and the 1 `entity_drift` case — the reflector's `next_strategy` tells the actor to complete every hop and re-ground each step in the context.

**Where it cost:** it introduced 2 `looping` cases where repeated reflections failed to change the answer. This is the known ceiling — reflection (and evaluator) quality caps the achievable gain.

## Bonus extensions implemented

- **structured_evaluator** — judge returns structured `JudgeResult` (missing_evidence / spurious_claims).
- **reflection_memory** — each reflection's `next_strategy` is fed back to the Actor on the next attempt.
- **benchmark_report_json** — full `report.json` + `report.md`.
- **mock_mode_for_autograding** — deterministic mock runtime preserved alongside the real one.

## Autograde

**90/100** — Schema 30/30 · Experiment 20/30 · Analysis 20/20 · Bonus 20/20.
(The Experiment deduction reflects the small 20-example golden set; the full `multihop_eval.json` run scores that section in full.)
