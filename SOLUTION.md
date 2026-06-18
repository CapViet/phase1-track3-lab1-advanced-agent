# Lab 16 — Reflexion Agent: Implementation Notes

This document records how the scaffold was completed and how to reproduce the results.

## What was implemented

| Step | File | What changed |
|---|---|---|
| 1. Schemas | `src/reflexion_lab/schemas.py` | `JudgeResult` (score, reason, missing_evidence, spurious_claims) and `ReflectionEntry` (attempt_id, failure_reason, lesson, next_strategy). |
| 2. Reflexion loop | `src/reflexion_lab/agents.py` | Act → Evaluate → (if wrong & reflexion & attempts left) Reflect → push `next_strategy` into `reflection_memory` for the next attempt. Token/latency now come from **real** usage, not hardcoded. |
| 3. Prompts | `src/reflexion_lab/prompts.py` | `ACTOR_SYSTEM`, `EVALUATOR_SYSTEM` (strict JSON judge), `REFLECTOR_SYSTEM` (strict JSON reflection). |
| 4. Real LLM | `src/reflexion_lab/llm.py`, `llm_runtime.py` | Provider-agnostic client (Ollama/OpenAI/Gemini/Anthropic) returning real token counts + measured latency; the three roles backed by a real model with robust JSON parsing and exact-match fallback. |
| 5. Mock kept | `src/reflexion_lab/mock_runtime.py` | Refactored into `MockRuntime` so the free/deterministic path still works (the `mock_mode_for_autograding` extension). |
| 6. Reporting | `src/reflexion_lab/reporting.py` | `failure_modes` keyed by mode (per-agent + total counts); data-driven `discussion`. |
| 7. Data | `scripts/make_dataset.py` → `data/multihop_eval.json` | 112 diverse multi-hop questions (8 templates), with distractor passages on medium/hard items. 112 examples × 2 agents = 224 records. |

## How to run

```bash
pip install -r requirements.txt

# Free, deterministic (good for understanding the flow / autograding):
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/mock_run --mode mock

# Real LLM (defaults to local Ollama qwen2.5:3b):
python scripts/make_dataset.py --out data/multihop_eval.json
python run_benchmark.py --dataset data/multihop_eval.json --out-dir outputs/real_run --mode real --provider ollama

# Switch backend by editing .env (see .env.example) or:
python run_benchmark.py --mode real --provider gemini --model gemini-2.0-flash --dataset data/multihop_eval.json

python autograde.py --report-path outputs/real_run/report.json
```

`--limit N` runs on the first N examples; `--reflexion-attempts K` sets the Reflexion budget.

## Bonus extensions implemented

- **structured_evaluator** — judge returns structured `JudgeResult` (missing_evidence / spurious_claims), not just 0/1.
- **reflection_memory** — each reflection's `next_strategy` is fed back to the Actor on the next attempt.
- **benchmark_report_json** — full `report.json` + `report.md`.
- **mock_mode_for_autograding** — deterministic mock runtime preserved alongside the real one.

## Results (real run, Ollama qwen2.5:3b, 112 examples → 224 records)

Reproduce: `python run_benchmark.py --dataset data/multihop_eval.json --out-dir outputs/real_run --mode real --provider ollama`

| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| Exact match (EM) | 0.9554 | 0.9732 | +0.0178 |
| Avg attempts | 1.00 | 1.07 | +0.07 |
| Avg tokens (real) | 598.6 | 675.8 | +77.2 |
| Avg latency ms (real) | 6851 | 8442 | +1591 |

Cost & runtime (local Ollama → $0; set `LLM_PRICE_PER_1M_TOKENS` for hosted models):

| Agent | Records | Total tokens | Total runtime | Avg/Q | Est. cost |
|---|---:|---:|---:|---:|---:|
| ReAct | 112 | 67,041 | 767 s | 6.85 s | $0.00 |
| Reflexion | 112 | 75,691 | 946 s | 8.44 s | $0.00 |
| **Total** | 224 | 142,732 | ~28.5 min | 7.65 s | $0.00 |

Failure modes (count by mode → per agent):

| Mode | ReAct | Reflexion |
|---|---:|---:|
| none (correct) | 107 | 109 |
| incomplete_multi_hop | 4 | 2 |
| wrong_final_answer | 1 | 0 |
| looping | 0 | 1 |

**Takeaways.** qwen2.5:3b already answers most 2-hop questions well (EM 0.955), so the headroom
for Reflexion is small. Within that headroom it helps where expected: it recovered 2 of the 4
`incomplete_multi_hop` cases (the model stopped at the intermediate hop) and the 1
`wrong_final_answer`, lifting EM to 0.973. The cost is real and measured: +0.07 attempts,
+77 tokens, +1.6 s latency per question. It also introduced one `looping` case where repeated
reflections failed to change the answer — a reminder that reflection quality (and evaluator
quality) caps the gain. A larger gap would appear with a weaker actor or harder questions.

**Autograde: 100/100** (Core 80/80, Bonus 20/20) — `python autograde.py --report-path outputs/real_run/report.json`.
