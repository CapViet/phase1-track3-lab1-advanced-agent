# Lab 16 Benchmark Report

## Metadata
- Dataset: multihop_eval.json
- Mode: llm:ollama:qwen2.5:3b
- Records: 224
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.9554 | 0.9732 | 0.0178 |
| Avg attempts | 1 | 1.0714 | 0.0714 |
| Avg token estimate | 598.58 | 675.81 | 77.23 |
| Avg latency (ms) | 6851.16 | 8441.98 | 1590.82 |

## Cost & Runtime (price = $0.0/1M tokens; local Ollama = free)
| Agent | Records | Total tokens | Total runtime (s) | Avg time/Q (s) | Est. cost (USD) |
|---|---:|---:|---:|---:|---:|
| ReAct | 112 | 67041 | 767.33 | 6.85 | 0.0 |
| Reflexion | 112 | 75691 | 945.5 | 8.44 | 0.0 |
| **Total** | 224 | 142732 | 1712.83 | 7.65 | 0.0 |

## Failure modes (by mode -> per-agent counts)
```json
{
  "none": {
    "react": 107,
    "total": 216,
    "reflexion": 109
  },
  "incomplete_multi_hop": {
    "react": 4,
    "total": 6,
    "reflexion": 2
  },
  "wrong_final_answer": {
    "react": 1,
    "total": 1
  },
  "looping": {
    "reflexion": 1,
    "total": 1
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Run mode: llm:ollama:qwen2.5:3b. ReAct (single attempt) reached EM=0.9554 while Reflexion (up to several self-reflection attempts) reached EM=0.9732, a delta of +0.0178. Reflexion improved exact-match accuracy. The most common failure modes were: incomplete_multi_hop (x6), wrong_final_answer (x1), looping (x1). Reflexion's gains come at a cost: on average it used +0.0714 extra attempts, +77.23 tokens, and +1590.82 ms of latency per question versus ReAct. Reflexion helps most when the first attempt stops after an intermediate hop (incomplete_multi_hop) or drifts to a wrong second-hop entity (entity_drift), because the reflector's next_strategy explicitly tells the actor to complete every hop and re-ground each step in the context. It helps least on 'wrong_final_answer' cases where the model is confidently wrong, and it can loop when reflections fail to change behaviour. The quality of the structured evaluator is the main ceiling on gains: noisy 0/1 judgments propagate into misleading reflections, so a stronger judge would likely widen the delta.
