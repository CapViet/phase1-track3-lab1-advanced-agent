# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_golden.json
- Mode: llm:ollama:qwen2.5:3b
- Records: 40
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.7 | 0.8 | 0.1 |
| Avg attempts | 1 | 1.5 | 0.5 |
| Avg token estimate | 627.1 | 1194.3 | 567.2 |
| Avg latency (ms) | 445863.95 | 180077.55 | -265786.4 |

## Cost & Runtime (price = $0.0/1M tokens; local Ollama = free)
| Agent | Records | Total tokens | Total runtime (s) | Avg time/Q (s) | Est. cost (USD) |
|---|---:|---:|---:|---:|---:|
| ReAct | 20 | 12542 | 8917.28 | 445.86 | 0.0 |
| Reflexion | 20 | 23886 | 3601.55 | 180.08 | 0.0 |
| **Total** | 40 | 36428 | 12518.83 | 312.97 | 0.0 |

## Failure modes (by mode -> per-agent counts)
```json
{
  "none": {
    "react": 14,
    "total": 30,
    "reflexion": 16
  },
  "incomplete_multi_hop": {
    "react": 5,
    "total": 7,
    "reflexion": 2
  },
  "looping": {
    "reflexion": 2,
    "total": 2
  },
  "entity_drift": {
    "react": 1,
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
Run mode: llm:ollama:qwen2.5:3b. ReAct (single attempt) reached EM=0.7 while Reflexion (up to several self-reflection attempts) reached EM=0.8, a delta of +0.1. Reflexion improved exact-match accuracy. The most common failure modes were: incomplete_multi_hop (x7), looping (x2), entity_drift (x1). Reflexion's gains come at a cost: on average it used +0.5 extra attempts, +567.2 tokens, and -265786.4 ms of latency per question versus ReAct. Reflexion helps most when the first attempt stops after an intermediate hop (incomplete_multi_hop) or drifts to a wrong second-hop entity (entity_drift), because the reflector's next_strategy explicitly tells the actor to complete every hop and re-ground each step in the context. It helps least on 'wrong_final_answer' cases where the model is confidently wrong, and it can loop when reflections fail to change behaviour. The quality of the structured evaluator is the main ceiling on gains: noisy 0/1 judgments propagate into misleading reflections, so a stronger judge would likely widen the delta.
