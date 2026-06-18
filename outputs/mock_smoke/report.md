# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: mock
- Records: 16
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.5 | 1.0 | 0.5 |
| Avg attempts | 1 | 1.5 | 0.5 |
| Avg token estimate | 485 | 1045 | 560 |
| Avg latency (ms) | 270 | 615 | 345 |

## Failure modes (by mode -> per-agent counts)
```json
{
  "none": {
    "react": 4,
    "total": 12,
    "reflexion": 8
  },
  "entity_drift": {
    "react": 2,
    "total": 2
  },
  "incomplete_multi_hop": {
    "react": 1,
    "total": 1
  },
  "wrong_final_answer": {
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
Run mode: mock. ReAct (single attempt) reached EM=0.5 while Reflexion (up to several self-reflection attempts) reached EM=1.0, a delta of +0.5. Reflexion improved exact-match accuracy. The most common failure modes were: entity_drift (x2), incomplete_multi_hop (x1), wrong_final_answer (x1). Reflexion's gains come at a cost: on average it used +0.5 extra attempts, +560 tokens, and +345 ms of latency per question versus ReAct. Reflexion helps most when the first attempt stops after an intermediate hop (incomplete_multi_hop) or drifts to a wrong second-hop entity (entity_drift), because the reflector's next_strategy explicitly tells the actor to complete every hop and re-ground each step in the context. It helps least on 'wrong_final_answer' cases where the model is confidently wrong, and it can loop when reflections fail to change behaviour. The quality of the structured evaluator is the main ceiling on gains: noisy 0/1 judgments propagate into misleading reflections, so a stronger judge would likely widen the delta.
