# Lab 16 Benchmark Report

## Metadata
- Dataset: multihop_eval.json
- Mode: llm:ollama:qwen2.5:3b
- Records: 4
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 1.0 | 1.0 | 0.0 |
| Avg attempts | 1 | 1 | 0 |
| Avg token estimate | 594.5 | 591.5 | -3.0 |
| Avg latency (ms) | 9079 | 5927 | -3152 |

## Failure modes (by mode -> per-agent counts)
```json
{
  "none": {
    "react": 2,
    "total": 4,
    "reflexion": 2
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Run mode: llm:ollama:qwen2.5:3b. ReAct (single attempt) reached EM=1.0 while Reflexion (up to several self-reflection attempts) reached EM=1.0, a delta of +0.0. Reflexion did not improve accuracy. The most common failure modes were: none observed. Reflexion's gains come at a cost: on average it used +0 extra attempts, -3.0 tokens, and -3152 ms of latency per question versus ReAct. Reflexion helps most when the first attempt stops after an intermediate hop (incomplete_multi_hop) or drifts to a wrong second-hop entity (entity_drift), because the reflector's next_strategy explicitly tells the actor to complete every hop and re-ground each step in the context. It helps least on 'wrong_final_answer' cases where the model is confidently wrong, and it can loop when reflections fail to change behaviour. The quality of the structured evaluator is the main ceiling on gains: noisy 0/1 judgments propagate into misleading reflections, so a stronger judge would likely widen the delta.
