from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import typer
from rich import print
from rich.progress import track
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = typer.Option("mock", help="Runtime: 'mock' (free, deterministic) or 'real' (live LLM)."),
    provider: Optional[str] = typer.Option(None, help="LLM provider for real mode: ollama|openai|gemini|anthropic."),
    model: Optional[str] = typer.Option(None, help="Model name override for real mode."),
    limit: Optional[int] = typer.Option(None, help="Only run on the first N examples of the dataset."),
) -> None:
    examples = load_dataset(dataset)
    if limit:
        examples = examples[:limit]

    # Build the runtime once and share it across both agents.
    if mode == "mock":
        from src.reflexion_lab.mock_runtime import MockRuntime

        runtime = MockRuntime()
        report_mode = "mock"
    else:
        from src.reflexion_lab.llm import LLMClient
        from src.reflexion_lab.llm_runtime import LLMRuntime

        client = LLMClient(provider=(provider or "ollama"), model=model)  # type: ignore[arg-type]
        runtime = LLMRuntime(client=client)
        report_mode = runtime.name

    react = ReActAgent(runtime=runtime)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, runtime=runtime)

    print(f"[cyan]Mode:[/cyan] {report_mode}  |  [cyan]Examples:[/cyan] {len(examples)}  |  [cyan]Reflexion attempts:[/cyan] {reflexion_attempts}")

    react_records = [react.run(ex) for ex in track(examples, description="ReAct     ")]
    reflexion_records = [reflexion.run(ex) for ex in track(examples, description="Reflexion ")]
    all_records = react_records + reflexion_records

    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=report_mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
