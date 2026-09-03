from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from qg_v2.config import Settings
from qg_v2.graph import KnowledgeGraph
from qg_v2.llm import LLMRouter, LLMError
from qg_v2.orchestrator import Orchestrator
from qg_v2.types import dumps_pretty
from qg_v2.render import RenderError, render_file


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qg", description="Controllable multiple-choice question generator")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate a question set for one concept")
    gen.add_argument("concept", help="Target concept name")
    gen.add_argument("--data", default="inputdata/MOOC", help="Input data directory (default: ./inputdata/MOOC)")
    gen.add_argument("--out", help="Output JSON path; prints to stdout when omitted")
    gen.add_argument("--mock", action="store_true", help="Use mock/rule mode without calling Qwen")
    gen.add_argument("--config", help="Optional TOML file overriding the model name and base URL")
    gen.add_argument("--env", default=".env", help="Path to .env (default: .env)")
    gen.add_argument("--max-attempts", type=int, default=3, help="Maximum validation attempts (default: 3)")
    gen.add_argument("--quiet", action="store_true", help="Suppress stage progress messages")
    gen.add_argument("--instruction-version", choices=["legacy-v1", "multisource-v2"], help="Generation instruction version (default: multisource-v2)")
    render = sub.add_parser("render", help="Render generated JSON as LaTeX or PDF")
    render.add_argument("input", help="Generated result JSON file")
    render.add_argument("--out", required=True, help="Output .tex or .pdf path")
    render.add_argument("--format", choices=["tex", "latex", "pdf"], help="Output format; inferred from the extension by default")
    render.add_argument("--title", help="Quiz title")
    render.add_argument("--with-answers", action="store_true", help="Include answers and explanations")
    render.add_argument("--xelatex", default="xelatex", help="XeLaTeX executable name or path")
    return parser


def cmd_generate(args: argparse.Namespace) -> int:
    settings = Settings.load(config_path=args.config, env_path=args.env, mock=args.mock)
    if args.instruction_version:
        settings.instruction_version = args.instruction_version
    graph = KnowledgeGraph.load(args.data)
    llm = LLMRouter(settings)
    def progress(message: str) -> None:
        if not args.quiet:
            print(f"[QG] {message}", file=sys.stderr, flush=True)

    orchestrator = Orchestrator(graph=graph, llm=llm, max_attempts=args.max_attempts, progress_callback=progress)
    out_path = Path(args.out).resolve() if args.out else None
    try:
        result = orchestrator.generate(args.concept)
    except LLMError as exc:
        print(f"LLM call failed: {exc}", file=sys.stderr)
        print("Check the model name, API key, and QWEN_BASE_URL in .env; use --mock to test without an API call.", file=sys.stderr)
        return 3
    if out_path:
        _relativize_asset_paths(result, out_path.parent)
    rendered = dumps_pretty(result)
    if args.out:
        assert out_path is not None
        out_path.parent.mkdir(parents=True, exist_ok=True) if out_path.parent != Path("") else None
        out_path.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(rendered)
    return 0 if result.get("status") == "ok" else 2


def _relativize_asset_paths(result: dict[str, object], base_dir: Path) -> None:
    def visit(value: object, parent_key: str = "") -> None:
        if isinstance(value, dict):
            if parent_key in {"image", "image_asset"} and isinstance(value.get("path"), str):
                image_path = Path(value["path"])
                if image_path.is_absolute():
                    value["path"] = Path(os.path.relpath(image_path, base_dir)).as_posix()
            for key, child in value.items():
                visit(child, str(key))
        elif isinstance(value, list):
            for child in value:
                visit(child, parent_key)

    visit(result)


def main(argv: list[str] | None = None) -> int:
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "render":
        try:
            output = render_file(args.input, args.out, fmt=args.format, title=args.title,
                                 with_answers=args.with_answers, xelatex=args.xelatex)
        except RenderError as exc:
            print(f"Rendering failed: {exc}", file=sys.stderr)
            return 4
        print(output)
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

