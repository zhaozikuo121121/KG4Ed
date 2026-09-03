from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BatchConfigurationError(ValueError):
    """Raised when the batch input files are missing or malformed."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    project_root: Path
    concepts_path: Path
    metadata_path: Path
    data_dir: Path
    output_dir: Path
    expected_chapters: int


@dataclass(frozen=True)
class GenerationJob:
    chapter: int
    row_number: int
    concept: str
    output_path: Path


@dataclass(frozen=True)
class FailedJob:
    job: GenerationJob
    return_code: int


@dataclass(frozen=True)
class BatchSummary:
    total: int
    generated: int
    skipped: int
    failures: tuple[FailedJob, ...]

    @property
    def exit_code(self) -> int:
        return 1 if self.failures else 0


CommandRunner = Callable[[Sequence[str], Path], int]
CHAPTER_SELECTOR_RE = re.compile(r"(?P<start>\d+)(?:-(?P<end>\d+))?")


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def get_dataset_config(dataset: str, project_root: Path = PROJECT_ROOT) -> DatasetConfig:
    name = dataset.upper()
    chapter_counts = {"DGL": 6, "MLR": 14}
    if name not in chapter_counts:
        raise BatchConfigurationError(f"Unsupported dataset: {dataset}; choose DGL or MLR.")

    root = project_root.resolve()
    return DatasetConfig(
        name=name,
        project_root=root,
        concepts_path=root / "inputdata" / name / f"{name}_concepts.csv",
        metadata_path=root / "inputdata" / name / f"{name}_concepts_metadata.csv",
        data_dir=root / "inputdata" / name,
        output_dir=root / f"{name}quiz",
        expected_chapters=chapter_counts[name],
    )


def _read_concepts(concepts_path: Path) -> list[str]:
    if not concepts_path.is_file():
        raise BatchConfigurationError(f"Concept file not found: {concepts_path}")

    concepts: list[str] = []
    try:
        with concepts_path.open("r", encoding="utf-8-sig", newline="") as concepts_file:
            for row in csv.reader(concepts_file):
                concepts.append(row[0].strip() if row else "")
    except csv.Error as exc:
        raise BatchConfigurationError(f"Invalid concept CSV {concepts_path}: {exc}") from exc
    return concepts


def _read_concept_chapters(metadata_path: Path, expected_chapters: int) -> dict[str, int]:
    """Map canonical concepts to chapters using the generated metadata table."""
    if not metadata_path.is_file():
        raise BatchConfigurationError(f"Concept metadata file not found: {metadata_path}")
    chapters: dict[str, int] = {}
    try:
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
            for row in csv.DictReader(metadata_file):
                concept = (row.get("concept") or "").strip()
                raw_chapter = (row.get("chapter") or row.get("lecture") or "").strip()
                if not concept or not raw_chapter:
                    continue
                try:
                    chapter = int(raw_chapter)
                except ValueError:
                    continue
                if 1 <= chapter <= expected_chapters:
                    chapters.setdefault(concept, chapter)
    except (OSError, csv.Error) as exc:
        raise BatchConfigurationError(f"Invalid concept metadata CSV {metadata_path}: {exc}") from exc
    if not chapters:
        raise BatchConfigurationError(f"Concept metadata contains no valid chapter assignments: {metadata_path}")
    return chapters


def build_jobs(config: DatasetConfig) -> list[GenerationJob]:
    """Read and fully validate batch inputs before any command is launched."""
    concepts = _read_concepts(config.concepts_path)
    concept_chapters = _read_concept_chapters(config.metadata_path, config.expected_chapters)
    jobs: list[GenerationJob] = []
    for row_number, concept in enumerate(concepts, start=1):
        if not concept:
            continue
        chapter = concept_chapters.get(concept)
        if chapter is None:
            raise BatchConfigurationError(f"Concept CSV row {row_number} has no chapter in {config.metadata_path.name}: {concept}")
        jobs.append(GenerationJob(chapter=chapter, row_number=row_number, concept=concept,
                                  output_path=config.output_dir / f"result_{chapter}_{row_number}.json"))
    return jobs


def parse_chapter_selection(
    selectors: Sequence[str],
    expected_chapters: int,
) -> tuple[int, ...]:
    """Convert arguments such as ``1 3`` or ``1-3,5`` into chapter numbers."""
    selected: set[int] = set()

    for selector in selectors:
        for value in selector.split(","):
            value = value.strip()
            match = CHAPTER_SELECTOR_RE.fullmatch(value)
            if not match:
                raise BatchConfigurationError(
                    f"Invalid chapter argument {value!r}. Use forms such as 1, 1 3, or 1-3."
                )

            start = int(match.group("start"))
            end = int(match.group("end") or start)
            if start > end:
                raise BatchConfigurationError(
                    f"Invalid chapter range {value!r}: the start cannot exceed the end."
                )
            if start < 1 or end > expected_chapters:
                raise BatchConfigurationError(
                    f"Chapter {value!r} is out of range; this dataset supports chapters 1 through {expected_chapters}."
                )
            selected.update(range(start, end + 1))

    return tuple(sorted(selected))


def select_chapter_jobs(
    jobs: Sequence[GenerationJob], chapters: Sequence[int],
) -> list[GenerationJob]:
    """Keep jobs for selected chapters in concept-table order."""
    if not chapters:
        return list(jobs)
    selected_chapters = set(chapters)
    return [job for job in jobs if job.chapter in selected_chapters]


def is_successful_output(output_path: Path) -> bool:
    if not output_path.is_file():
        return False
    try:
        with output_path.open("r", encoding="utf-8-sig") as result_file:
            result = json.load(result_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(result, dict) and result.get("status") == "ok"


def resolve_qg_executable(command: str = "qg") -> str:
    executable = shutil.which(command)
    if executable is None:
        raise BatchConfigurationError(
            "The qg command was not found. Install the project and verify that this shell can run "
            "`qg generate ...`。"
        )
    return executable


def build_command(
    executable: str,
    config: DatasetConfig,
    job: GenerationJob,
) -> list[str]:
    data_dir = config.data_dir.relative_to(config.project_root).as_posix()
    output_path = job.output_path.relative_to(config.project_root).as_posix()
    return [
        executable,
        "generate",
        job.concept,
        "--data",
        data_dir,
        "--out",
        output_path,
    ]


def run_command(command: Sequence[str], cwd: Path) -> int:
    return subprocess.run(list(command), cwd=cwd, check=False).returncode


def run_batch(
    config: DatasetConfig,
    jobs: Sequence[GenerationJob],
    executable: str,
    *,
    force: bool = False,
    runner: CommandRunner = run_command,
) -> BatchSummary:
    generated = 0
    skipped = 0
    failures: list[FailedJob] = []
    total = len(jobs)

    for index, job in enumerate(jobs, start=1):
        identity = f"chapter {job.chapter} / CSV row {job.row_number} / {job.concept}"
        if not force and is_successful_output(job.output_path):
            skipped += 1
            print(f"[{index}/{total}] Skipped (successful result exists): {identity}", flush=True)
            continue

        print(f"[{index}/{total}] Generating: {identity}", flush=True)
        command = build_command(executable, config, job)
        try:
            return_code = runner(command, config.project_root)
        except OSError as exc:
            print(f"[{index}/{total}] Execution failed: {exc}", file=sys.stderr, flush=True)
            return_code = 127

        if return_code == 0 and is_successful_output(job.output_path):
            generated += 1
            print(f"[{index}/{total}] Generated: {job.output_path.name}", flush=True)
        else:
            failures.append(FailedJob(job=job, return_code=return_code))
            print(
                f"[{index}/{total}] Generation failed (exit code {return_code}): {identity}",
                file=sys.stderr,
                flush=True,
            )

    summary = BatchSummary(
        total=total,
        generated=generated,
        skipped=skipped,
        failures=tuple(failures),
    )
    print_summary(config, summary)
    return summary


def print_summary(config: DatasetConfig, summary: BatchSummary) -> None:
    print("\nBatch generation complete.")
    print(f"Dataset: {config.name}")
    print(f"Total: {summary.total}")
    print(f"Generated: {summary.generated}")
    print(f"Skipped existing successful results: {summary.skipped}")
    print(f"Failed: {len(summary.failures)}")
    if summary.failures:
        print("Failed jobs:", file=sys.stderr)
        for failure in summary.failures:
            job = failure.job
            print(
                f"  - chapter {job.chapter}, CSV row {job.row_number}, "
                f"{job.concept} (exit code {failure.return_code})",
                file=sys.stderr,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate questions for concepts assigned to DGL or MLR chapters."
    )
    parser.add_argument(
        "dataset",
        type=str.upper,
        choices=("DGL", "MLR"),
        help="Dataset to generate: DGL or MLR",
    )
    parser.add_argument(
        "chapters",
        nargs="*",
        metavar="CHAPTER",
        help=(
            "Optional chapters, for example 2, 1 3 5, 1,3,5, or 1-3; "
            "omit to generate all chapters."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all jobs, including existing status=ok results",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_console_encoding()
    args = build_parser().parse_args(argv)
    try:
        config = get_dataset_config(args.dataset)
        selected_chapters = parse_chapter_selection(
            args.chapters,
            config.expected_chapters,
        )
        jobs = select_chapter_jobs(build_jobs(config), selected_chapters)
        executable = resolve_qg_executable()
        chapter_scope = (
            ", ".join(map(str, selected_chapters)) if selected_chapters else "all chapters"
        )
        print(
            f"Preflight complete: {config.name}, chapters {chapter_scope}, "
            f"{len(jobs)} generation jobs."
        )
        summary = run_batch(config, jobs, executable, force=args.force)
        return summary.exit_code
    except BatchConfigurationError as exc:
        print(f"Batch generation configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user. Completed results are preserved and the command can be rerun.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
