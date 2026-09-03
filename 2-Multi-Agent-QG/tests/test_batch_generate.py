from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Sequence

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import batch_generate


def make_config(
    tmp_path: Path,
    *,
    concepts: str,
    metadata: str,
    expected_chapters: int = 1,
    name: str = "DGL",
) -> batch_generate.DatasetConfig:
    input_dir = tmp_path / "inputdata" / name
    input_dir.mkdir(parents=True)
    concepts_path = input_dir / f"{name}_concepts.csv"
    metadata_path = input_dir / f"{name}_concepts_metadata.csv"
    concepts_path.write_text(concepts, encoding="utf-8")
    metadata_path.write_text(metadata, encoding="utf-8")
    return batch_generate.DatasetConfig(
        name=name,
        project_root=tmp_path,
        concepts_path=concepts_path,
        metadata_path=metadata_path,
        data_dir=input_dir,
        output_dir=tmp_path / f"{name}quiz",
        expected_chapters=expected_chapters,
    )


def successful_runner(calls: list[list[str]]):
    def run(command: Sequence[str], cwd: Path) -> int:
        command_list = list(command)
        calls.append(command_list)
        output_arg = command_list[command_list.index("--out") + 1]
        output_path = cwd / output_arg
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        return 0

    return run


def test_real_metadata_maps_all_concepts_in_canonical_order() -> None:
    dgl_config = batch_generate.get_dataset_config("DGL", PROJECT_ROOT)
    dgl_jobs = batch_generate.build_jobs(dgl_config)
    assert len(dgl_jobs) == 392
    assert (dgl_jobs[0].chapter, dgl_jobs[0].row_number, dgl_jobs[0].concept) == (
        1,
        1,
        "adjacency matrix",
    )
    assert (dgl_jobs[1].chapter, dgl_jobs[1].row_number, dgl_jobs[1].concept) == (
        1,
        2,
        "graph representation",
    )
    assert dgl_jobs[0].output_path.name == "result_1_1.json"
    assert dgl_jobs[1].output_path.name == "result_1_2.json"

    mlr_config = batch_generate.get_dataset_config("mlr", PROJECT_ROOT)
    mlr_jobs = batch_generate.build_jobs(mlr_config)
    assert len(mlr_jobs) == 285
    assert (mlr_jobs[0].chapter, mlr_jobs[0].row_number, mlr_jobs[0].concept) == (
        1,
        1,
        "machine learning",
    )
    assert (mlr_jobs[-1].chapter, mlr_jobs[-1].row_number) == (14, 285)


def test_build_command_matches_required_invocation() -> None:
    config = batch_generate.get_dataset_config("DGL", PROJECT_ROOT)
    job = batch_generate.build_jobs(config)[0]
    assert batch_generate.build_command("qg", config, job) == [
        "qg",
        "generate",
        "adjacency matrix",
        "--data",
        "inputdata/DGL",
        "--out",
        "DGLquiz/result_1_1.json",
    ]


def test_metadata_assigns_chapter_to_quoted_csv_concept(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        concepts='"concept, with comma",alias\n',
        metadata="concept,chapter\n\"concept, with comma\",1\n",
    )
    jobs = batch_generate.build_jobs(config)
    assert [(job.chapter, job.row_number, job.concept) for job in jobs] == [
        (1, 1, "concept, with comma")
    ]


def test_successful_existing_output_is_skipped_unless_force_is_used(tmp_path: Path) -> None:
    config = make_config(tmp_path, concepts="graph representation,alias\n", metadata="concept,chapter\ngraph representation,1\n")
    jobs = batch_generate.build_jobs(config)
    jobs[0].output_path.parent.mkdir(parents=True)
    jobs[0].output_path.write_text('{"status":"ok"}', encoding="utf-8")
    calls: list[list[str]] = []

    summary = batch_generate.run_batch(
        config,
        jobs,
        "qg",
        runner=successful_runner(calls),
    )
    assert (summary.generated, summary.skipped, summary.failures) == (0, 1, ())
    assert calls == []

    forced_summary = batch_generate.run_batch(
        config,
        jobs,
        "qg",
        force=True,
        runner=successful_runner(calls),
    )
    assert (forced_summary.generated, forced_summary.skipped, forced_summary.failures) == (1, 0, ())
    assert len(calls) == 1


@pytest.mark.parametrize("existing_content", ["not json", '{"status":"failed"}'])
def test_invalid_or_failed_existing_output_is_retried(tmp_path: Path, existing_content: str) -> None:
    config = make_config(tmp_path, concepts="concept\n", metadata="concept,chapter\nconcept,1\n")
    jobs = batch_generate.build_jobs(config)
    jobs[0].output_path.parent.mkdir(parents=True)
    jobs[0].output_path.write_text(existing_content, encoding="utf-8")
    calls: list[list[str]] = []

    summary = batch_generate.run_batch(
        config,
        jobs,
        "qg",
        runner=successful_runner(calls),
    )
    assert summary.generated == 1
    assert summary.skipped == 0
    assert not summary.failures
    assert len(calls) == 1


def test_failed_job_is_recorded_and_later_jobs_continue(tmp_path: Path) -> None:
    config = make_config(tmp_path, concepts="first\nsecond\n", metadata="concept,chapter\nfirst,1\nsecond,1\n")
    jobs = batch_generate.build_jobs(config)
    calls: list[list[str]] = []

    def runner(command: Sequence[str], cwd: Path) -> int:
        command_list = list(command)
        calls.append(command_list)
        if len(calls) == 1:
            return 7
        return successful_runner([])(command_list, cwd)

    summary = batch_generate.run_batch(config, jobs, "qg", runner=runner)
    assert len(calls) == 2
    assert summary.generated == 1
    assert summary.skipped == 0
    assert len(summary.failures) == 1
    assert summary.failures[0].job.concept == "first"
    assert summary.failures[0].return_code == 7
    assert summary.exit_code == 1


@pytest.mark.parametrize("concepts, metadata, message", [
    ("concept\n", "concept,chapter\n", "has no chapter"),
    ("concept\n", "concept,chapter\nconcept,x\n", "has no chapter"),
])
def test_preflight_rejects_invalid_inputs(
    tmp_path: Path,
    concepts: str,
    metadata: str,
    message: str,
) -> None:
    config = make_config(tmp_path, concepts=concepts, metadata=metadata)
    with pytest.raises(batch_generate.BatchConfigurationError, match=message):
        batch_generate.build_jobs(config)


def test_missing_qg_executable_has_clear_error() -> None:
    with pytest.raises(batch_generate.BatchConfigurationError, match="qg command was not found"):
        batch_generate.resolve_qg_executable("definitely-not-a-real-qg-command")


@pytest.mark.parametrize(
    ("selectors", "expected"),
    [
        (["2"], (2,)),
        (["1", "3", "5"], (1, 3, 5)),
        (["1,3,5"], (1, 3, 5)),
        (["1-3", "5"], (1, 2, 3, 5)),
        (["3", "1-3"], (1, 2, 3)),
    ],
)
def test_parse_chapter_selection_accepts_individual_lists_and_ranges(
    selectors: list[str],
    expected: tuple[int, ...],
) -> None:
    assert batch_generate.parse_chapter_selection(selectors, 6) == expected


@pytest.mark.parametrize("selector", ["0", "7", "3-1", "one", "1,,2"])
def test_parse_chapter_selection_rejects_invalid_values(selector: str) -> None:
    with pytest.raises(batch_generate.BatchConfigurationError):
        batch_generate.parse_chapter_selection([selector], 6)


def test_select_chapter_jobs_keeps_concept_table_order(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        concepts="first\nsecond\nthird\n",
        metadata="concept,chapter\nfirst,1\nsecond,2\nthird,3\n",
        expected_chapters=3,
    )
    jobs = batch_generate.build_jobs(config)

    selected = batch_generate.select_chapter_jobs(jobs, (3, 1))
    assert [(job.chapter, job.concept) for job in selected] == [(1, "first"), (3, "third")]
    assert batch_generate.select_chapter_jobs(jobs, ()) == jobs


def test_parser_accepts_optional_chapter_arguments() -> None:
    args = batch_generate.build_parser().parse_args(["DGL", "1-3,5", "--force"])
    assert args.dataset == "DGL"
    assert args.chapters == ["1-3,5"]
    assert args.force is True
