from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import organize_quiz


def write_result(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"status": "ok", "items": [{"item": item} for item in items]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_result_question_ids_follow_dataset_source_and_item_number(tmp_path: Path) -> None:
    path = tmp_path / "DGLquiz" / "result_1_2.json"
    write_result(
        path,
        [
            {"stem": "Question one", "explanation": "Explanation one"},
            {"stem": "Question two", "explanation": "Explanation two"},
            {
                "stem": "Question three",
                "options": {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
                "answer": "C",
                "explanation": "Explanation three",
            },
        ],
    )

    output = organize_quiz.organize_one(path, 3, output_dir=tmp_path / "zhengliquiz")

    assert output.name == "preview_DGL-1_2-3.txt"
    assert output.read_text(encoding="utf-8") == (
        "ID: DGL-1_2-3\n"
        "Question: Question three\n"
        "Options:\n"
        "A. First\n"
        "B. Second\n"
        "C. Third\n"
        "D. Fourth\n"
        "Answer: C. Third\n"
        "Explanation: Explanation three\n"
    )


def test_organize_all_uses_dataset_then_natural_file_order(tmp_path: Path) -> None:
    write_result(
        tmp_path / "DGLquiz" / "result_1_10.json",
        [{"stem": "DGL ten", "explanation": "DGL ten explanation"}],
    )
    write_result(
        tmp_path / "DGLquiz" / "result_1_2.json",
        [{"stem": "DGL two", "explanation": "DGL two explanation"}],
    )
    write_result(
        tmp_path / "MLRquiz" / "result_1_1.json",
        [
            {"stem": "MLR first", "explanation": "MLR first explanation"},
            {"stem": "MLR second", "explanation": "MLR second explanation"},
        ],
    )

    output, count = organize_quiz.organize_all(tmp_path)
    text = output.read_text(encoding="utf-8")

    assert output == tmp_path / "zhengliquiz" / "zhengliquiz.txt"
    assert count == 4
    assert text.index("ID: DGL-1_2-1") < text.index("ID: DGL-1_10-1")
    assert text.index("ID: DGL-1_10-1") < text.index("ID: MLR-1_1-1")
    assert text.count("ID:") == text.count("Question:") == text.count("Explanation:") == 4


def test_calculation_questions_include_all_question_facing_fields(tmp_path: Path) -> None:
    path = tmp_path / "MLRquiz" / "result_2_4.json"
    write_result(
        path,
        [
            {
                "item_type": "open_calculation",
                "stem": "Calculate the score.",
                "given": {"x": "2", "y": "3"},
                "ask": ["Find z."],
                "formula_used": "z = x + y",
                "final_answer": "5",
                "unit": "points",
                "solution_steps": ["Add x and y."],
                "explanation": "Substitute the given values into the formula.",
            }
        ],
    )

    output = organize_quiz.organize_one(path, 1, output_dir=tmp_path / "zhengliquiz")
    text = output.read_text(encoding="utf-8")

    for expected in (
        "Question type: calculation",
        "Given:\nx: 2\ny: 3",
        "Solve:\n- Find z.",
        "Formula: z = x + y",
        "Answer: 5",
        "Unit: points",
        "Solution steps:\n1. Add x and y.",
        "Explanation: Substitute the given values into the formula.",
    ):
        assert expected in text


def test_split_export_writes_sequential_groups_without_losing_records(tmp_path: Path) -> None:
    source = tmp_path / "zhengliquiz" / "zhengliquiz.txt"
    records = [
        f"ID: DGL-1_2-{number}\nQuestion: Question {number}\nExplanation: Explanation {number}"
        for number in range(1, 6)
    ]
    source.parent.mkdir()
    source.write_text("\n\n".join(records) + "\n", encoding="utf-8")

    outputs = organize_quiz.split_export(source, output_dir=source.parent, questions_per_file=2)

    assert [path.name for path in outputs] == [
        "zhengliquiz1.txt",
        "zhengliquiz2.txt",
        "zhengliquiz3.txt",
    ]
    assert [path.read_text(encoding="utf-8").count("ID:") for path in outputs] == [2, 2, 1]
    assert "ID: DGL-1_2-5" in outputs[-1].read_text(encoding="utf-8")


def test_organize_one_rejects_out_of_range_question_number(tmp_path: Path) -> None:
    path = tmp_path / "MLRquiz" / "result_2_4.json"
    write_result(path, [{"stem": "Only question", "explanation": "Only explanation"}])

    with pytest.raises(organize_quiz.QuizOrganizationError, match="only 1 question"):
        organize_quiz.organize_one(path, 2, output_dir=tmp_path / "zhengliquiz")


def test_organize_all_does_not_write_a_partial_export_on_validation_error(tmp_path: Path) -> None:
    write_result(
        tmp_path / "DGLquiz" / "result_1_2.json",
        [{"stem": "Valid question", "explanation": "Valid explanation"}],
    )
    write_result(
        tmp_path / "MLRquiz" / "result_1_1.json",
        [{"stem": "Missing explanation"}],
    )

    with pytest.raises(organize_quiz.QuizOrganizationError, match="missing explanation"):
        organize_quiz.organize_all(tmp_path)

    assert not (tmp_path / "zhengliquiz" / "zhengliquiz.txt").exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "Unable to read JSON"),
        (json.dumps({"items": [{"item": {"explanation": "missing stem"}}]}), "missing stem"),
        (json.dumps({"items": [{"item": {"stem": "missing explanation"}}]}), "missing explanation"),
    ],
)
def test_invalid_json_or_missing_required_fields_are_reported(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "DGLquiz" / "result_1_2.json"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")

    with pytest.raises(organize_quiz.QuizOrganizationError, match=message):
        organize_quiz.organize_one(path, 1, output_dir=tmp_path / "zhengliquiz")


def test_selected_pdf_groups_csv_rows_and_preserves_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_result(
        tmp_path / "DGLquiz" / "result_1_2.json",
        [
            {"stem": "DGL first", "explanation": "DGL explanation"},
            {"stem": "DGL second", "explanation": "DGL explanation"},
        ],
    )
    write_result(
        tmp_path / "MLRquiz" / "result_14_263.json",
        [{"stem": "MLR first", "explanation": "MLR explanation"}],
    )
    scores = tmp_path / "zhengliquiz" / "quiz_scores_selected.csv"
    scores.parent.mkdir()
    scores.write_text(
        "ID,Score\n"
        "DGL-1_2-2,8.0\n"
        "MLR-14_263-1,8.1\n"
        "DGL-1_2-1,8.2\n",
        encoding="utf-8",
    )
    calls: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []

    def fake_render_data(data: dict[str, Any], output_path: Path, **kwargs: Any) -> Path:
        calls.append((data, output_path, kwargs))
        return output_path

    monkeypatch.setattr(organize_quiz, "render_data", fake_render_data)
    outputs = organize_quiz.export_selected_pdfs(
        scores_path=scores,
        output_dir=tmp_path / "quizPDF",
        project_root=tmp_path,
    )

    assert [path.name for path in outputs] == ["DGL-1.pdf", "MLR-14.pdf"]
    assert [entry["item"]["stem"] for entry in calls[0][0]["items"]] == [
        "DGL second",
        "DGL first",
    ]
    assert calls[0][0]["title"] == "DGL Quiz: Chapter 1"
    assert calls[0][1] == tmp_path / "quizPDF" / "DGL-1.pdf"
    assert calls[0][2]["cover_title"] == "DGL Quiz: Chapter 1"


def test_selected_pdf_rejects_duplicate_or_out_of_range_ids(tmp_path: Path) -> None:
    write_result(
        tmp_path / "DGLquiz" / "result_1_1.json",
        [{"stem": "Question", "explanation": "Explanation"}],
    )
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "ID\nDGL-1_1-1\nDGL-1_1-1\n",
        encoding="utf-8",
    )
    with pytest.raises(organize_quiz.QuizOrganizationError, match="duplicate ID"):
        organize_quiz._read_selected_groups(scores, tmp_path)

    scores.write_text("ID\nDGL-1_1-2\n", encoding="utf-8")
    with pytest.raises(organize_quiz.QuizOrganizationError, match="only 1 question"):
        organize_quiz._read_selected_groups(scores, tmp_path)
