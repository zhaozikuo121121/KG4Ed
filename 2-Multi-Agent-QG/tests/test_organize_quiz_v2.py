from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import organize_quiz_v2


def write_result(path: Path, stems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "items": [
                    {
                        "item": {
                            "stem": stem,
                            "answer": "A",
                            "explanation": f"Explanation for {stem}",
                        },
                        "metadata": {"source": stem},
                    }
                    for stem in stems
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_v2_selection_maps_files_groups_chapters_and_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_result(
        tmp_path / "DGLquiz" / "result_1_16.json",
        [f"DGL 16 question {number}" for number in range(1, 6)],
    )
    write_result(
        tmp_path / "DGLquiz" / "result_1_19.json",
        [f"DGL 19 question {number}" for number in range(1, 4)],
    )
    write_result(
        tmp_path / "MLRquiz" / "result_10_200.json",
        [f"MLR 200 question {number}" for number in range(1, 5)],
    )
    scores = tmp_path / "zhengliquiz" / "quiz_scores_selected_v2.csv"
    scores.parent.mkdir()
    scores.write_text(
        "DGL-16:1345\nDGL-19:2\n\nMLR-200:234\n",
        encoding="utf-8",
    )
    calls: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []

    def fake_render_data(data: dict[str, Any], output_path: Path, **kwargs: Any) -> Path:
        calls.append((data, output_path, kwargs))
        return output_path

    monkeypatch.setattr(organize_quiz_v2, "render_data", fake_render_data)
    outputs = organize_quiz_v2.export_selected_pdfs(
        project_root=tmp_path,
        with_answers=True,
    )

    assert [path.name for path in outputs] == ["DGL-1.pdf", "MLR-10.pdf"]
    assert [entry["item"]["stem"] for entry in calls[0][0]["items"]] == [
        "DGL 16 question 1",
        "DGL 16 question 3",
        "DGL 16 question 4",
        "DGL 16 question 5",
        "DGL 19 question 2",
    ]
    assert [entry["item"]["stem"] for entry in calls[1][0]["items"]] == [
        "MLR 200 question 2",
        "MLR 200 question 3",
        "MLR 200 question 4",
    ]
    assert calls[0][0]["title"] == "DGL Quiz: Chapter 1"
    assert calls[0][1] == tmp_path / "quizPDFv2" / "DGL-1.pdf"
    assert calls[0][2] == {
        "fmt": "pdf",
        "title": "DGL Quiz: Chapter 1",
        "cover_title": "DGL Quiz: Chapter 1",
        "with_answers": True,
        "base_dir": tmp_path,
        "xelatex": "xelatex",
    }


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ("invalid", "invalid ID format"),
        ("DGL-16:11", "duplicate question number"),
        ("DGL-16:4", "only 3 questions"),
    ],
)
def test_v2_selection_rejects_invalid_rows(
    tmp_path: Path,
    selection: str,
    message: str,
) -> None:
    write_result(
        tmp_path / "DGLquiz" / "result_1_16.json",
        ["Question 1", "Question 2", "Question 3"],
    )
    scores = tmp_path / "scores.csv"
    scores.write_text(f"{selection}\n", encoding="utf-8")

    with pytest.raises(organize_quiz_v2.QuizOrganizationError, match=message):
        organize_quiz_v2._read_selected_groups(scores, tmp_path)


def test_v2_selection_rejects_missing_ambiguous_and_duplicate_sources(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "scores.csv"
    scores.write_text("DGL-16:1\n", encoding="utf-8")
    with pytest.raises(organize_quiz_v2.QuizOrganizationError, match="matching JSON not found"):
        organize_quiz_v2._read_selected_groups(scores, tmp_path)

    write_result(tmp_path / "DGLquiz" / "result_1_16.json", ["Question 1"])
    write_result(tmp_path / "DGLquiz" / "result_2_16.json", ["Question 1"])
    with pytest.raises(organize_quiz_v2.QuizOrganizationError, match="multiple matching JSON"):
        organize_quiz_v2._read_selected_groups(scores, tmp_path)

    (tmp_path / "DGLquiz" / "result_2_16.json").unlink()
    scores.write_text("DGL-16:1\nDGL-16:1\n", encoding="utf-8")
    with pytest.raises(organize_quiz_v2.QuizOrganizationError, match="duplicate ID"):
        organize_quiz_v2._read_selected_groups(scores, tmp_path)

    scores.write_text("DGL-16:1\nDGL-16:2\n", encoding="utf-8")
    with pytest.raises(organize_quiz_v2.QuizOrganizationError, match="duplicate JSON ID"):
        organize_quiz_v2._read_selected_groups(scores, tmp_path)


def test_v2_selection_rejects_invalid_json(tmp_path: Path) -> None:
    source = tmp_path / "DGLquiz" / "result_1_16.json"
    source.parent.mkdir()
    source.write_text("not json", encoding="utf-8")
    scores = tmp_path / "scores.csv"
    scores.write_text("DGL-16:1\n", encoding="utf-8")

    with pytest.raises(organize_quiz_v2.QuizOrganizationError, match="Unable to read JSON"):
        organize_quiz_v2._read_selected_groups(scores, tmp_path)


def test_v2_main_forwards_selected_pdf_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_export_selected_pdfs(**kwargs: Any) -> list[Path]:
        calls.append(kwargs)
        return [tmp_path / "quizPDFv2" / "DGL-1.pdf"]

    monkeypatch.setattr(organize_quiz_v2, "export_selected_pdfs", fake_export_selected_pdfs)

    assert organize_quiz_v2.main(["--selected-pdf", "--with-answers"]) == 0
    assert calls == [
        {
            "scores_path": None,
            "output_dir": None,
            "with_answers": True,
            "xelatex": "xelatex",
        }
    ]
