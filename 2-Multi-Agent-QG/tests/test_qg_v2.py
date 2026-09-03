from __future__ import annotations

import json
from pathlib import Path

import pytest

from qg_v2.config import Settings
from qg_v2.graph import KnowledgeGraph
from qg_v2.llm import LLMRouter
from qg_v2.orchestrator import Orchestrator
from qg_v2.agents import PlannerAgent, CognitiveDifficultyAgent, DistractorReviewAgent


@pytest.fixture()
def sample_data(tmp_path: Path) -> Path:
    (tmp_path / "similarity").mkdir()
    (tmp_path / "concepts.txt").write_text(
        "precision::;positive predictive value\n"
        "recall\n"
        "accuracy\n"
        "f1 score\n"
        "neural network\n"
        "hidden layer\n"
        "input layer\n"
        "gradient descent\n"
        "cost function\n"
        "cross validation\n"
        "cross validation error\n"
        "cross validation set\n"
        "hold-out validation\n"
        "resampling method\n"
        "model evaluation\n"
        "validation technique\n",
        encoding="utf-8",
    )
    profiles = [
        {"concept_id": 0, "concept_name": "precision", "profile": "Precision is the proportion of predicted positive examples that are truly positive. It is calculated as TP / (TP + FP)."},
        {"concept_id": 1, "concept_name": "recall", "profile": "Recall is the proportion of actual positive examples that are found."},
        {"concept_id": 2, "concept_name": "accuracy", "profile": "Accuracy is the proportion of all predictions that are correct."},
        {"concept_id": 3, "concept_name": "f1 score", "profile": "F1 score combines precision and recall."},
        {"concept_id": 4, "concept_name": "neural network", "profile": "A neural network is composed of layers and units."},
        {"concept_id": 5, "concept_name": "hidden layer", "profile": "A hidden layer is an internal layer of a neural network."},
        {"concept_id": 6, "concept_name": "input layer", "profile": "An input layer receives features."},
        {"concept_id": 7, "concept_name": "gradient descent", "profile": "Gradient descent optimizes parameters."},
        {"concept_id": 8, "concept_name": "cost function", "profile": "A cost function measures model error."},
        {"concept_id": 9, "concept_name": "cross validation", "profile": "Cross validation is a resampling method for evaluating generalization error."},
        {"concept_id": 10, "concept_name": "cross validation error", "profile": "Cross validation error is an estimated error value from cross validation."},
        {"concept_id": 11, "concept_name": "cross validation set", "profile": "A validation set is a subset used for validation."},
        {"concept_id": 12, "concept_name": "hold-out validation", "profile": "Hold-out validation is a model evaluation method using a held-out subset."},
        {"concept_id": 13, "concept_name": "resampling method", "profile": "A resampling method repeatedly samples data for evaluation."},
        {"concept_id": 14, "concept_name": "model evaluation", "profile": "Model evaluation estimates model performance."},
        {"concept_id": 15, "concept_name": "validation technique", "profile": "A validation technique is used to evaluate models."},
    ]
    (tmp_path / "concept_profiles.jsonl").write_text("\n".join(json.dumps(p) for p in profiles), encoding="utf-8")
    (tmp_path / "confusion.txt").write_text("precision :: recall\nprecision :: accuracy\ncross validation :: cross validation error\ncross validation :: cross validation set\ncross validation :: hold-out validation\n", encoding="utf-8")
    (tmp_path / "partof.txt").write_text("hidden layer part-of neural network\ninput layer part-of neural network\nresampling method part-of cross validation\n", encoding="utf-8")
    (tmp_path / "prerequisite.txt").write_text("cost function\t\tgradient descent\t\tmeta\nmodel evaluation\t\tcross validation\t\tmeta\ncross validation\t\thold-out validation\t\tmeta\nhold-out validation\t\tvalidation technique\t\tmeta\n", encoding="utf-8")
    (tmp_path / "similarity" / "edges_knn.csv").write_text(
        "source_id,target_id,source,target,weight\n0,3,precision,f1 score,0.9\n",
        encoding="utf-8",
    )
    return tmp_path


def test_graph_aliases_and_relation_directions(sample_data: Path) -> None:
    graph = KnowledgeGraph.load(sample_data)
    assert graph.resolve("positive predictive value") == "precision"
    assert graph.confusables_of("recall")[0].target == "precision"
    assert graph.parts_of("neural network")[0].source == "hidden layer"
    assert graph.wholes_of("hidden layer")[0].target == "neural network"
    assert graph.prereqs_of("gradient descent")[0].source == "cost function"
    assert graph.dependents_of("cost function")[0].target == "gradient descent"


def test_planner_priority_and_definition_fallback(sample_data: Path) -> None:
    graph = KnowledgeGraph.load(sample_data)
    planner = PlannerAgent(graph)
    assert planner.plan("precision")["question_type"] == "concept_discrimination"
    assert planner.plan("gradient descent")["question_type"] == "prerequisite_dependency"
    assert planner.plan("neural network")["question_type"] == "component_membership"
    assert planner.plan("precision", force_definition=True)["question_type"] == "definition"


def test_cognitive_mapping() -> None:
    agent = CognitiveDifficultyAgent()
    result = agent.decide(
        {"question_type": "concept_discrimination"},
        {"num_direct_prereqs": 0, "has_clean_2hop_chain": False, "num_part_components": 0, "has_strong_confusable": True},
    )
    assert result["bloom_candidates"] == ["analysis", "evaluation"]
    assert result["selected_bloom"] == "analysis"
    assert result["target_difficulty"] == "medium"


def test_orchestrator_mock_e2e(sample_data: Path) -> None:
    graph = KnowledgeGraph.load(sample_data)
    settings = Settings(mock=True)
    result = Orchestrator(graph, LLMRouter(settings)).generate("precision")
    assert result["status"] == "ok"
    assert "items" in result and result["items"]
    first = result["items"][0]
    assert set(first["item"]["options"]) == {"A", "B", "C", "D"}
    assert first["item"]["answer"] in {"A", "B", "C", "D"}
    assert first["metadata"]["distractor_review_result"]["passed"] is True
    assert first["metadata"]["course_scope_distractor_result"]["checked_once"] is True
    assert set(first["validation_record"]["checks"]) == {
        "unique_answer",
        "answerable",
        "on_topic",
        "clear_language",
        "options_exclusive",
        "distractors_valid",
        "bloom_aligned",
        "graph_faithful",
        "explanation_supported",
    }


def test_option_shuffle_remaps_explanation_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedRandom:
        @staticmethod
        def shuffle(values: list[str]) -> None:
            # New A/B/C/D contain old C/A/D/B respectively.
            values[:] = [values[2], values[0], values[3], values[1]]

    monkeypatch.setattr("qg_v2.orchestrator.random.SystemRandom", FixedRandom)
    item = {
        "stem": "Which component?",
        "options": {
            "A": "bias problem",
            "B": "variance problem",
            "C": "underfitting",
            "D": "bias unit",
        },
        "answer": "A",
        "explanation": (
            "The correct answer is A. Option B is variance problem; Option C is underfitting; "
            "Option D is bias unit. Correct answer is A; options B, C, and D are incorrect."
        ),
    }

    Orchestrator._shuffle_choice_options(item)

    assert item["options"] == {
        "A": "underfitting",
        "B": "bias problem",
        "C": "bias unit",
        "D": "variance problem",
    }
    assert item["answer"] == "B"
    assert item["explanation"] == (
        "The correct answer is B. Option D is variance problem; Option A is underfitting; "
        "Option C is bias unit. Correct answer is B; options D, A, and C are incorrect."
    )

    blueprint = {
        "writer_output": item,
        "answer_plan": {
            "correct_answer": {"concept": "bias problem"},
            "distractors": [
                {"concept": "variance problem"},
                {"concept": "underfitting"},
                {"concept": "bias unit"},
            ],
        },
    }
    metadata = Orchestrator.__new__(Orchestrator)._metadata(blueprint)
    assert set(metadata["distractor_plan"]) == {"option_A", "option_C", "option_D"}
    assert metadata["distractor_plan"]["option_A"]["concept"] == "underfitting"



def test_planner_plan_all_returns_all_feasible_types(sample_data: Path) -> None:
    graph = KnowledgeGraph.load(sample_data)
    planner = PlannerAgent(graph)
    result = planner.plan_all("cross validation")
    assert result["status"] == "ok"
    types = [plan["question_type"] for plan in result["plans"]]
    assert "concept_discrimination" in types
    assert "prerequisite_dependency" in types
    assert "component_membership" in types
    assert "multi_hop_reasoning" in types
    assert "definition" not in types


def test_planner_plan_all_definition_only_when_no_structure(tmp_path: Path) -> None:
    (tmp_path / "similarity").mkdir()
    (tmp_path / "concepts.txt").write_text("isolated concept\n", encoding="utf-8")
    (tmp_path / "concept_profiles.jsonl").write_text(
        json.dumps({"concept_id": 0, "concept_name": "isolated concept", "profile": "An isolated concept."}),
        encoding="utf-8",
    )
    (tmp_path / "confusion.txt").write_text("", encoding="utf-8")
    (tmp_path / "partof.txt").write_text("", encoding="utf-8")
    (tmp_path / "prerequisite.txt").write_text("", encoding="utf-8")
    (tmp_path / "similarity" / "edges_knn.csv").write_text("source_id,target_id,source,target,weight\n", encoding="utf-8")
    graph = KnowledgeGraph.load(tmp_path)
    result = PlannerAgent(graph).plan_all("isolated concept")
    assert [plan["question_type"] for plan in result["plans"]] == ["definition"]


def test_distractor_review_rejects_surface_eliminable_method_options(sample_data: Path) -> None:
    graph = KnowledgeGraph.load(sample_data)
    reviewer = DistractorReviewAgent(graph, LLMRouter(Settings(mock=True)))
    blueprint = {
        "writer_output": {
            "stem": "In model evaluation, what is the methodology of resampling into multiple folds and repeating training and validation called?",
            "options": {
                "A": "cross-validation",
                "B": "cross-validation error",
                "C": "cross-validation set",
                "D": "hold-out validation",
            },
            "answer": "A",
        },
        "answer_plan": {"correct_answer": {"concept": "cross validation", "basis": "method"}},
        "distractor_plan": {
            "distractors": [
                {"concept": "cross validation error", "source_relation": "similar/confusable"},
                {"concept": "cross validation set", "source_relation": "similar/confusable"},
                {"concept": "hold-out validation", "source_relation": "similar/confusable"},
            ]
        },
    }
    result = reviewer.review(blueprint)
    assert result["passed"] is False
    assert {item["label"] for item in result["weak_options"]} >= {"B", "C"}
    assert result["issue_type"] in {"answer_type_mismatch", "surface_eliminable"}


def test_distractor_review_accepts_metric_peer_set(sample_data: Path) -> None:
    graph = KnowledgeGraph.load(sample_data)
    reviewer = DistractorReviewAgent(graph, LLMRouter(Settings(mock=True)))
    blueprint = {
        "writer_output": {
            "stem": "In binary classification evaluation, which metric measures the proportion of predicted-positive samples that are truly positive?",
            "options": {"A": "precision", "B": "accuracy", "C": "recall", "D": "f1 score"},
            "answer": "A",
        },
        "answer_plan": {"correct_answer": {"concept": "precision", "basis": "metric"}},
        "distractor_plan": {
            "distractors": [
                {"concept": "accuracy", "source_relation": "similar/confusable"},
                {"concept": "recall", "source_relation": "similar/confusable"},
                {"concept": "f1 score", "source_relation": "same_topic"},
            ]
        },
    }
    result = reviewer.review(blueprint)
    assert result["passed"] is True
    assert result["weak_options"] == []



def test_orchestrator_progress_callback(sample_data: Path) -> None:
    graph = KnowledgeGraph.load(sample_data)
    messages: list[str] = []
    result = Orchestrator(
        graph,
        LLMRouter(Settings(mock=True)),
        progress_callback=messages.append,
    ).generate("precision")
    assert result["status"] == "ok"
    assert any("Planner" in message for message in messages)
    assert any("Writer" in message for message in messages)
    assert any("Course-scope distractor review" in message for message in messages)
    assert any("Validator" in message for message in messages)
    assert messages[-1].startswith("Complete: generated")


