from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_concept_profiles import (  # noqa: E402
    Concept,
    generate_dataset_profiles,
    read_concepts_csv,
    validate_profiles,
)


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][0]["content"]
        concepts_json = prompt.split("Input concepts:\n", 1)[1]
        concepts = json.loads(concepts_json)
        profiles = []
        for item in concepts:
            name = item["name"]
            profiles.append(
                {
                    "id": item["id"],
                    "name": name,
                    "profile": f"{name}: This is a concise educational profile used for testing machine learning concept profile generation. It explains the central mathematical or algorithmic idea, the typical workflow role, and nearby techniques such as optimization, representation learning, and evaluation. The text is deterministic so tests can verify parsing, writing, and resume behavior without calling an external language model API. It also describes practical usage in model development, feature analysis, training decisions, and interpretation while remaining concise, fluent, and suitable for embedding based prerequisite relation prediction experiments.",
                }
            )
        content = json.dumps({"profiles": profiles})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class GenerateConceptProfilesTests(unittest.TestCase):
    def test_read_concepts_csv_preserves_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            concept_file = Path(tmpdir) / "concepts.csv"
            concept_file.write_text("Neural Networks,NN, neural nets\n\nSVM,Support Vector Machine\n", encoding="utf-8")

            concepts = read_concepts_csv(concept_file)

        self.assertEqual(
            concepts,
            [
                Concept(0, "Neural Networks", ["Neural Networks", "NN", "neural nets"]),
                Concept(1, "SVM", ["SVM", "Support Vector Machine"]),
            ],
        )

    def test_generate_dataset_profiles_writes_jsonl_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            concept_file = tmp / "concepts.csv"
            output_file = tmp / "profiles.jsonl"
            concept_file.write_text("A Concept,A alias\nSecond Concept\n", encoding="utf-8")
            client = FakeClient()

            summary = generate_dataset_profiles(concept_file, output_file, client=client, batch_size=1)

            self.assertEqual(summary["concept_count"], 2)
            self.assertEqual(summary["generated"], 2)
            rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["concept_id"], 0)
            self.assertEqual(rows[0]["concept_name"], "A Concept")
            self.assertEqual(rows[0]["aliases"], ["A Concept", "A alias"])
            self.assertTrue(rows[0]["profile"].startswith("A Concept:"))
            self.assertEqual(rows[0]["model"], "qwen3.8-max")
            self.assertTrue(rows[0]["prompt_version"])

            second_summary = generate_dataset_profiles(concept_file, output_file, client=client, batch_size=1)

            self.assertEqual(second_summary["generated"], 0)
            self.assertEqual(second_summary["skipped_existing"], 2)
            self.assertEqual(client.completions.calls, 2)
            self.assertEqual(len(output_file.read_text(encoding="utf-8").splitlines()), 2)

    def test_quiet_mode_suppresses_logs_and_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            concept_file = tmp / "concepts.csv"
            output_file = tmp / "profiles.jsonl"
            concept_file.write_text("A Concept,A alias\n", encoding="utf-8")
            stdout = StringIO()
            stderr = StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                summary = generate_dataset_profiles(
                    concept_file,
                    output_file,
                    client=FakeClient(),
                    batch_size=1,
                    quiet=True,
                )

            self.assertEqual(summary["generated"], 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")

    def test_validate_profiles_requires_concept_name_prefix(self):
        concepts = [Concept(0, "Gradient Descent", ["Gradient Descent"])]
        payload = {"profiles": [{"id": "0", "profile": "Gradient Descent is missing the required colon."}]}

        with self.assertRaisesRegex(ValueError, "must start"):
            validate_profiles(payload, concepts)

    def test_validate_profiles_accepts_case_changed_prefix_and_restores_canonical_name(self):
        concepts = [Concept(0, "Dialog Systems", ["Dialog Systems"])]
        profile = (
            "Dialog systems: This educational profile describes an interactive machine learning concept in natural "
            "language processing, where models interpret user turns, track dialogue state, and generate appropriate "
            "responses. Its core algorithmic ideas include intent classification, slot filling, policy learning, and "
            "sequence generation with neural encoders or large language models. In machine learning workflows, dialog "
            "systems connect language understanding, response ranking, and evaluation for conversational assistants. "
            "Closely related techniques include dialogue state tracking, reinforcement learning, and natural language "
            "generation in practical deployed conversational AI applications."
        )
        payload = {"profiles": [{"id": "0", "profile": profile}]}

        validated = validate_profiles(payload, concepts)

        self.assertTrue(validated[0].startswith("Dialog Systems:"))

    def test_validate_profiles_accepts_concept_names_that_contain_colons(self):
        concepts = [Concept(0, "Information Theory: coding", ["Information Theory: coding"])]
        profile = (
            "Information Theory: coding: This concept studies the mathematical limits of representing, compressing, "
            "and transmitting data through noisy or capacity limited channels. Its core ideas include entropy, mutual "
            "information, source coding, and channel coding, which quantify uncertainty and guide efficient encodings. "
            "In machine learning workflows, information theory supports feature selection, representation learning, "
            "regularization, compression, and analysis of generalization or bottlenecks. Closely related techniques "
            "include entropy estimation, error correcting codes, and the information bottleneck method for learning "
            "compact predictive representations."
        )
        payload = {"profiles": [{"id": "0", "profile": profile}]}

        validated = validate_profiles(payload, concepts)

        self.assertTrue(validated[0].startswith("Information Theory: coding:"))

    def test_validate_profiles_accepts_underscore_space_prefix_variants(self):
        concepts = [Concept(0, "Graph_theory", ["Graph_theory"])]
        profile = (
            "Graph theory: Graph theory studies networks of vertices and edges as mathematical structures for "
            "representing pairwise relationships. Its core algorithmic ideas include paths, connectivity, centrality, "
            "graph traversal, and optimization over discrete structures. In machine learning workflows, graph theory "
            "supports graph neural networks, knowledge graphs, dependency structures, recommendation systems, and "
            "relational data modeling. It helps define how information propagates between connected entities and how "
            "structural patterns become predictive features. Closely related concepts include network analysis, "
            "spectral graph theory, and graph neural networks for representation learning."
        )
        payload = {"profiles": [{"id": "0", "profile": profile}]}

        validated = validate_profiles(payload, concepts)

        self.assertTrue(validated[0].startswith("Graph_theory:"))

    def test_validate_profiles_accepts_slightly_short_profiles(self):
        concepts = [Concept(0, "Hierarchical models", ["Hierarchical models"])]
        profile = (
            "Hierarchical models: These models organize parameters, variables, or representations across multiple "
            "levels so that local patterns share information through broader group structure. Their core idea is to "
            "define nested dependencies, often with conditional distributions or layered latent variables, enabling "
            "partial pooling and structured inference. In machine learning workflows, hierarchical models help model "
            "grouped data, multi task learning, Bayesian estimation, and interpretable abstraction. Closely related "
            "concepts include Bayesian networks, mixed effects models, and latent variable models."
        )
        self.assertEqual(len(profile.split()), 75)
        payload = {"profiles": [{"id": "0", "profile": profile}]}

        validated = validate_profiles(payload, concepts)

        self.assertTrue(validated[0].startswith("Hierarchical models:"))


if __name__ == "__main__":
    unittest.main()
