# KG4Ed

The KG4Ed project is the companion codebase for the paper *KG4Ed: When Knowledge Graphs Meet LLMs for Quiz Generation from ML Educational Curricula*. It covers knowledge graph construction, prerequisite relation prediction, and knowledge-structure-driven educational question generation.


The project is organized as a three-stage research pipeline:

1. **Data processing**: extracts concepts, profiles, formulas, figures, evidence, and educational relations from the Machine Learning Refined (MLR) textbook and Deep Graph-Based Learning (DGL) lecture materials.
2. **ACG-CD**: learns directed prerequisite relations for educational knowledge graphs by combining concept semantics, graph structure, collaborative distillation, and quality-controlled pseudo-label expansion. It supports supervised evaluation, transfer experiments, and target-domain adaptation.
3. **Multi-agent question generation**: uses the resulting course knowledge structures to plan, author, review, and validate educational multiple-choice and calculation questions.

## Research pipeline

```text
Instructional resources
        ↓
Concepts, profiles, formulas, and source evidence
        ↓
Weak semantic graphs and annotated relations
        ↓
ACG-CD prerequisite-relation prediction
        ↓
Educational knowledge graphs
        ↓
Multi-agent question planning, generation, review, and validation
        ↓
Structured educational assessment items
```

## Repository structure

- [`0-Data-Processing`](0-Data-Processing): preparation of MLR and DGL instructional-resource datasets.
- [`1-ACG-CD`](1-ACG-CD): the alternating content–graph collaborative distillation framework and its evaluation workflow.
- [`2-Multi-Agent-QG`](2-Multi-Agent-QG): the knowledge-graph-constrained question-generation and validation system.

Each subproject contains its own README with implementation-specific information. The root README intentionally provides only the research-level overview; operational commands, configuration details, data schemas, and testing instructions remain in the relevant subproject documentation.

## Reproducibility and security

The repository separates source code, configuration, intermediate checkpoints, and generated data. API credentials are supplied through local environment variables and are not part of the project documentation. Before sharing or publishing the repository, verify that local `.env` files, private checkpoints, raw provider responses, and machine-specific caches are excluded.

The datasets and generated artifacts should be interpreted together with the dissertation’s methodology and experimental protocol. The implementation may contain cleaned or extended components that are newer than the exact version described in the dissertation.
