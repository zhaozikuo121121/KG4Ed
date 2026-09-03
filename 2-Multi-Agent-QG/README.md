# 2-Multi-Agent-QG

`2-Multi-Agent-QG` is a controllable multiple-choice question generator built around a course knowledge graph and a cooperative agent pipeline. Given a target concept, it resolves aliases, inspects graph relations and course context, plans all feasible question types, generates an item set, reviews distractors, validates the final items, and writes JSON output.

The current generator supports definition, graph-relation, and explicit-formula calculation questions. Image-question generation and the former single-item entry point are not part of the current interface.

## Features

- Load concepts, aliases, profiles, course metadata, prerequisite edges, part-of edges, confusable edges, similarity edges, and explicit formulas.
- Plan definition, concept discrimination, prerequisite dependency, component membership, whole decomposition, multi-hop reasoning, and calculation questions.
- Select Bloom-style cognitive level and target difficulty from graph structure.
- Produce one correct answer and three distractors for multiple-choice items.
- Review distractors in the context of the final stem.
- Check that distractors are grounded in the current course when suitable candidates exist.
- Use deterministic Mock mode or an OpenAI-compatible Qwen/DeepSeek endpoint.
- Retry failed generation/validation attempts and record detailed metadata.
- Randomize answer-option order while keeping answer labels and explanations consistent.
- Render result JSON as LaTeX or PDF.
- Batch-generate DGL and MLR datasets by chapter.

## Question Types

- `definition`: basic concept meaning
- `concept_discrimination`: boundaries between confusable concepts
- `prerequisite_dependency`: prerequisite direction in the graph
- `component_membership`: part-of membership
- `whole_decomposition`: whole/component decomposition
- `multi_hop_reasoning`: reasoning over a short graph chain
- `calculation`: open numerical task based on an explicit formula

If no structural relation is available, the planner falls back to `definition`. Calculation planning only applies when the target concept has an explicit dataset/profile formula.

## Project Layout

```text
src/qg_v2/agents/       Planner, cognitive, answer/distractor, Writer, review, Validator
src/qg_v2/graph/        Knowledge-graph loading and queries
src/qg_v2/llm/          Text LLM client and routing
src/qg_v2/render.py     JSON to LaTeX/PDF rendering
src/qg_v2/cli.py        qg command-line entry point
src/batch_generate.py   DGL/MLR batch generation
inputdata/              MOOC, DGL, and MLR data
tests/                  Automated tests
```

## Installation

Use Python 3.10 or newer:

```powershell
cd G:\KG4Ed\2-Multi-Agent-QG
py -m pip install -e ".[test]"
```

The installation registers the `qg` command from `qg_v2.cli`.

## Generate a Question Set

Mock mode does not require an API key:

```powershell
qg generate "precision" --mock
qg generate "activation function" --data inputdata/MOOC --mock --out result.json
```

For a real model, configure `.env` first:

```powershell
Copy-Item .env.example .env
notepad .env
qg generate "precision" --data inputdata/MLR --out result.json
```

Common options:

```text
--data PATH             Input data directory
--out PATH              Output JSON path; stdout when omitted
--mock                  Use deterministic Mock mode
--config PATH           Optional TOML configuration file
--env PATH              .env file, default .env
--max-attempts N        Maximum validation retries
--quiet                 Suppress progress messages
--instruction-version   legacy-v1 or multisource-v2
```

There is no `--question-type`, `--no-visual`, or `--assets-dir` option.

## Batch Generation

Generate all DGL chapters:

```powershell
python src/batch_generate.py DGL
```

Generate all MLR chapters:

```powershell
python src/batch_generate.py MLR
```

Generate selected chapters:

```powershell
python src/batch_generate.py DGL 2
python src/batch_generate.py DGL 1 3 5
python src/batch_generate.py DGL 1,3,5
python src/batch_generate.py MLR 2-4
```

Force regeneration of existing successful outputs:

```powershell
python src/batch_generate.py MLR 1-3,6 --force
```

DGL supports chapters 1-6; MLR supports chapters 1-14. Results are written to `DGLquiz` or `MLRquiz`. Existing files with `status=ok` are skipped by default. Failed jobs are recorded and processing continues.

Batch generation derives each concept's chapter directly from `DGL_concepts_metadata.csv` or `MLR_concepts_metadata.csv`. Concepts are processed in canonical concept-table order; no separate menu file is required.

## Render LaTeX or PDF

```powershell
qg render result.json --out quiz.tex --title "Machine Learning Quiz"
qg render result.json --out quiz.pdf --title "Machine Learning Quiz" --with-answers
```

PDF output requires XeLaTeX on `PATH`. With `--with-answers`, the answer key and explanations are placed at the end of the document.

## Configuration

Use `.env` for API keys, model names, timeouts, and review switches. An optional TOML file may be supplied with `--config` for model and review overrides.

```env
QG_ENABLE_DISTRACTOR_REVIEW=true
QG_DISTRACTOR_REVIEW_MAX_REVISIONS=2
QG_ENABLE_COURSE_SCOPE_DISTRACTOR_REVIEW=true
QG_ALLOW_LLM_FALLBACK=false
```

Real API failures are reported by default. Use `--mock` for local workflow checks.

## Tests

```powershell
py -m pytest -q
```

The test suite covers graph loading, planning, Mock generation, calculation validation, distractor reviews, batch chapter selection, and LaTeX rendering.
