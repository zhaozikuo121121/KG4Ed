# 0-Data-Processing

This project builds the MLR and DGL datasets used by `ACG-CD`.

- **MLR**: concepts, formulas, figures, evidence, and prerequisite relations extracted from *Machine Learning Refined*.
- **DGL**: concepts, formulas, figures, page summaries, evidence, and prerequisite relations extracted from six annotated graph deep-learning lectures.

All implementation code and tests are under `code/`. Dataset inputs, outputs, and restart checkpoints remain under `MLR/` and `DGL/`. API configuration is read from the project-root `.env` only; do not create `.env` files inside either dataset directory.

## Layout

```text
0-Data-Processing/
  code/                 Python implementation and tests
  MLR/                  MLR PDF, data outputs, and checkpoints
  DGL/                  DGL lecture PDFs, data outputs, and checkpoints
  .env.example          API configuration template
  requirements.txt      Python dependencies
```

## Setup

Run the following commands from the `0-Data-Processing` directory. The commands use the generic `python` launcher so they work with the install selected by the reviewer.

```powershell
python --version
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in the required values in `.env`:

- `QWEN_API_KEY` is used for MLR/DGL extraction, concept profiles, and prerequisite generation.
- `DEEPSEEK_API_KEY` is used only by the optional evidence auditor and `generate_profiles_v2.py`.

Python 3.10 or newer is recommended. A virtual environment is optional. When one is activated, continue to use `python ...`; do not put a machine-specific path such as `\.venv\Scripts\python.exe` in shared instructions.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

All `--dry-run` commands validate inputs without calling an API or publishing output. Run them before a real generation job.

## Recommended workflow

1. Extract concepts and source assets from MLR and DGL.
2. Generate concept profiles.
3. Generate prerequisite relations within each dataset.
4. Optionally run the evidence auditor or the second-generation profile pipeline.

## MLR extraction

`code/extract_mlr.py` processes one textbook chapter per run. Completed chapters are checkpointed; rerunning a command resumes safely. The script excludes the appendix, references, and index after PDF page 511.

```powershell
python code\extract_mlr.py --list-chapters
python code\extract_mlr.py --chapter 1 --dry-run
python code\extract_mlr.py --chapter 1
python code\extract_mlr.py --next
python code\extract_mlr.py --chapter 1 --force
```

Main outputs include `MLR/MLR_concepts.csv`, `MLR/MLR_concepts_metadata.csv`, `MLR/MLR_formula.csv`, `MLR/MLR_graph/`, and the private checkpoint directory `MLR/.mlr_extraction/`.

## DGL extraction

`code/extract_dgl.py` processes one annotated lecture PDF. Each physical page is committed separately. Handwritten annotations require an image-capable `QWEN_VISION_MODEL`.

```powershell
python code\extract_dgl.py --list-lectures
python code\extract_dgl.py --lecture 1 --dry-run
python code\extract_dgl.py --lecture 1
python code\extract_dgl.py --next
python code\extract_dgl.py --lecture 1 --force
```

Main outputs include `DGL/DGL_concepts.csv`, `DGL/DGL_concepts_metadata.csv`, `DGL/DGL_formula.csv`, `DGL/DGL_graph/`, `DGL/DGL_page_summaries.csv`, and `DGL/.dgl_extraction/`.

## Concept profiles

`code/generate_profiles.py` creates one standalone English glossary profile for each canonical MLR or DGL concept. It uses extraction evidence and associated formulas/figures, and publishes the JSONL file only after every current concept has a valid checkpoint.

```powershell
python code\generate_profiles.py --dataset all --dry-run
python code\generate_profiles.py --dataset MLR
python code\generate_profiles.py --dataset DGL
python code\generate_profiles.py --dataset all

# Targeted or resumable runs
python code\generate_profiles.py --dataset DGL --next
python code\generate_profiles.py --dataset DGL --concept-id 12
python code\generate_profiles.py --dataset DGL --concept-id 12 --force
```

Outputs are `MLR/MLR_profiles.jsonl` and `DGL/DGL_profiles.jsonl`; per-concept checkpoints are under `.profile_generation/`.

## Prerequisite relations

Prerequisite generation is dataset-local. It never creates MLR-to-DGL edges and sends only canonical concept names to Qwen, not profiles, source evidence, formulas, or figures.

### Pairwise batches

`code/generate_prerequisites.py` labels every unordered concept pair with `1` (left is a direct prerequisite), `-1` (right is a direct prerequisite), or `0` (no confirmed direct relation). The published CSV uses `source,target,label` rows.

```powershell
python code\generate_prerequisites.py --dataset MLR --dry-run
python code\generate_prerequisites.py --dataset DGL --dry-run
python code\generate_prerequisites.py --dataset MLR
python code\generate_prerequisites.py --dataset DGL
python code\generate_prerequisites.py --dataset all

# Resume or regenerate batches
python code\generate_prerequisites.py --dataset MLR --next-batch
python code\generate_prerequisites.py --dataset DGL --force
```

With `--dataset all`, MLR is completed before DGL starts. Checkpoints are under `.prerequisite_generation/`; outputs are `MLR/MLR_prerequisite.csv` and `DGL/DGL_prerequisite.csv`.

### Concept-centric alternative

`code/generate_prerequisites_by_concept.py` assigns each pair to its lower-ID anchor and returns only highly confident positive directed edges.

```powershell
python code\generate_prerequisites_by_concept.py --dataset MLR --dry-run
python code\generate_prerequisites_by_concept.py --dataset DGL --dry-run
python code\generate_prerequisites_by_concept.py --dataset MLR
python code\generate_prerequisites_by_concept.py --dataset DGL
python code\generate_prerequisites_by_concept.py --dataset MLR --next-concept
python code\generate_prerequisites_by_concept.py --dataset DGL --force
```

This alternative does not support `--dataset all`, `--workers`, or `--batch-size`; `--next-concept` and `--force` are mutually exclusive. Its checkpoints are under `.prerequisite_generation_by_concept/`.

## Optional tools

These tools use DeepSeek and are not required for the standard extraction/profile/prerequisite workflow:

```powershell
python code\generate_profiles_v2.py --dataset both --dry-run
python code\generate_profiles_v2.py --dataset MLR

python code\audit_and_revise_evidence.py --dataset both --dry-run
python code\audit_and_revise_evidence.py --dataset MLR --apply
python code\audit_and_revise_evidence.py --dataset both --reconcile-v2
```

`--dry-run`, `--apply`, and `--reconcile-v2` cannot be combined for the evidence auditor.

## Tests

Tests use mocked API clients and do not consume API quota:

```powershell
python -m unittest discover -s code\tests -t code -v
```
