# Machine Learning Refined concept extraction

`code/extract_mlr.py` processes one textbook chapter at a time, calls Qwen to extract machine-learning concepts, and searches one page before and after each concept anchor for related formulas and images. Pages after PDF page 511 (appendices, references, and index) are excluded.

## Setup

Run from the `0-Data-Processing` root:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `QWEN_API_KEY` in the root `.env`. The default endpoint is DashScope's OpenAI-compatible API and model `qwen3.8-max`; set `QWEN_BASE_URL` for another compatible service. Keep the real key only in this `.env` file.

## Usage

```powershell
# Show the fixed 14-chapter page mapping
python code\extract_mlr.py --list-chapters

# Validate the PDF, body pages, and chunks without calling the API
python code\extract_mlr.py --chapter 1 --dry-run

# Process chapter 1
python code\extract_mlr.py --chapter 1

# Process the next unfinished chapter
python code\extract_mlr.py --next

# Reprocess chapter 1 and rebuild aggregate files from checkpoints
python code\extract_mlr.py --chapter 1 --force
```

Runtime output reports PDF validation, concept extraction, evidence checks, formula/image analysis, and result writes. Add `--no-progress` when redirecting logs or invoking the program from another process.

There is no whole-book batch command. Each non-dry-run invocation submits at most one chapter; public output is updated only after page, count, and duplicate checks pass.

## Outputs

- `MLR_concepts.csv`: headerless list; first column is the lowercase English canonical concept, followed by aliases.
- `MLR_concepts_metadata.csv`: chapter, PDF page, book page, and source evidence.
- `MLR_formula.csv`: first three columns are `formula_id,concept,latex`, followed by chapter and page sources.
- `MLR_graph/*.png`: high-resolution crops of figures and captions rendered from the source PDF.
- `MLR_graph/index.csv`: many-to-many image/concept relationships and sources.
- `.mlr_extraction/`: chapter checkpoints, raw model responses, deduplication records, and image manifests.

Each concept appears once in the global CSV but may reference several chapters in metadata. Each image is stored once and may reference multiple concepts through `index.csv`.

## Failures and resume

- Rate limits, server failures, and network errors use exponential backoff.
- Invalid JSON triggers one model-repair attempt.
- If the configured model rejects page images, a warning is emitted and text-only analysis is used; set `QWEN_VISION_MODEL` to select a vision model.
- Each chapter must contain 20–100 valid concepts or it is not committed.
- If merging and evidence checks leave fewer than 20 concepts, up to five targeted supplementation rounds are attempted.
- Completed chapters are never overwritten accidentally; use `--force` explicitly.

## Tests

```powershell
python -m unittest discover -s code\tests -t code -v
```

Tests mock Qwen and consume no API quota. For real-API acceptance, process chapter 1 first and manually inspect the CSV, LaTeX, and crops in `MLR_graph`.
