# DGL annotated lecture extraction

`code/extract_dgl.py` extracts graph-learning concepts, formulas, diagrams, and English page summaries from the six annotated DGL lecture PDFs. It never reads the Clean PDFs and never modifies or imports the MLR implementation.

Each command selects one Lecture PDF. Internally, every physical page is processed and committed separately, so an interrupted run resumes at the first unfinished page.

## Setup

Run these commands from the `0-Data-Processing` root directory:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill `QWEN_API_KEY` in the root `.env`. `QWEN_VISION_MODEL` must support OpenAI-compatible image input. The program intentionally fails instead of falling back to text-only extraction because the handwritten annotations are flattened into the pages.

## Commands

```powershell
# Show the six supported Annotated PDFs and their page counts
python code\extract_dgl.py --list-lectures

# Validate one PDF, body crops, and 2x2 overlapping tiles without API calls
python code\extract_dgl.py --lecture 1 --dry-run

# Process every unfinished page in Lecture 1
python code\extract_dgl.py --lecture 1

# Process the next Lecture PDF that still has unfinished pages
python code\extract_dgl.py --next

# Reprocess every page in a completed Lecture; old page checkpoints remain
# active until each replacement page passes validation
python code\extract_dgl.py --lecture 1 --force
```

There is no whole-course command. A failed page stops the current command; completed earlier pages remain committed, and rerunning the same Lecture resumes automatically.

## Vision workflow

Only the body rectangle is sent to Qwen. The top credits, logos, book links, and bottom BASIRA/YouTube/GitHub strip are excluded. Each page uses one body overview plus four high-resolution overlapping tiles. Printed PDF text clipped to the same body or tile is included as supporting evidence, but visual input remains mandatory.

Handwritten-only concepts and formulas use stricter confidence thresholds than printed material. Ambiguous handwriting is saved in the private page checkpoint for review and is excluded from public CSV files.

## Outputs

- `DGL_concepts.csv`: headerless global concept list in `canonical,alias...` format.
- `DGL_concepts_metadata.csv`: lecture, page, evidence, source type, confidence, and body-relative bounding box.
- `DGL_formula.csv`: one row per formula-concept relationship. The first columns are `formula_id,concept,latex`; one formula ID may appear on several rows.
- `DGL_graph/*.png`: high-resolution crops of meaningful diagrams, plots, matrices, architectures, graph examples, and tables.
- `DGL_graph/index.csv`: one row per visual-concept relationship; one PNG and visual ID may appear on several rows.
- `DGL_page_summaries.csv`: English page titles, summaries, and key topics.
- `.dgl_extraction/`: page checkpoints, raw Qwen responses, deduplication decisions, uncertain handwriting, and generated-file manifests.

There is no minimum or maximum concept count. A valid page may contain zero retained concepts.

## Tests

```powershell
python -m unittest discover -s code\tests -t code -v
```

The tests use mocked Qwen responses and do not consume API quota. Before adding a real key, use `--dry-run` on all six lectures to validate the local PDF set.
