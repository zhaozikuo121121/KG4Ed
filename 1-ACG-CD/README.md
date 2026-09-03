# ACG-CD

## Project Overview

ACG-CD predicts concept prerequisite relations for educational knowledge graphs. It combines concept text, static Concept Profiles, weak semantic graphs, and manually annotated relations. The Content Model, Graph Model, and Label Cleaning components learn directed prerequisite relations.

| Stage  | Main function                                                                              | Outputs                                                        |
| ------ | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Stage0 | Concept encoding, Concept Profile processing, and initial weak semantic graph construction | concept table, embeddings, similarity matrix, adjacency matrix |
| Stage1 | Supervised splitting, positive/negative sample construction, and leakage checks            | train/validation/test labels                                   |
| Stage2 | Train Content Model, Rel-GraphSAGE, AKD, and fusion parameters                             | checkpoints, pseudo-labels, fusion parameters                  |
| Stage3 | Independent or cross-dataset transfer testing                                              | predictions, Precision, Recall, F1-score                       |

Each stage passes its outputs to the next through explicit directories. Stage3 only uses trained models and the Stage1 test set; it does not retrain models or search fusion parameters.

Supported datasets: `moocml`, `lecturebank`, `universitycourse`, `mlr`, `dgl`, and `mc_lb_uc`. `mc_lb_uc` is a disjoint union of MOOCML, LectureBank, and UniversityCourse. Its concept IDs are globally reindexed and no cross-dataset prerequisite edges are added.

Concept files are headerless CSV files:
```text
canonical_concept,alias_1,alias_2
```
Prerequisite label files are headerless CSV files:
```text
source_concept,target_concept,label
```
`label=1` means `source_concept -> target_concept` is a prerequisite relation; `label=0` is a non-prerequisite candidate.

## Environment and Configuration

Python 3.10 or later is required. See [requirements.txt](requirements.txt) for the complete dependency list.

```powershell
# Create and activate a virtual environment
py -m venv .venv
.venv\Scripts\Activate.ps1
# Upgrade pip and install dependencies
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

On Linux/macOS, replace `py` with `python3` and activate with `source .venv/bin/activate`. For GPU use, install the PyTorch version matching CUDA. When PyTorch is preinstalled, use [requirements_autodl.txt](requirements_autodl.txt) for the remaining dependencies.

Run commands from the `ACG-CD` project directory (the directory containing this README), or set the project root explicitly. Do not assume a particular drive or parent directory:

```powershell
$env:PROJECT_ROOT = (Get-Location).Path
```
```bash
export PROJECT_ROOT="$PWD"
```

Relative configuration paths are resolved from `project_root`. Stage0, Stage1, and Stage3 configurations are stored in their respective `config` directories. The default configurations are `stage0/config/config.yaml`, `stage1/config/config.yaml`, `stage2/config.yaml`, and `stage3/config/config.yaml`.

Stage2 enables the Qwen LLM Judge by default. Set `DASHSCOPE_API_KEY`; do not store API keys in YAML or source code.

```powershell
$env:DASHSCOPE_API_KEY = "your DashScope API key"
```
```yaml
use_llm_judge: true
llm:
  provider: qwen
  model: qwen3.8-max
  api_key_env: DASHSCOPE_API_KEY
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

Without an API key, set `use_llm_judge: false`; `profile.use_llm_profiles` can also be set to `false` to use local fallback profiles. Device values include `auto`, `cpu`, `cuda`, and `cuda:0`.

## Stage Commands

Commands below use Windows `py`; replace it with `python3` on Linux/macOS.

### Stage0

Stage0 creates concept tables, static Concept Profiles, BERT embeddings, similarity matrices, and initial weak semantic graphs.

```powershell
py stage0\run_stage0.py --config stage0\config\config_moocml.yaml
py stage0\run_stage0.py --config stage0\config\config_lecturebank.yaml
py stage0\run_stage0.py --config stage0\config\config_universitycourse.yaml
py stage0\run_stage0.py --config stage0\config\config_mlr.yaml
py stage0\run_stage0.py --config stage0\config\config_dgl.yaml
py stage0\run_stage0.py --config stage0\config\config_mc_lb_uc.yaml
```

Dataset shortcuts: `--dataset-name moocml`, `lecturebank`, `universitycourse`, `mlr`, `dgl`, or `mc_lb_uc`.

```powershell
# Common overrides
py stage0\run_stage0.py --config stage0\config\config_moocml.yaml --device cuda
py stage0\run_stage0.py --config stage0\config\config_moocml.yaml --project-root (Get-Location).Path
py stage0\run_stage0.py --config stage0\config\config_moocml.yaml --output-dir stage0\outputs_custom
py stage0\export_graphml.py
```

### Stage1

Stage1 creates training, validation, and test splits plus negative samples. `--train-ratio` must be `0.15`, `0.30`, or `0.60`; the legacy `m` argument is unsupported.

```powershell
py stage1\run_stage1.py --config stage1\config\config_moocml.yaml --train-ratio 0.15
py stage1\run_stage1.py --config stage1\config\config_moocml.yaml --train-ratio 0.30
py stage1\run_stage1.py --config stage1\config\config_moocml.yaml --train-ratio 0.60
py stage1\run_stage1.py --config stage1\config\config_lecturebank.yaml --train-ratio 0.60
py stage1\run_stage1.py --config stage1\config\config_universitycourse.yaml --train-ratio 0.60
py stage1\run_stage1.py --config stage1\config\config_mlr.yaml --train-ratio 0.60
py stage1\run_stage1.py --config stage1\config\config_dgl.yaml --train-ratio 0.60
py stage1\run_stage1.py --config stage1\config\config_mc_lb_uc.yaml --train-ratio 0.60
```

For the MOOCML label-file-only setup:
```powershell
py stage1\run_stage1.py --config stage1\config\config_moocml_label_file_only.yaml --train-ratio 0.60
```

Outputs are under `stage1/outputs_<dataset>`, including `train_labels_initial.csv`, `val_labels.csv`, `test_labels.csv`, `heldout_pairs.csv`, and negative-sample files.

### Stage2

Stage2 trains the Content Model, Rel-GraphSAGE, AKD, and fusion parameters.

Dataset-specific parameter tuning is required when training on different datasets.

The provided parameters are only default parameters.

```powershell
py stage2\run_stage2.py --config stage2\config.yaml --dataset-name moocml --device cuda
py stage2\run_stage2.py --config stage2\config.yaml --dataset-name lecturebank --device cuda
py stage2\run_stage2.py --config stage2\config.yaml --dataset-name universitycourse --device cuda
py stage2\run_stage2.py --config stage2\config.yaml --dataset-name mlr --device cuda
py stage2\run_stage2.py --config stage2\config.yaml --dataset-name dgl --device cuda
py stage2\run_stage2.py --config stage2\config.yaml --dataset-name mc_lb_uc --device cuda
```

Main outputs: `stage2/checkpoints_<dataset>/content_final.pt`, `graph_final.pt`, and `stage2/outputs_<dataset>/fusion_params.json`.

### Stage3

Run independent tests with the matching dataset configuration:

```powershell
py stage3\run_stage3.py --config stage3\config\config_moocml.yaml --device cuda
py stage3\run_stage3.py --config stage3\config\config_lecturebank.yaml --device cuda
py stage3\run_stage3.py --config stage3\config\config_universitycourse.yaml --device cuda
py stage3\run_stage3.py --config stage3\config\config_mlr.yaml --device cuda
py stage3\run_stage3.py --config stage3\config\config_dgl.yaml --device cuda
py stage3\run_stage3.py --config stage3\config\config_mc_lb_uc.yaml --device cuda
```

Single-source transfer tests:
```powershell
py stage3\run_stage3.py --config stage3\config\config_moocml_to_lecturebank.yaml --device cuda
py stage3\run_stage3.py --config stage3\config\config_moocml_to_universitycourse.yaml --device cuda
py stage3\run_stage3.py --config stage3\config\config_lecturebank_to_moocml.yaml --device cuda
py stage3\run_stage3.py --config stage3\config\config_lecturebank_to_universitycourse.yaml --device cuda
```

For MC+LB+UC joint training followed by MLR/DGL transfer testing, run `tools\prepare_mc_lb_uc.py`, Stage0 and Stage1 (`--train-ratio 0.60`) for the combined dataset, then run Stage2 with `stage2/config.yaml --dataset-name mc_lb_uc`. Generate Stage0 and Stage1 outputs for MLR and DGL, then run:

```powershell
py stage3\run_stage3.py --config stage3\config\config_mc_lb_uc_to_mlr.yaml --device cuda
py stage3\run_stage3.py --config stage3\config\config_mc_lb_uc_to_dgl.yaml --device cuda
```

`test_metrics.json` and console output contain `precision`, `recall`, and `f1`; per-sample results are saved in `test_predictions.csv`.

## Run the Full Workflow

[run_all.py](run_all.py) prepares the combined data, runs Stage0 for six datasets, creates three Stage1 splits per dataset, runs six Stage2 trainings, six independent Stage3 tests, and six transfer-test configurations. The last Stage1 split is `0.60`, which is used by later Stage2/Stage3 commands.

```powershell
# Full GPU workflow
py run_all.py --device cuda
# Full CPU workflow
py run_all.py --device cpu
# Print commands without running them
py run_all.py --dry-run
# Continue after an individual command fails
py run_all.py --device cuda --continue-on-error
```

The full workflow requires substantial time, disk space, and working model-download/inference access. Set `DASHSCOPE_API_KEY` before production use, or disable `use_llm_judge`.

## Testing and Troubleshooting

```powershell
py -m unittest discover -s stage0\tests -v
py -m unittest discover -s stage1\tests -v
py -m unittest discover -s stage2\tests -v
py -m unittest discover -s stage3\tests -v
```

- `Missing Stage0 ...`: Run Stage0 for the relevant dataset first.
- `Missing Stage1 ...`: Run Stage1 for the relevant dataset and confirm that `--train-ratio` is valid.
- `Missing fusion params`: Run Stage2 for the relevant dataset first.
- `Not enough label-file negatives`: Reduce `label_file_negative_ratio_to_train` or check the available negative labels.
- Missing `DASHSCOPE_API_KEY`: Set the key or set `use_llm_judge` to `false`.
- CUDA/out-of-memory error: use `--device cpu`, or reduce batch size, maximum sequence length, and training epochs.
