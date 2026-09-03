from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.concept_profile import load_stage0_profiles
from src.content_model import load_content_model
from src.data_loader import load_stage2_data
from src.fusion import predict_and_score
from src.train_graph import GraphTrainer
from src.utils import ensure_dir, expand_env_vars, resolve_device, set_seed, write_json

DATASET_REGISTRY = {
    "moocml": {"suffix": "_moocml"},
    "lecturebank": {"suffix": "_lecturebank"},
    "universitycourse": {"suffix": "_universitycourse"},
    "mlr": {"suffix": "_mlr"},
    "dgl": {"suffix": "_dgl"},
    "mc_lb_uc": {"suffix": "_mc_lb_uc"},
}


def _dataset_defaults(dataset_name: str) -> Dict[str, Any]:
    key = dataset_name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset_name={dataset_name!r}. Expected one of {sorted(DATASET_REGISTRY)}")
    suffix = DATASET_REGISTRY[key]["suffix"]
    checkpoint_dir = f"stage2/checkpoints{suffix}"
    stage2_outputs_dir = f"stage2/outputs{suffix}"
    return {
        "dataset_name": key,
        "stage0_outputs_dir": f"stage0/outputs{suffix}",
        "stage1_outputs_dir": f"stage1/outputs{suffix}",
        "stage2_outputs_dir": stage2_outputs_dir,
        "stage3_outputs_dir": f"stage3/outputs{suffix}",
        "checkpoint_dir": checkpoint_dir,
        "content_checkpoint": f"{checkpoint_dir}/content_final.pt",
        "graph_checkpoint": f"{checkpoint_dir}/graph_final.pt",
        "fusion_params_path": f"{stage2_outputs_dir}/fusion_params.json",
    }


def _apply_dataset_defaults(cfg: Dict[str, Any], project_root: Path, *, force: bool = False) -> None:
    fill_content_checkpoint = force or not cfg.get("content_checkpoint")
    fill_graph_checkpoint = force or not cfg.get("graph_checkpoint")
    fill_fusion_params_path = force or not cfg.get("fusion_params_path")
    defaults = _dataset_defaults(str(cfg.get("dataset_name", "moocml")))
    for key, value in defaults.items():
        if force or key not in cfg or cfg[key] in (None, ""):
            cfg[key] = value
    if fill_content_checkpoint:
        cfg["content_checkpoint"] = str(Path(cfg["checkpoint_dir"]) / "content_final.pt")
    if fill_graph_checkpoint:
        cfg["graph_checkpoint"] = str(Path(cfg["checkpoint_dir"]) / "graph_final.pt")
    if fill_fusion_params_path:
        cfg["fusion_params_path"] = str(Path(cfg["stage2_outputs_dir"]) / "fusion_params.json")
    profile = cfg.setdefault("profile", {})
    if isinstance(profile, dict) and (force or not profile.get("cache_path")):
        profile["cache_path"] = str(Path(cfg["stage0_outputs_dir"]) / "concept_profiles.jsonl")


def _expand_nested(value: Any) -> Any:
    if isinstance(value, str):
        return expand_env_vars(value)
    if isinstance(value, dict):
        return {k: _expand_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_nested(v) for v in value]
    return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        return _expand_nested(dict(yaml.safe_load(text) or {}))
    except Exception:
        from src.utils import _simple_yaml_load

        return _simple_yaml_load(text)


def _resolve_project_root(cfg: Dict[str, Any], override: str | None) -> Path:
    raw = override if override is not None else str(cfg.get("project_root", "."))
    root = Path(expand_env_vars(raw)).expanduser().resolve()
    missing = [name for name in ["stage0", "stage1", "stage2", "stage3"] if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Invalid project_root={root}. Missing {missing}.")
    return root


def _resolve_under_root(project_root: Path, value: Any) -> str:
    p = Path(str(value))
    if not p.is_absolute():
        p = project_root / p
    return str(p.expanduser().resolve())


def load_stage3_config(config_path: str | Path, project_root_override: str | None = None) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    cfg = _load_yaml(path)
    cfg["config_path"] = str(path)
    project_root = _resolve_project_root(cfg, project_root_override)
    cfg["project_root"] = str(project_root)
    _apply_dataset_defaults(cfg, project_root)
    for key in [
        "stage0_outputs_dir",
        "stage1_outputs_dir",
        "stage2_outputs_dir",
        "stage3_outputs_dir",
        "checkpoint_dir",
        "content_checkpoint",
        "graph_checkpoint",
        "fusion_params_path",
    ]:
        if key in cfg and cfg[key] is not None:
            cfg[key] = _resolve_under_root(project_root, cfg[key])
    profile = cfg.get("profile")
    if isinstance(profile, dict) and profile.get("cache_path"):
        profile["cache_path"] = _resolve_under_root(project_root, profile["cache_path"])
    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "dataset_name", None) is not None:
        cfg["dataset_name"] = args.dataset_name
        _apply_dataset_defaults(cfg, Path(cfg["project_root"]), force=True)
        for key in [
            "stage0_outputs_dir",
            "stage1_outputs_dir",
            "stage2_outputs_dir",
            "stage3_outputs_dir",
            "checkpoint_dir",
            "content_checkpoint",
            "graph_checkpoint",
            "fusion_params_path",
        ]:
            cfg[key] = _resolve_under_root(Path(cfg["project_root"]), cfg[key])
        profile = cfg.get("profile")
        if isinstance(profile, dict) and profile.get("cache_path"):
            profile["cache_path"] = _resolve_under_root(Path(cfg["project_root"]), profile["cache_path"])
    if getattr(args, "device", None) is not None:
        cfg["device"] = args.device
    if getattr(args, "output_dir", None) is not None:
        cfg["stage3_outputs_dir"] = _resolve_under_root(Path(cfg["project_root"]), args.output_dir)
    return cfg


def _load_fusion_params(path: str | Path) -> Dict[str, float]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing fusion params: {p}. Run Stage2 training first.")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if "selected_alpha" not in payload or "selected_threshold" not in payload:
        raise ValueError(f"Invalid fusion params at {p}: expected selected_alpha and selected_threshold.")
    return {
        "selected_alpha": float(payload["selected_alpha"]),
        "selected_threshold": float(payload["selected_threshold"]),
    }


def run_stage3(cfg: Dict[str, Any], device) -> Dict:
    set_seed(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(cfg["stage3_outputs_dir"])
    data = load_stage2_data(cfg["stage0_outputs_dir"], cfg["stage1_outputs_dir"])
    profiles = load_stage0_profiles(data.concepts, cfg)
    fusion = _load_fusion_params(cfg["fusion_params_path"])

    content_model = load_content_model(cfg["content_checkpoint"], cfg, data, device)
    graph_trainer = GraphTrainer.load_checkpoint(cfg["graph_checkpoint"], cfg, data, device)

    dtest = data.test_labels.copy().reset_index(drop=True)
    content_scores = content_model.predict_scores(dtest, profiles, batch_size=int(cfg.get("content_batch_size", 8)))
    graph_scores = graph_trainer.predict_scores(dtest)
    pred, test_metrics = predict_and_score(
        dtest,
        content_scores,
        graph_scores,
        alpha=fusion["selected_alpha"],
        threshold=fusion["selected_threshold"],
    )

    pred.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")
    performance = {
        "precision": float(test_metrics.get("precision", 0.0)),
        "recall": float(test_metrics.get("recall", 0.0)),
        "f1": float(test_metrics.get("f1", 0.0)),
    }
    metrics_payload = {
        "precision": performance["precision"],
        "recall": performance["recall"],
        "f1": performance["f1"],
    }
    write_json(output_dir / "test_metrics.json", metrics_payload)
    summary = {
        "stage": "stage3",
        "dtest_rows": int(len(dtest)),
        "outputs": {
            "test_predictions": str(output_dir / "test_predictions.csv"),
            "test_metrics": str(output_dir / "test_metrics.json"),
        },
        "test_metrics": performance,
    }
    write_json(output_dir / "stage3_summary.json", summary)
    return summary
