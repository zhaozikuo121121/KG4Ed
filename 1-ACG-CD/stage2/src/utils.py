from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

LABEL_COLUMNS = [
    "source_id",
    "target_id",
    "source_index",
    "target_index",
    "source",
    "target",
    "label",
    "split",
    "negative_type",
    "similarity",
]

PSEUDO_COLUMNS = [
    "source_id",
    "target_id",
    "source_index",
    "target_index",
    "source",
    "target",
    "label",
    "teacher_model",
    "model_score",
    "llm_score",
    "combined_score",
    "label_confidence",
    "iteration",
    "pseudo_type",
]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

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
    return {
        "dataset_name": key,
        "stage0_outputs_dir": f"stage0/outputs{suffix}",
        "stage1_outputs_dir": f"stage1/outputs{suffix}",
        "stage2_outputs_dir": f"stage2/outputs{suffix}",
        "checkpoint_dir": f"stage2/checkpoints{suffix}",
    }


def _apply_dataset_defaults(cfg: Dict[str, Any], project_root: Path, *, force: bool = False) -> None:
    defaults = _dataset_defaults(str(cfg.get("dataset_name", "moocml")))
    for key, value in defaults.items():
        if force or key not in cfg or cfg[key] in (None, ""):
            cfg[key] = value
    profile = cfg.setdefault("profile", {})
    if isinstance(profile, dict) and (force or not profile.get("cache_path")):
        profile["cache_path"] = str(Path(cfg["stage0_outputs_dir"]) / "concept_profiles.jsonl")
    llm = cfg.setdefault("llm", {})
    if isinstance(llm, dict) and (force or not llm.get("cache_path")):
        llm["cache_path"] = str(Path(cfg["stage2_outputs_dir"]) / "llm_judge_cache.jsonl")


def expand_env_vars(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")
    return _ENV_PATTERN.sub(repl, value)


def _coerce_scalar(value: str) -> Any:
    value = expand_env_vars(value.strip())
    if value == "":
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(part.strip()) for part in inner.split(",")]
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none", "~"}:
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _simple_yaml_load(text: str) -> Dict[str, Any]:
    """Small YAML subset parser supporting top-level keys and one nested mapping level."""
    data: Dict[str, Any] = {}
    current_parent: str | None = None
    for raw_line in text.splitlines():
        no_comment = raw_line.split("#", 1)[0].rstrip()
        if not no_comment.strip() or ":" not in no_comment:
            continue
        indent = len(no_comment) - len(no_comment.lstrip(" "))
        line = no_comment.strip()
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            if value == "":
                data[key] = {}
                current_parent = key
            else:
                data[key] = _coerce_scalar(value)
                current_parent = None
        elif current_parent is not None:
            if not isinstance(data.get(current_parent), dict):
                data[current_parent] = {}
            data[current_parent][key] = _coerce_scalar(value)
    return data


def _expand_nested(value: Any) -> Any:
    if isinstance(value, str):
        return expand_env_vars(value)
    if isinstance(value, dict):
        return {k: _expand_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_nested(v) for v in value]
    return value


def _resolve_project_root(cfg: Dict[str, Any], cli_project_root: str | None = None) -> Path:
    raw = cli_project_root if cli_project_root is not None else str(cfg.get("project_root", "."))
    root = Path(expand_env_vars(raw)).expanduser().resolve()
    required = ["stage0", "stage1", "stage2"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Invalid project_root={root}. Missing {missing}. Run from the project root, set PROJECT_ROOT, "
            "or pass --project-root /path/to/KG4Ed."
        )
    return root


def _resolve_under_root(project_root: Path, value: Any) -> str:
    p = Path(str(value))
    if not p.is_absolute():
        p = project_root / p
    return str(p.expanduser().resolve())


def load_config(config_path: str | Path, project_root_override: str | None = None) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(text) or {}
        cfg = _expand_nested(dict(cfg))
    except Exception:
        cfg = _simple_yaml_load(text)
    cfg = dict(cfg)
    cfg["config_path"] = str(path)
    project_root = _resolve_project_root(cfg, project_root_override)
    cfg["project_root"] = str(project_root)
    cfg["stage2_dir"] = str((project_root / "stage2").resolve())

    # Backward compatibility for old config key.
    if "stage2_outputs_dir" not in cfg and "output_dir" in cfg:
        cfg["stage2_outputs_dir"] = cfg["output_dir"]
    _apply_dataset_defaults(cfg, project_root)
    for key in ["stage0_outputs_dir", "stage1_outputs_dir", "stage2_outputs_dir", "checkpoint_dir"]:
        if key in cfg and cfg[key] is not None:
            cfg[key] = _resolve_under_root(project_root, cfg[key])
    cfg["output_dir"] = cfg["stage2_outputs_dir"]

    # Resolve nested cache paths relative to project_root as well.
    for section_name in ["profile", "llm"]:
        section = cfg.get(section_name)
        if isinstance(section, dict) and section.get("cache_path"):
            section["cache_path"] = _resolve_under_root(project_root, section["cache_path"])
    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "dataset_name", None) is not None:
        project_root = Path(cfg["project_root"])
        cfg["dataset_name"] = args.dataset_name
        _apply_dataset_defaults(cfg, project_root, force=True)
        for key in ["stage0_outputs_dir", "stage1_outputs_dir", "stage2_outputs_dir", "checkpoint_dir"]:
            cfg[key] = _resolve_under_root(project_root, cfg[key])
        cfg["output_dir"] = cfg["stage2_outputs_dir"]
        for section_name in ["profile", "llm"]:
            section = cfg.get(section_name)
            if isinstance(section, dict) and section.get("cache_path"):
                section["cache_path"] = _resolve_under_root(project_root, section["cache_path"])
    if getattr(args, "device", None) is not None:
        cfg["device"] = args.device
    if getattr(args, "output_dir", None) is not None:
        cfg["stage2_outputs_dir"] = _resolve_under_root(Path(cfg["project_root"]), args.output_dir)
        cfg["output_dir"] = cfg["stage2_outputs_dir"]
        llm = cfg.get("llm")
        if isinstance(llm, dict):
            llm["cache_path"] = str(Path(cfg["stage2_outputs_dir"]) / "llm_judge_cache.jsonl")
    return cfg

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def pair_set(df) -> set[tuple[int, int]]:
    if df is None or len(df) == 0:
        return set()
    return set(zip(df["source_index"].astype(int), df["target_index"].astype(int)))


def manual_sgd_step(module: torch.nn.Module, learning_rate: float) -> None:
    with torch.no_grad():
        for param in module.parameters():
            if param.grad is not None:
                param -= learning_rate * param.grad
    module.zero_grad(set_to_none=True)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")




