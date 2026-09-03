from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

DATASET_REGISTRY = {
    "moocml": {
        "data_dir": "data/MOOCML",
        "label_file": "MOOCML_prerequisites.csv",
        "suffix": "_moocml",
    },
    "lecturebank": {
        "data_dir": "data/LectureBank",
        "label_file": "LectureBank_prerequisites.csv",
        "suffix": "_lecturebank",
    },
    "universitycourse": {
        "data_dir": "data/UniversityCourse",
        "label_file": "UniversityCourse_prerequisites.csv",
        "suffix": "_universitycourse",
    },
    "mlr": {
        "data_dir": "data/MLR",
        "label_file": "MLR_prerequisite.csv",
        "suffix": "_mlr",
    },
    "dgl": {
        "data_dir": "data/DGL",
        "label_file": "DGL_prerequisite.csv",
        "suffix": "_dgl",
    },
    "mc_lb_uc": {
        "data_dir": "data/MC_LB_UC",
        "label_file": "MC_LB_UC_prerequisites.csv",
        "suffix": "_mc_lb_uc",
    },
}


def _dataset_defaults(dataset_name: str) -> Dict[str, Any]:
    key = dataset_name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset_name={dataset_name!r}. Expected one of {sorted(DATASET_REGISTRY)}")
    item = DATASET_REGISTRY[key]
    suffix = item["suffix"]
    return {
        "dataset_name": key,
        "data_dir": item["data_dir"],
        "label_file": item["label_file"],
        "stage0_outputs_dir": f"stage0/outputs{suffix}",
        "stage1_outputs_dir": f"stage1/outputs{suffix}",
    }


def _apply_dataset_defaults(cfg: Dict[str, Any], project_root: Path, *, force: bool = False) -> None:
    defaults = _dataset_defaults(str(cfg.get("dataset_name", "moocml")))
    for key, value in defaults.items():
        if force or key not in cfg or cfg[key] in (None, ""):
            cfg[key] = value


def expand_env_vars(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")
    return _ENV_PATTERN.sub(repl, value)


def _coerce_scalar(value: str) -> Any:
    value = expand_env_vars(value.strip())
    if value == "":
        return ""
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
    data: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _coerce_scalar(value)
    return data


def _resolve_project_root(cfg: Dict[str, Any], cli_project_root: str | None = None) -> Path:
    raw = cli_project_root if cli_project_root is not None else str(cfg.get("project_root", "."))
    root = Path(expand_env_vars(raw)).expanduser().resolve()
    if not (root / "stage0").exists() or not (root / "stage1").exists() or not (root / "data").exists():
        raise FileNotFoundError(
            f"Invalid project_root={root}. Run from the project root, set PROJECT_ROOT, "
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
        cfg = {k: expand_env_vars(v) if isinstance(v, str) else v for k, v in dict(cfg).items()}
    except Exception:
        cfg = _simple_yaml_load(text)

    cfg = dict(cfg)
    cfg["config_path"] = str(path)
    project_root = _resolve_project_root(cfg, project_root_override)
    cfg["project_root"] = str(project_root)
    cfg["stage1_dir"] = str((project_root / "stage1").resolve())

    if "stage1_outputs_dir" not in cfg and "output_dir" in cfg:
        cfg["stage1_outputs_dir"] = cfg["output_dir"]
    _apply_dataset_defaults(cfg, project_root)
    for key in ["data_dir", "stage0_outputs_dir", "stage1_outputs_dir"]:
        if key in cfg and cfg[key] is not None:
            cfg[key] = _resolve_under_root(project_root, cfg[key])
    cfg["output_dir"] = cfg["stage1_outputs_dir"]
    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "dataset_name", None) is not None:
        project_root = Path(cfg["project_root"])
        cfg["dataset_name"] = args.dataset_name
        _apply_dataset_defaults(cfg, project_root, force=True)
        for key in ["data_dir", "stage0_outputs_dir", "stage1_outputs_dir"]:
            cfg[key] = _resolve_under_root(project_root, cfg[key])
        cfg["output_dir"] = cfg["stage1_outputs_dir"]
    overrides = {
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "stage1_outputs_dir": args.output_dir,
        "reverse_negative_count": args.reverse_negative_count,
        "hard_negative_count": args.hard_negative_count,
    }
    project_root = Path(cfg["project_root"])
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "stage1_outputs_dir":
            cfg[key] = _resolve_under_root(project_root, value)
            cfg["output_dir"] = cfg[key]
        else:
            cfg[key] = value
    return cfg
