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
        "concept_file": "MOOCML_concepts.csv",
        "label_file": "MOOCML_prerequisites.csv",
        "profile_file": "MOOCML_profiles.jsonl",
        "caption_files": [
            "Captions_machine-learning_Stanford.json",
            "Captions_machine-learning_Washington.json",
        ],
        "suffix": "_moocml",
    },
    "lecturebank": {
        "data_dir": "data/LectureBank",
        "concept_file": "LectureBank_concepts.csv",
        "label_file": "LectureBank_prerequisites.csv",
        "profile_file": "LectureBank_profiles.jsonl",
        "caption_files": ["LectureBank_captions.csv"],
        "suffix": "_lecturebank",
    },
    "universitycourse": {
        "data_dir": "data/UniversityCourse",
        "concept_file": "UniversityCourse_concepts.csv",
        "label_file": "UniversityCourse_prerequisites.csv",
        "profile_file": "UniversityCourse_profiles.jsonl",
        "caption_files": ["UniversityCourse_captions.csv"],
        "suffix": "_universitycourse",
    },
    "mlr": {
        "data_dir": "data/MLR",
        "concept_file": "MLR_concepts.csv",
        "label_file": "MLR_prerequisite.csv",
        "profile_file": "MLR_profiles.jsonl",
        "caption_files": [],
        "suffix": "_mlr",
    },
    "dgl": {
        "data_dir": "data/DGL",
        "concept_file": "DGL_concepts.csv",
        "label_file": "DGL_prerequisite.csv",
        "profile_file": "DGL_profiles.jsonl",
        "caption_files": [],
        "suffix": "_dgl",
    },
    "mc_lb_uc": {
        "data_dir": "data/MC_LB_UC",
        "concept_file": "MC_LB_UC_concepts.csv",
        "label_file": "MC_LB_UC_prerequisites.csv",
        "profile_file": "MC_LB_UC_profiles.jsonl",
        "caption_files": [],
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
        "concept_file": item["concept_file"],
        "label_file": item["label_file"],
        "profile_file": item["profile_file"],
        "caption_files": list(item["caption_files"]),
        "stage0_outputs_dir": f"stage0/outputs{suffix}",
    }


def _apply_dataset_defaults(cfg: Dict[str, Any], project_root: Path, *, force: bool = False) -> None:
    defaults = _dataset_defaults(str(cfg.get("dataset_name", "moocml")))
    for key, value in defaults.items():
        if force or key not in cfg or cfg[key] in (None, ""):
            cfg[key] = value
    profile = cfg.setdefault("profile", {})
    if isinstance(profile, dict) and (force or not profile.get("cache_path")):
        profile["cache_path"] = str(Path(cfg["stage0_outputs_dir"]) / "concept_profiles.jsonl")


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
    if not (root / "stage0").exists() or not (root / "data").exists():
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
        cfg = _expand_nested(dict(cfg))
    except Exception:
        cfg = _simple_yaml_load(text)

    cfg = dict(cfg)
    cfg["config_path"] = str(path)
    project_root = _resolve_project_root(cfg, project_root_override)
    cfg["project_root"] = str(project_root)
    cfg["stage0_dir"] = str((project_root / "stage0").resolve())

    if "stage0_outputs_dir" not in cfg and "output_dir" in cfg:
        cfg["stage0_outputs_dir"] = cfg["output_dir"]
    _apply_dataset_defaults(cfg, project_root)
    for key in ["data_dir", "stage0_outputs_dir"]:
        if key in cfg and cfg[key] is not None:
            cfg[key] = _resolve_under_root(project_root, cfg[key])
    cfg["output_dir"] = cfg["stage0_outputs_dir"]
    profile = cfg.get("profile")
    if isinstance(profile, dict) and profile.get("cache_path"):
        profile["cache_path"] = _resolve_under_root(project_root, profile["cache_path"])
    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "dataset_name", None) is not None:
        project_root = Path(cfg["project_root"])
        cfg["dataset_name"] = args.dataset_name
        _apply_dataset_defaults(cfg, project_root, force=True)
        for key in ["data_dir", "stage0_outputs_dir"]:
            cfg[key] = _resolve_under_root(project_root, cfg[key])
        cfg["output_dir"] = cfg["stage0_outputs_dir"]
        profile = cfg.get("profile")
        if isinstance(profile, dict) and profile.get("cache_path"):
            profile["cache_path"] = _resolve_under_root(project_root, profile["cache_path"])
    override_map = {
        "epochs": args.epochs,
        "max_pairs_per_epoch": args.max_pairs_per_epoch,
        "stage0_outputs_dir": args.output_dir,
        "device": args.device,
    }
    project_root = Path(cfg["project_root"])
    for key, value in override_map.items():
        if value is not None:
            if key == "stage0_outputs_dir":
                cfg[key] = _resolve_under_root(project_root, value)
                cfg["output_dir"] = cfg[key]
                profile = cfg.get("profile")
                if isinstance(profile, dict):
                    profile["cache_path"] = str(Path(cfg[key]) / "concept_profiles.jsonl")
            else:
                cfg[key] = value
    return cfg
