from __future__ import annotations

from typing import Any
import json

JsonDict = dict[str, Any]


def ensure_json_dict(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected dict, got {type(value).__name__}")


def dumps_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
