from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib
from typing import Any


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(slots=True)
class ModelConfig:
    planner: str = "qwen3.8-max"
    cognitive_difficulty: str = "qwen3.8-max"
    answer_distractor: str = "qwen3.8-max"
    writer: str = "qwen3.8-max"
    validator: str = "qwen3.8-max"
    distractor_review: str = "qwen3.8-max"
    course_scope_distractor: str = "qwen3.8-max"
    solver: str = "qwen3.8-max"
    calculation: str = "qwen3.8-max"

    def for_agent(self, agent_name: str) -> str:
        return getattr(self, agent_name)


@dataclass(slots=True)
class Settings:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    models: ModelConfig = field(default_factory=ModelConfig)
    mock: bool = False
    request_timeout: float = 60.0
    allow_llm_fallback: bool = False
    enable_distractor_review: bool = True
    distractor_review_max_revisions: int = 2
    enable_course_scope_distractor_review: bool = True
    instruction_version: str = "multisource-v2"

    @classmethod
    def load(cls, config_path: str | Path | None = None, env_path: str | Path | None = ".env", mock: bool = False) -> "Settings":
        load_dotenv(env_path)
        default_models = ModelConfig()
        shared_model = os.getenv("DEEPSEEK_MODEL")

        def text_model(env_name: str, default: str) -> str:
            return os.getenv(env_name) or shared_model or default

        models = ModelConfig(
            planner=text_model("QG_MODEL_PLANNER", default_models.planner),
            cognitive_difficulty=text_model("QG_MODEL_COGNITIVE_DIFFICULTY", default_models.cognitive_difficulty),
            answer_distractor=text_model("QG_MODEL_ANSWER_DISTRACTOR", default_models.answer_distractor),
            writer=text_model("QG_MODEL_WRITER", default_models.writer),
            validator=text_model("QG_MODEL_VALIDATOR", default_models.validator),
            distractor_review=text_model("QG_MODEL_DISTRACTOR_REVIEW", default_models.distractor_review),
            course_scope_distractor=text_model("QG_MODEL_COURSE_SCOPE_DISTRACTOR", default_models.course_scope_distractor),
            solver=text_model("QG_MODEL_SOLVER", default_models.solver),
            calculation=text_model("QG_MODEL_CALCULATION", default_models.calculation),
        )
        settings = cls(
            api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or None,
            base_url=os.getenv("DEEPSEEK_BASE_URL") or os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL),
            models=models,
            mock=mock,
            request_timeout=parse_float(
                os.getenv("DEEPSEEK_TIMEOUT_SECONDS")
                or os.getenv("QG_REQUEST_TIMEOUT")
                or os.getenv("QWEN_REQUEST_TIMEOUT"),
                default=60.0,
            ),
            allow_llm_fallback=parse_bool(os.getenv("QG_ALLOW_LLM_FALLBACK"), default=False),
            enable_distractor_review=parse_bool(os.getenv("QG_ENABLE_DISTRACTOR_REVIEW"), default=True),
            distractor_review_max_revisions=parse_int(os.getenv("QG_DISTRACTOR_REVIEW_MAX_REVISIONS"), default=2),
            enable_course_scope_distractor_review=parse_bool(
                os.getenv("QG_ENABLE_COURSE_SCOPE_DISTRACTOR_REVIEW"), default=True
            ),
            instruction_version=parse_instruction_version(os.getenv("QG_INSTRUCTION_VERSION")),
        )
        if config_path:
            settings.apply_toml(config_path)
        if not settings.api_key:
            settings.mock = True
        return settings

    def apply_toml(self, config_path: str | Path) -> None:
        path = Path(config_path)
        with path.open("rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
        api = data.get("api", {})
        if "base_url" in api:
            self.base_url = str(api["base_url"])
        if "request_timeout" in api:
            self.request_timeout = float(api["request_timeout"])
        review_data = data.get("distractor_review", {})
        if "enabled" in review_data:
            self.enable_distractor_review = bool(review_data["enabled"])
        if "max_revisions" in review_data:
            self.distractor_review_max_revisions = int(review_data["max_revisions"])
        if "course_scope_enabled" in review_data:
            self.enable_course_scope_distractor_review = bool(review_data["course_scope_enabled"])
        generation_data = data.get("generation", {})
        if "instruction_version" in generation_data:
            self.instruction_version = parse_instruction_version(str(generation_data["instruction_version"]))
        model_data = data.get("models", {})
        for key, value in model_data.items():
            if hasattr(self.models, key):
                setattr(self.models, key, str(value))


def load_dotenv(env_path: str | Path | None) -> None:
    if not env_path:
        return
    path = Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_inline_comment(value).strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def strip_inline_comment(value: str) -> str:
    """Strip unquoted inline comments in .env values.

    Example: qwen3.8-max   # cheap model -> qwen3.8-max
    """
    in_single = False
    in_double = False
    for idx, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or value[idx - 1].isspace():
                return value[:idx].rstrip()
    return value


def parse_float(value: str | None, *, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def parse_int(value: str | None, *, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def parse_instruction_version(value: str | None) -> str:
    selected = (value or "multisource-v2").strip().lower()
    if selected not in {"legacy-v1", "multisource-v2"}:
        return "multisource-v2"
    return selected
