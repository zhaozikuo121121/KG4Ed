"""Render generated quiz JSON as a polished XeLaTeX worksheet.

The renderer deliberately keeps presentation separate from the LLM pipeline:
the JSON remains the machine-readable source of truth while this module turns
it into a printable artifact.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class RenderError(RuntimeError):
    """Raised when a quiz cannot be rendered or compiled."""


def latex_escape(value: Any) -> str:
    """Escape text for LaTeX while preserving Unicode (handled by XeLaTeX)."""
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


_MATH_SYMBOLS = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon",
    "ϵ": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "ϑ": r"\vartheta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu",
    "ξ": r"\xi", "π": r"\pi", "ϖ": r"\varpi", "ρ": r"\rho", "ϱ": r"\varrho",
    "σ": r"\sigma", "ς": r"\varsigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "ϕ": r"\varphi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi",
    "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    "∇": r"\nabla", "∂": r"\partial", "∑": r"\sum", "∏": r"\prod", "∫": r"\int",
    "∮": r"\oint", "√": r"\sqrt", "∞": r"\infty", "±": r"\pm", "×": r"\times", "÷": r"\div",
    "·": r"\cdot", "⋆": r"\star", "∗": r"\ast", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx",
    "≡": r"\equiv", "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊆": r"\subseteq",
    "∪": r"\cup", "∩": r"\cap", "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇔": r"\Leftrightarrow", "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}",
}
_MATH_START = "".join(re.escape(char) for char in _MATH_SYMBOLS)
_SCRIPT_VALUE = rf"(?:\{{[^{{}}\n]+\}}|[{_MATH_START}A-Za-z0-9.*]+)"
_BASE_IDENTIFIER = rf"[{_MATH_START}A-Za-z][{_MATH_START}A-Za-z0-9]*"
_IDENTIFIER = rf"{_BASE_IDENTIFIER}(?:[_^]{_SCRIPT_VALUE})*"
_BRACED_SCRIPTED_IDENTIFIER = rf"{_BASE_IDENTIFIER}(?:[_^]\{{[^{{}}\n]+\}})+"
_ATOM = rf"{_IDENTIFIER}(?:\([^()\s]*\))?|[0-9]+(?:\.[0-9]+)?"
_GROUP = rf"\([\s{_MATH_START}A-Za-z0-9_.+*/^\-]+\)"
_OPERAND = rf"(?:{_GROUP}|{_ATOM})"
_EXPRESSION = rf"{_OPERAND}(?:\s*[+*/^\-]\s*{_OPERAND})*"
_SYMBOLIC_ATOM = (
    rf"[{_MATH_START}][{_MATH_START}A-Za-z0-9]*"
    rf"(?:[_^]{_SCRIPT_VALUE})*(?:\([^()\s]*\))?"
)
_SYMBOLIC_EXPRESSION = rf"{_SYMBOLIC_ATOM}(?:\s*[+*/^\-]\s*{_SYMBOLIC_ATOM})*"
_MATH_RUN = re.compile(
    rf"(?<![A-Za-z0-9])(?:{_IDENTIFIER}\s*(?::=|=|[≥≤≠≈])\s*{_EXPRESSION}|"
    rf"[∫∮∑∏][{_MATH_START}A-Za-z0-9_()/*^+\-]*(?:\s+[{_MATH_START}A-Za-z][{_MATH_START}A-Za-z0-9_()/*^+\-]*){{1,3}}|"
    rf"{_BRACED_SCRIPTED_IDENTIFIER}|"
    rf"{_SYMBOLIC_EXPRESSION})"
)
_EXPLICIT_MATH = re.compile(r"(\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$[^$\n]+\$)", re.DOTALL)
_RAW_LATEX_COMMAND = re.compile(
    r"\\(?:(?:alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|varsigma|tau|upsilon|phi|varphi|chi|psi|omega|"
    r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|nabla|partial|sum|prod|int|oint|sqrt|frac|hat|bar|tilde|vec|dot|ddot|overline|underline|mathbf|mathrm|mathbb|mathcal|mathsf|mathtt|left|right|cdot|times|div|pm|leq|geq|neq|approx|equiv|in|notin|subset|subseteq|cup|cap|to|leftarrow|leftrightarrow|Rightarrow|Leftrightarrow)(?![A-Za-z])"
    r"|\|)"
)
_RAW_MATH_BRIDGE = re.compile(
    rf"(?:\s+|[{_MATH_START}A-Za-z](?:[_^]{_SCRIPT_VALUE})*|[0-9]+(?:\.[0-9]+)?|[+*/=()\[\]{{}}^_\-])*"
)


def latex_text(value: Any) -> str:
    """Escape prose and put common Unicode formula runs into math mode.

    XeLaTeX's CJK font is not guaranteed to contain Greek/math glyphs. Mapping
    these symbols to standard LaTeX commands also makes the output portable
    across TeX Live and MiKTeX installations.
    """
    raw = "" if value is None else _repair_malformed_latex_escapes(str(value))
    def convert_formula(source: str) -> str:
        parts: list[str] = []
        for position, char in enumerate(source.strip()):
            replacement = _MATH_SYMBOLS.get(char, char)
            parts.append(replacement)
            if char in _MATH_SYMBOLS and position + 1 < len(source) and source[position + 1].isalpha():
                parts.append(" ")
        return "".join(parts).strip()

    pieces: list[str] = []
    cursor = 0
    # Explicit math supplied by an LLM/user is preserved, while prose outside
    # it is escaped normally. Implicit Unicode/ASCII formulas are then found
    # only in the prose segments.
    for explicit in _EXPLICIT_MATH.finditer(raw):
        if explicit.start() > cursor:
            pieces.append(_implicit_math(raw[cursor:explicit.start()], convert_formula))
        pieces.append(_convert_explicit_math(explicit.group(), convert_formula))
        cursor = explicit.end()
    pieces.append(_implicit_math(raw[cursor:], convert_formula))
    return "".join(pieces)


def _repair_malformed_latex_escapes(text: str) -> str:
    """Recover common LaTeX commands decoded as JSON control characters.

    A few source strings contain ``\b``, ``\t``, ``\f`` or ``\r`` commands in
    JSON without escaping the backslash, so ``json.loads`` turns the first
    character into backspace, tab, form-feed, or carriage-return. Repair only
    recognizable command tails and leave ordinary whitespace untouched.
    """
    text = re.sub(r"\x08(?=(?:oldsymbol|ar|eta))", r"\\b", text)
    text = re.sub(r"\t(?=(?:extbf|ext|imes|heta|ilde|anh))", r"\\t", text)
    text = re.sub(r"\x0c(?=rac)", r"\\f", text)
    text = re.sub(r"\r(?=ho)", r"\\r", text)
    return text


def _convert_explicit_math(source: str, convert_formula: Any) -> str:
    """Convert Unicode math symbols inside an already delimited math block."""
    def convert_body(body: str) -> str:
        converted = convert_formula(body)
        # Some JSON records over-escaped LaTeX commands (``\\begin``). Keep
        # matrix row breaks (``\\`` followed by whitespace) untouched while
        # collapsing doubled command prefixes.
        return re.sub(r"\\\\(?=[A-Za-z()\[\]])", lambda _match: "\\", converted)

    if source.startswith("\\(") and source.endswith("\\)"):
        return r"\(" + convert_body(source[2:-2]) + r"\)"
    if source.startswith("\\[") and source.endswith("\\]"):
        return r"\[" + convert_body(source[2:-2]) + r"\]"
    if source.startswith("$$") and source.endswith("$$"):
        body = convert_body(source[2:-2])
        if (body.startswith(r"\(") and body.endswith(r"\)")) or (
            body.startswith(r"\[") and body.endswith(r"\]")
        ):
            return body
        return "$$" + body + "$$"
    if source.startswith("$") and source.endswith("$"):
        body = convert_body(source[1:-1])
        if (body.startswith(r"\(") and body.endswith(r"\)")) or (
            body.startswith(r"\[") and body.endswith(r"\]")
        ):
            return body
        return "$" + body + "$"
    return source


def _implicit_math(text: str, convert_formula: Any) -> str:
    pieces: list[str] = []
    cursor = 0
    matches: list[tuple[int, int, str, str]] = []
    for match in _MATH_RUN.finditer(text):
        matches.append((match.start(), match.end(), "unicode", match.group()))
    for start, end, raw in _raw_latex_spans(text):
        matches.append((start, end, "latex", raw))
    matches.sort(key=lambda value: (value[0], value[1]))
    for start, end, kind, value in matches:
        if start < cursor:
            continue
        if start > cursor:
            pieces.append(latex_escape(text[cursor:start]))
        formula = value if kind == "latex" else convert_formula(value)
        pieces.append(r"\(" + formula + r"\)")
        cursor = end
    pieces.append(latex_escape(text[cursor:]))
    return "".join(pieces)


def _raw_latex_spans(text: str) -> list[tuple[int, int, str]]:
    """Find raw LaTeX math commands emitted without surrounding math delimiters."""
    command_spans: list[tuple[int, int]] = []
    cursor = 0
    for command in _RAW_LATEX_COMMAND.finditer(text):
        if command.start() < cursor:
            continue
        end = _consume_latex_suffix(text, command.end())
        command_spans.append((command.start(), end))
        cursor = end
    spans: list[tuple[int, int, str]] = []
    for start, end in command_spans:
        if spans and _RAW_MATH_BRIDGE.fullmatch(text[spans[-1][1]:start]):
            previous_start = spans[-1][0]
            spans[-1] = (previous_start, end, text[previous_start:end])
        else:
            spans.append((start, end, text[start:end]))
    return spans


def _consume_latex_suffix(text: str, position: int) -> int:
    """Consume braces, parentheses, subscripts, and superscripts after a command."""
    end = position
    while end < len(text):
        whitespace = end
        while whitespace < len(text) and text[whitespace].isspace():
            whitespace += 1
        if whitespace < len(text) and text[whitespace] in "({":
            closing = ")" if text[whitespace] == "(" else "}"
            boundary = _balanced_delimiter_end(text, whitespace, text[whitespace], closing)
            if boundary is None:
                break
            end = boundary
            continue
        if whitespace < len(text) and text[whitespace] in "_^":
            end = whitespace + 1
            if end < len(text) and text[end] in "({":
                closing = ")" if text[end] == "(" else "}"
                boundary = _balanced_delimiter_end(text, end, text[end], closing)
                if boundary is None:
                    break
                end = boundary
            else:
                while end < len(text) and text[end].isalnum():
                    end += 1
                if end < len(text) and text[end] == "*":
                    end += 1
            continue
        break
    return end


def _balanced_delimiter_end(text: str, start: int, opening: str, closing: str) -> int | None:
    depth = 0
    for position in range(start, len(text)):
        if text[position] == opening:
            depth += 1
        elif text[position] == closing:
            depth -= 1
            if depth == 0:
                return position + 1
    return None


def _latex_math_fragment(value: Any) -> str:
    """Format one value for use inside an existing math environment."""
    if isinstance(value, str):
        stripped = value.strip()
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", stripped) and not any(
            marker in stripped for marker in ("\\", "^", "_", "=", "+", "*", "/")
        ):
            return r"\text{" + latex_escape(value) + "}"
    rendered = latex_text(value)
    # The table template supplies one outer ``\(...\)`` delimiter. Values
    # from generated JSON may already contain several inline math fragments;
    # remove only those delimiters while preserving their math content.
    return re.sub(r"\\\(|\\\)|\\\[|\\\]|\$\$?", "", rendered)


def _format_given_value(value: Any) -> str:
    """Render JSON calculation inputs as compact vectors or matrices."""
    if isinstance(value, dict):
        if not value:
            return r"\varnothing"
        entries = [
            r"\texttt{" + latex_escape(name) + r"} & " + _format_given_value(item)
            for name, item in value.items()
        ]
        return r"\left\{\begin{array}{rl}" + r" \\".join(entries) + r"\end{array}\right."
    if isinstance(value, str) and re.search(r"\\\(|\\\)|\\\[|\\\]|\$", value):
        # Keep commands such as ``\to`` and ``\{`` intact, but remove any
        # inline delimiters because the template supplies one outer pair.
        return re.sub(r"\\\(|\\\)|\\\[|\\\]|\$\$?", "", value)
    if isinstance(value, list):
        if not value:
            return r"\varnothing"
        if all(isinstance(row, list) for row in value):
            rows = [" & ".join(_latex_math_fragment(cell) for cell in row) for row in value]
            return r"\begin{bmatrix}" + r" \\".join(rows) + r"\end{bmatrix}"
        if not any(isinstance(entry, list) for entry in value):
            entries = r" \\".join(_latex_math_fragment(entry) for entry in value)
            return r"\begin{bmatrix}" + entries + r"\end{bmatrix}"
    return _latex_math_fragment(value)


def _normalise_given(value: Any) -> list[dict[str, str]]:
    """Prepare calculation inputs for the question-facing LaTeX layout."""
    if not isinstance(value, dict):
        return []
    return [
        {
            "label": latex_text(name),
            "value": _format_given_value(item),
        }
        for name, item in value.items()
    ]


def _normalise(data: dict[str, Any], title: str | None = None, base_dir: Path | None = None) -> dict[str, Any]:
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raw_items = [data] if isinstance(data.get("item"), dict) else []
        if raw_items:
            raw_items[0] = {"item": data["item"], "metadata": data.get("metadata", {})}
    questions: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_items, 1):
        item = entry.get("item", entry) if isinstance(entry, dict) else {}
        meta = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        item = item if isinstance(item, dict) else {}
        options = item.get("options") or {}
        if isinstance(options, list):
            options = {chr(65 + i): value for i, value in enumerate(options)}
        image_data = item.get("image") if isinstance(item.get("image"), dict) else None
        rendered_image: dict[str, str] | None = None
        if image_data and image_data.get("path"):
            image_path = Path(str(image_data["path"]))
            if not image_path.is_absolute():
                image_path = (base_dir or Path.cwd()) / image_path
            image_path = image_path.resolve()
            if not image_path.is_file():
                raise RenderError(f"Image file not found: {image_path}")
            rendered_image = {
                "path": image_path.as_posix(),
                "alt_text": latex_text(image_data.get("alt_text", "")),
                "display_caption": latex_text(image_data.get("display_caption", "")),
            }
        questions.append({
            "number": index, "stem": latex_text(item.get("stem", "")),
            "options": [(latex_escape(k), latex_text(v)) for k, v in options.items()],
            "answer": latex_escape(item.get("answer", item.get("final_answer", ""))),
            "explanation": latex_text(item.get("explanation", item.get("solution", ""))),
            "item_type": item.get("item_type", "choice"),
            "image": rendered_image,
            "given": _normalise_given(item.get("given")),
            "metadata": meta,
        })
    source = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    subject = source.get("target_concept") or data.get("target_concept") or ""
    return {"title": latex_escape(title or data.get("title") or "Generated Quiz"),
            "subject": latex_escape(subject), "questions": questions,
            "with_answers": False}


def render_latex(data: dict[str, Any], *, title: str | None = None, with_answers: bool = False,
                 cover_title: str | None = None, template_dir: Path | None = None,
                 base_dir: Path | None = None) -> str:
    """Return rendered LaTeX for a generated result dictionary."""
    base = template_dir or Path(__file__).with_name("templates")
    env = Environment(loader=FileSystemLoader(str(base)), undefined=StrictUndefined,
                      autoescape=False, keep_trailing_newline=True)
    context = _normalise(data, title, base_dir)
    context["with_answers"] = with_answers
    context["cover_title"] = latex_escape(cover_title) if cover_title else None
    return env.get_template("quiz.tex.j2").render(**context)


def render_data(data: dict[str, Any], output_path: str | Path, *, fmt: str | None = None,
                title: str | None = None, with_answers: bool = False,
                cover_title: str | None = None, base_dir: str | Path | None = None,
                xelatex: str = "xelatex") -> Path:
    """Render an in-memory result dictionary to `.tex` or `.pdf`."""
    if not isinstance(data, dict):
        raise RenderError("The JSON root must be an object")
    destination = Path(output_path)
    kind = (fmt or destination.suffix.lstrip(".") or "tex").lower()
    if kind not in {"tex", "latex", "pdf"}:
        raise RenderError("Output format must be tex, latex, or pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tex_path = destination if kind in {"tex", "latex"} else destination.with_suffix(".tex")
    resolved_base = Path(base_dir).resolve() if base_dir is not None else None
    tex_path.write_text(
        render_latex(data, title=title, with_answers=with_answers, cover_title=cover_title,
                     base_dir=resolved_base),
        encoding="utf-8",
    )
    if kind in {"tex", "latex"}:
        return tex_path
    executable = shutil.which(xelatex)
    if not executable:
        raise RenderError("XeLaTeX was not found; install TeX Live/MiKTeX and ensure xelatex is on PATH")
    command = [executable, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    # References such as \pageref{LastPage} are resolved on the second pass.
    run = None
    for _ in range(2):
        run = subprocess.run(command, cwd=tex_path.parent, capture_output=True,
                             text=True, encoding="utf-8")
        if run.returncode != 0 or re.search(r"(?m)^! ", run.stdout + "\n" + run.stderr):
            break
    if run is None or run.returncode != 0 or re.search(r"(?m)^! ", run.stdout + "\n" + run.stderr) or not destination.exists():
        log = ((run.stdout if run else "") + "\n" + (run.stderr if run else ""))[-3000:]
        raise RenderError(f"XeLaTeX compilation failed:\n{log}")
    return destination


def render_file(input_path: str | Path, output_path: str | Path, *, fmt: str | None = None,
                title: str | None = None, with_answers: bool = False,
                cover_title: str | None = None, xelatex: str = "xelatex") -> Path:
    """Render a JSON file to `.tex` or compile it to `.pdf` with XeLaTeX."""
    source = Path(input_path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"Unable to read JSON: {source} ({exc})") from exc
    if not isinstance(data, dict):
        raise RenderError("The JSON root must be an object")
    return render_data(data, output_path, fmt=fmt, title=title, with_answers=with_answers,
                       cover_title=cover_title, base_dir=source.parent, xelatex=xelatex)
