from __future__ import annotations

import json
from pathlib import Path

from qg_v2.render import render_data, render_file, render_latex


def test_render_group_and_escape() -> None:
    data = {
        "status": "ok",
        "items": [{"item": {"stem": "a_b & c", "options": {"A": "x%", "B": "y"}, "answer": "A"},
                   "metadata": {"target_concept": "demo"}}],
    }
    tex = render_latex(data, title="Quiz #1", with_answers=True)
    assert "Quiz \\#1" in tex
    assert "a\\_b \\& c" in tex
    assert "Answer: A" in tex
    assert "\\item x\\%" in tex


def test_unicode_formula_is_math_mode() -> None:
    tex = render_latex({"item": {"stem": "The update rule is θ = θ - α∇J(θ), where α is the learning rate."}})
    assert r"\(\theta = \theta - \alpha\nabla J(\theta)\)" in tex
    assert "θ = θ" not in tex


def test_broad_formula_symbols_and_explicit_math_are_supported() -> None:
    tex = render_latex({"item": {"stem": "The loss is L = σ(z), with x ≥ 0 and integral ∫_0^1 x^2 dx."}})
    assert r"\(L = \sigma(z)\)" in tex
    assert r"\(x \geq 0\)" in tex
    assert r"\(\int_0^1 x^2 dx\)" in tex
    explicit = render_latex({"item": {"stem": r"Formula $E=mc^2$."}})
    assert r"$E=mc^2$" in explicit


def test_english_prose_around_formula_stays_text_mode() -> None:
    text = "During training, the update formula is θ = θ - α∇J(θ) (where α is the learning rate). This method is effective."
    tex = render_latex({"item": {"stem": text}})
    assert r"During training, the update formula is \(\theta = \theta - \alpha\nabla J(\theta)\) (where \(\alpha\) is the learning rate)." in tex
    assert r"\(During training" not in tex


def test_inline_variables_do_not_consume_explanatory_prose() -> None:
    text = "The parameter θ_j is 2.5 and ∂J(θ)/∂θ_j is 3.0."
    tex = render_latex({"item": {"stem": text}})
    assert r"The parameter \(\theta_j\) is 2.5" in tex
    assert r"\(\partial J(\theta)/\partial \theta_j\) is 3.0" in tex


def test_implicit_formula_supports_braced_subscripts() -> None:
    tex = render_latex(
        {"item": {"stem": "Update w_k = w_{k-1} + d_{k-1} and choose d_{k-1}."}}
    )
    assert r"\(w_k = w_{k-1} + d_{k-1}\)" in tex
    assert r"d\_\{k-1\}" not in tex


def test_unicode_operator_keeps_nested_braced_subscript_in_one_math_block() -> None:
    text = "The gradient is ∇L(Θ_{t−1}) = (0.8, −0.4, 1.2)."

    tex = render_latex({"item": {"stem": text}})

    assert r"\(\nabla L(\Theta_{t−1})\)" in tex
    assert r"\Theta_\)\{t−1\}" not in tex


def test_raw_latex_commands_are_wrapped_in_math_mode() -> None:
    text = r"Use \hat{A}, \sigma(x), \Omega_k, and \theta^* in the update rule."
    tex = render_latex({"item": {"stem": text}})
    assert r"\(\hat{A}\)" in tex
    assert r"\(\sigma(x)\)" in tex
    assert r"\(\Omega_k\)" in tex
    assert r"\(\theta^*\)" in tex
    assert r"\textbackslash{}hat" not in tex


def test_mathcal_command_is_rendered_as_math() -> None:
    tex = render_latex(
        {
            "item": {
                "stem": r"The batch is \mathcal{B}.",
                "given": {r"\mathcal{B}": ["node1", "node2"]},
            }
        }
    )

    assert r"The batch is \(\mathcal{B}\)." in tex
    assert r"\(\mathcal{B}\) & \(\displaystyle \begin{bmatrix}" in tex
    assert r"\text{node1}" in tex and r"\text{node2}" in tex
    assert r"\textbackslash{}mathcal" not in tex


def test_contiguous_raw_latex_formula_is_kept_in_one_math_block() -> None:
    text = r"Cost: \frac{1}{P}\sum_{p=1}^{P} \left\| C C^T x_p - x_p \right\|_2^2."
    tex = render_latex({"item": {"stem": text}})
    assert r"\(\frac{1}{P}\sum_{p=1}^{P} \left\| C C^T x_p - x_p \right\|_2^2\)" in tex


def test_unicode_symbols_inside_explicit_math_are_converted() -> None:
    tex = render_latex({"item": {"stem": "Update $η = ∇L$ and $ϵ = 10^{-5}$."}})
    assert "$\\eta = \\nabla L$" in tex
    assert "$\\varepsilon = 10^{-5}$" in tex


def test_latex_commands_inside_explicit_math_are_not_corrupted() -> None:
    tex = render_latex(
        {"item": {"stem": r"$$A = \\begin{bmatrix}1 & 0\\0 & 1\\end{bmatrix}$$"}}
    )
    assert r"\begin{bmatrix}" in tex
    assert r"\end{bmatrix}" in tex
    assert r"\ begin" not in tex


def test_malformed_json_latex_control_characters_are_repaired() -> None:
    tex = render_latex(
        {"item": {"stem": "Use $\x08oldsymbol{h} \times \x0crac{a}{b} + \rho$."}}
    )
    assert r"\boldsymbol{h}" in tex
    assert r"\times" in tex
    assert r"\frac{a}{b}" in tex
    assert r"\rho" in tex


def test_answer_explanation_allows_paragraphs() -> None:
    tex = render_latex(
        {"item": {"stem": "Question", "answer": "A", "explanation": "First paragraph.\n\nSecond paragraph."}},
        with_answers=True,
    )
    assert r"\begingroup\itshape Explanation: First paragraph." in tex
    assert r"Second paragraph.\par\endgroup" in tex


def test_open_calculation_uses_blank_answer_space_without_rule() -> None:
    tex = render_latex({"item": {"item_type": "open_calculation", "stem": "Calculate the updated parameter."}})
    assert r"\vspace{4\baselineskip}" in tex
    assert r"\hrulefill" not in tex


def test_open_calculation_renders_given_vectors_and_matrices() -> None:
    tex = render_latex(
        {
            "item": {
                "item_type": "open_calculation",
                "stem": "Compute the optimal weight vector.",
                "given": {
                    "X_tilde": [[1, 1], [2, 1], [4, 1], [5, 1]],
                    "y": [2, 3, 5, 7],
                },
            }
        }
    )

    assert r"\textit{Given:}" in tex
    assert r"X\_tilde & \(\displaystyle \begin{bmatrix}1 & 1 \\2 & 1 \\4 & 1 \\5 & 1\end{bmatrix}\)" in tex
    assert r"y & \(\displaystyle \begin{bmatrix}2 \\3 \\5 \\7\end{bmatrix}\)" in tex


def test_given_dictionary_and_text_are_latex_safe() -> None:
    tex = render_latex(
        {
            "item": {
                "stem": "Use the supplied graph data.",
                "given": {
                    "hyperedges": {"e1": ["v1", "v2"], "e2": ["v2", "v3"]},
                    "instruction": "node labels are ordered",
                },
            }
        }
    )

    assert r"hyperedges & \(\displaystyle \left\{\begin{array}{rl}\texttt{e1} & \begin{bmatrix}\text{v1} \\\text{v2}\end{bmatrix}" in tex
    assert r"\text{node labels are ordered}" in tex
    assert "{'e1'" not in tex


def test_given_value_with_multiple_inline_math_fragments_is_not_nested() -> None:
    tex = render_latex(
        {
            "item": {
                "stem": "Use the supplied sets.",
                "given": {
                    "P_N": r"\(u=1\) \(\to\) \{3\}, \(u=2\) \(\to\) \{1,2\}",
                },
            }
        }
    )

    assert r"P\_N & \(\displaystyle u=1 \to \{3\}, u=2 \to \{1,2\}\)" in tex
    assert r"\(\displaystyle \(" not in tex


def test_render_legacy_single_item(tmp_path: Path) -> None:
    source = tmp_path / "result.json"
    output = tmp_path / "quiz.tex"
    source.write_text(json.dumps({"item": {"stem": "Question stem", "options": {"A": "Option"}, "answer": "A"}}, ensure_ascii=False), encoding="utf-8")
    assert render_file(source, output) == output
    assert "Question stem" in output.read_text(encoding="utf-8")


def test_render_cover_title_is_optional_and_adds_a_title_page() -> None:
    data = {"items": [{"item": {"stem": "Question", "options": {"A": "Option"}}}]}
    without_cover = render_latex(data, title="DGL Quiz: Chapter 1")
    with_cover = render_latex(
        data,
        title="DGL Quiz: Chapter 1",
        cover_title="DGL Quiz: Chapter 1",
    )

    assert "\\begin{titlepage}" not in without_cover
    assert "\\begin{titlepage}" in with_cover
    assert "DGL Quiz: Chapter 1" in with_cover
    assert with_cover.index("\\end{titlepage}") < with_cover.index("Question")


def test_render_data_supports_in_memory_tex_output(tmp_path: Path) -> None:
    output = tmp_path / "chapter.tex"
    result = render_data(
        {"items": [{"item": {"stem": "Question", "explanation": "Why"}}]},
        output,
        title="MLR Quiz: Chapter 14",
        cover_title="MLR Quiz: Chapter 14",
    )

    assert result == output
    text = output.read_text(encoding="utf-8")
    assert "\\begin{titlepage}" in text
    assert "MLR Quiz: Chapter 14" in text
