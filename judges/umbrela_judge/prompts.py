"""Verbatim UMBRELA prompt templates.

Loads the prompt templates shipped alongside this module (copied verbatim,
byte-for-byte, from the official UMBRELA repository's YAML `prefix_user`
fields: https://github.com/castorini/umbrela/tree/main/src/umbrela/prompts/
prompt_templates). These are the exact "DNA" prompts from Thomas et al.,
"Large Language Models can Accurately Predict Searcher Preferences"
(https://arxiv.org/abs/2309.10621), as packaged by UMBRELA.

Each template expects `{query}` and `{passage}` placeholders, filled in via
`str.format`.
"""

from functools import lru_cache
from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"

# Maps (style, few_shot) -> template filename, mirroring UMBRELA's own
# `qrel_{zeroshot,fewshot}_{bing,basic}.yaml` naming/selection scheme.
_TEMPLATE_FILES = {
    ("bing", False): "qrel_zeroshot_bing.txt",
    ("bing", True): "qrel_fewshot_bing.txt",
    ("basic", False): "qrel_zeroshot_basic.txt",
}

SUPPORTED_STYLES = ("bing", "basic")


@lru_cache(maxsize=None)
def _load(filename: str) -> str:
    return (_TEMPLATE_DIR / filename).read_text(encoding="utf-8")


def get_prompt_template(style: str = "bing", few_shot: bool = False) -> str:
    """Return the raw UMBRELA `prefix_user` template for the given style.

    `style` is "bing" (the original Bing/DNA prompt from Thomas et al., the
    default and most widely used UMBRELA prompt) or "basic" (UMBRELA's
    simpler, non-step-by-step prompt). `few_shot` selects the variant with
    UMBRELA's built-in worked examples baked in (bing style only; UMBRELA's
    fewshot_basic template requires externally supplied examples that are
    not part of this judge's self-contained setup).
    """
    key = (style, few_shot)
    if key not in _TEMPLATE_FILES:
        if style not in SUPPORTED_STYLES:
            raise ValueError(f"Unsupported UMBRELA prompt style: {style!r}; expected one of {SUPPORTED_STYLES}")
        raise ValueError("UMBRELA few-shot prompting is only available for style='bing'")
    return _load(_TEMPLATE_FILES[key])


def render_prompt(style: str, few_shot: bool, query: str, passage: str) -> str:
    """Fill in `{query}`/`{passage}` placeholders of the selected template."""
    template = get_prompt_template(style=style, few_shot=few_shot)
    return template.format(query=query, passage=passage)
