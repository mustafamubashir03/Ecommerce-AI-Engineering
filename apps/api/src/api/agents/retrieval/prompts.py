"""Prompt templates, loaded from the `prompts` directory.

Kept apart from the catalogue so a prompt can be found, checked and edited
without reading any retrieval code.
"""

from pathlib import Path

import yaml
from jinja2 import Template

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def resolve_prompt_path(file_name: str) -> Path:
    """Locate a prompt file regardless of the process working directory."""
    candidates = [
        PROMPTS_DIR / file_name,
        PROMPTS_DIR / "utils" / file_name,
        Path("prompts") / file_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Prompt file {file_name!r} not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def render_prompt(prompt_name: str, file_name: str, **variables) -> str:
    """Render one prompt, refusing a file whose declared name does not match."""
    with open(resolve_prompt_path(file_name), "r", encoding="utf-8") as handle:
        prompt_config = yaml.safe_load(handle)

    if prompt_config["name"] != prompt_name:
        raise ValueError(
            f"Expected prompt '{prompt_name}', "
            f"but found '{prompt_config['name']}'"
        )

    return Template(prompt_config["template"]).render(**variables)
