"""Utility for loading built-in Jinja2 templates."""

import os

from jinja2 import Template
from jinja2.sandbox import SandboxedEnvironment

# Built-in templates live under harness_optimizer/templates/
_BUILTIN_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "templates")

_ENV = SandboxedEnvironment(autoescape=True)


def create_template(source: str) -> Template:
    """Create a sandboxed Jinja2 Template from a string.

    Args:
        source: Jinja2 template source string.

    Returns:
        Jinja2 Template object backed by a SandboxedEnvironment.
    """
    return _ENV.from_string(source)


def load_builtin_template(path: str) -> Template:
    """Load a built-in Jinja2 template by relative path.

    Args:
        path: Relative path within the built-in templates directory.
            e.g., "contrastive_reflection/task_message_system_prompt.jinja"

    Returns:
        Jinja2 Template object backed by a SandboxedEnvironment.

    Raises:
        FileNotFoundError: If the template does not exist.
    """
    full_path = os.path.normpath(os.path.join(_BUILTIN_TEMPLATE_DIR, path))
    with open(full_path) as f:
        return create_template(f.read())


def list_builtin_templates() -> list[str]:
    """List all available built-in template paths.

    Returns:
        List of relative paths to built-in templates.
    """
    templates = []
    base = os.path.normpath(_BUILTIN_TEMPLATE_DIR)
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".jinja"):
                rel = os.path.relpath(os.path.join(root, f), base)
                templates.append(rel)
    return sorted(templates)
