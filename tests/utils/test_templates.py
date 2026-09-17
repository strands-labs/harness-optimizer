"""Tests for the built-in template utilities."""

from strands_harness_optimizer.utils.templates import (
    create_template,
    list_builtin_templates,
    load_builtin_template,
)


def test_variables_are_not_html_escaped():
    """Prompts are not HTML: quotes, angle brackets and ampersands must pass through."""
    body = 'search(query: str) -> "x" & <tag>'
    assert create_template("{{ body }}").render(body=body) == body


def test_sandbox_still_blocks_attribute_escapes():
    """Turning autoescape off must not turn the sandbox off."""
    import pytest
    from jinja2.exceptions import SecurityError

    t = create_template("{{ x.__class__.__mro__ }}")
    with pytest.raises(SecurityError):
        t.render(x="s")


def test_builtin_templates_load():
    names = list_builtin_templates()
    assert "skill_library/task_message.jinja" in names
    assert "multi_surface/task_message.jinja" in names
    load_builtin_template("skill_library/system_prompt.jinja")
    load_builtin_template("multi_surface/system_prompt.jinja")
