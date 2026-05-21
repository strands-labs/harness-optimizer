# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydoc-markdown>=4.8.2",
# ]
# ///
"""Generate markdown documentation for harness_optimizer using pydoc-markdown.

Adapted from strands-agents/docs/scripts/api-generation-python.py.

This script generates per-module markdown files in the docs/src/content/docs/api/ directory,
organized by package subdirectory for proper sidebar grouping.

Usage:
    pip install pydoc-markdown && python docs/scripts/api-generation-python.py
"""

import shutil
from pathlib import Path

from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.processors.crossref import CrossrefProcessor
from pydoc_markdown.contrib.processors.smart import SmartProcessor


def generate_docs():
    # Paths relative to project root
    input_path = "."
    output_path = "docs/src/content/docs/api"

    output_dir = Path(output_path)

    # Delete existing output directory to ensure clean generation
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Deleted existing output directory: {output_dir}")

    output_dir.mkdir(exist_ok=True, parents=True)

    # Configure the session
    session = PydocMarkdown()

    # Configure the Python loader
    loader = PythonLoader(
        search_path=[input_path],
        packages=["harness_optimizer"],
    )
    session.loaders = [loader]

    # Configure processors
    session.processors = [
        FilterProcessor(skip_empty_modules=True),
        CrossrefProcessor(),
        SmartProcessor(),
    ]

    # Configure the renderer
    renderer = MarkdownRenderer(
        render_module_header=False,
        descriptive_class_title="",
        add_module_prefix=True,
        render_toc=False,
    )
    session.renderer = renderer

    # Load and process modules
    modules = session.load_modules()
    session.process(modules)

    # Modules to exclude from documentation
    excluded_modules = {
        "harness_optimizer.compat",  # Legacy names, not primary API
    }

    module_files = []

    # Write each module to a separate file, organized by subdirectory
    for module in modules:
        module_name = module.name

        # Skip private modules
        if any(part.startswith("_") for part in module_name.split(".")):
            print(f"Skipping private module: {module_name}")
            continue

        # Skip excluded modules
        if module_name in excluded_modules:
            print(f"Skipping excluded module: {module_name}")
            continue

        # Create subdirectory structure: harness_optimizer.formulas.formula -> formulas/formula.md
        parts = module_name.split(".")
        if len(parts) > 1:
            # Remove "harness_optimizer" prefix for path
            rel_parts = parts[1:]
        else:
            rel_parts = parts

        if len(rel_parts) > 1:
            subdir = output_dir / "/".join(rel_parts[:-1])
            subdir.mkdir(parents=True, exist_ok=True)
            filepath = subdir / f"{rel_parts[-1]}.md"
        else:
            filepath = output_dir / f"{rel_parts[0]}.md"

        # Short title: last part of module name
        short_title = rel_parts[-1]

        # Render single module
        content = renderer.render_to_string([module])

        content = f"""---
title: "{module_name}"
description: API reference for {module_name}
sidebar:
  label: {short_title}
---

{content}
""".strip()

        if content.strip():
            filepath.write_text(content, encoding="utf-8")
            module_files.append((module_name, str(filepath.relative_to(output_dir))))
            print(f"Generated: {filepath}")

    print(f"\nTotal modules documented: {len(module_files)}")


if __name__ == "__main__":
    generate_docs()
