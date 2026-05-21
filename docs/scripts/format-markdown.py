"""Format plain markdown files for Starlight by adding frontmatter.

Reads markdown files from docs/ directories, extracts the # Title heading,
generates Starlight-compatible frontmatter, and writes formatted files to
docs/src/content/docs/ for the Astro build.

Source files stay as plain markdown — this script bridges to Starlight.

Usage:
    python docs/scripts/format-markdown.py
"""

import re
import shutil
from pathlib import Path

# Mapping of source directories to Starlight content directories
CONTENT_DIRS = {
    "docs/user-guide": "docs/src/content/docs/user-guide",
    "docs/ROADMAP.md": "docs/src/content/docs/roadmap.md",
}

# Custom descriptions for known pages
DESCRIPTIONS = {
    "Formulas": "Define tunable context units for LLM agents.",
    "Adapters": "Bridge Formulas to agent frameworks.",
    "Rewards": "Score agent rollouts with reward functions.",
    "Optimizers": "Optimize Formula parameters from agent rollouts.",
    "ContrastiveReflectionOptimizer": "Contrastive learning on rollout traces.",
    "Training": "Automated training loop with Dataset, DataLoader, and Trainer.",
    "Implementation Roadmap": "Staged CR plan for the Harness Optimizer core package.",
}


def extract_title(content: str) -> str | None:
    """Extract the first # heading from markdown content."""
    match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else None


def add_frontmatter(content: str, title: str, description: str = "") -> str:
    """Add Starlight frontmatter and remove the # heading (Starlight renders it from title)."""
    # Remove the first # heading since Starlight generates it from frontmatter
    content = re.sub(r"^#\s+.+\n*", "", content, count=1)
    frontmatter = f"---\ntitle: \"{title}\"\n"
    if description:
        frontmatter += f"description: \"{description}\"\n"
    frontmatter += "---\n\n"
    return frontmatter + content


def format_file(src: Path, dst: Path) -> None:
    """Format a single markdown file with frontmatter."""
    content = src.read_text(encoding="utf-8")

    # Skip if already has frontmatter
    if content.startswith("---"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {src} → {dst} (already has frontmatter)")
        return

    title = extract_title(content)
    if not title:
        print(f"  Skipping {src} (no # heading found)")
        return

    description = DESCRIPTIONS.get(title, f"Documentation for {title}.")
    formatted = add_frontmatter(content, title, description)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(formatted, encoding="utf-8")
    print(f"  {src} → {dst}")


def main():
    print("Formatting markdown files for Starlight...\n")

    for src_path, dst_path in CONTENT_DIRS.items():
        src = Path(src_path)
        dst = Path(dst_path)

        if src.is_file():
            format_file(src, dst)
        elif src.is_dir():
            # Remove existing destination to avoid stale files
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)

            for md_file in sorted(src.rglob("*.md")):
                rel = md_file.relative_to(src)
                format_file(md_file, dst / rel)
        else:
            print(f"  Warning: {src} not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
