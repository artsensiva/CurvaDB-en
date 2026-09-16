"""Shared utility for incrementally updating benchmarks/results/step1.md:
several scripts (step1_clean.py, step1_spline_fit.py, ...) each write their
own section, without clobbering the others."""

from __future__ import annotations

import os


def upsert_section(path: str, header: str, body: str) -> None:
    """Replaces the section starting at the line `header` (exactly
    `## ...`) and ending before the next `## ` or end of file, with
    `body`. If no such section exists, appends `body` to the end of the
    file."""
    content = ""
    if os.path.exists(path):
        with open(path) as f:
            content = f.read()

    lines = content.split("\n") if content else []
    start = next((i for i, ln in enumerate(lines) if ln.strip() == header.strip()), None)

    if start is None:
        new_content = (content.rstrip("\n") + "\n\n" if content.strip() else "") + body.rstrip("\n") + "\n"
    else:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if lines[j].startswith("## "):
                end = j
                break
        new_lines = lines[:start] + body.rstrip("\n").split("\n") + [""] + lines[end:]
        new_content = "\n".join(new_lines).rstrip("\n") + "\n"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(new_content)
