"""Stage 6 of the pipeline: write results into README.md automatically.

README.md contains pairs of markers like:

    <!-- PART_A_START -->
    ...results go here...
    <!-- PART_A_END -->

Scripts call inject_into_readme("PART_A", text) to replace whatever is between
the markers, so the README always shows the latest results.
"""

from pathlib import Path

from src.utils import REPO_ROOT


def inject_into_readme(section: str, content: str, readme_path: Path | None = None) -> bool:
    """Replace the text between the START/END markers of `section`. Returns False if markers are missing."""
    readme_path = Path(readme_path or REPO_ROOT / "README.md")
    start, end = f"<!-- {section}_START -->", f"<!-- {section}_END -->"
    text = readme_path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        print(f"[report] markers for {section} not found in README; skipped")
        return False
    before = text.split(start, 1)[0]
    after = text.split(end, 1)[1]
    readme_path.write_text(f"{before}{start}\n{content.strip()}\n{end}{after}", encoding="utf-8")
    print(f"[report] README section {section} updated")
    return True


def markdown_table(header: list[str], rows: list[list]) -> str:
    """Build a markdown table from a header and a list of rows."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)
