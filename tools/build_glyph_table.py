"""Build a glyph-id to text table from an upstream JetBrains Mono TTF. Issue #48.

The PDF embeds JetBrains Mono v2.221 as a subset that keeps the original glyph order. Its
ToUnicode table maps the ligature spacer glyph `SPC` to `=`, so `->` reads as `=>`. The upstream
font names every glyph: `SPC`, `hyphen_greater.liga`, `exclam_equal_equal.liga`. This script turns
those names into the text each glyph stands for.

Usage:
    uv run --with fonttools python tools/build_glyph_table.py \
        path/to/JetBrainsMono-Regular.ttf src/flp_rag/ingest/glyph_tables/jetbrainsmono-2.221-regular.json

Only glyphs whose text differs from what ToUnicode would give are needed: the spacer (empty
text) and the `.liga` glyphs (their component characters). Plain glyphs are left to ToUnicode.
"""

import json
import sys
from pathlib import Path

from fontTools import agl
from fontTools.ttLib import TTFont

SPACER_NAMES = {"SPC"}  # the only empty spacer observed in the PDF (glyph id 1205)
LIGA_SUFFIX = ".liga"


def liga_text(name: str) -> str:
    """`hyphen_greater.liga` -> `->`. Every component is an Adobe Glyph List name."""
    parts = name[: -len(LIGA_SUFFIX)].split("_")
    chars = [agl.toUnicode(p) for p in parts]
    missing = [p for p, c in zip(parts, chars, strict=True) if c == ""]
    if missing:
        raise ValueError(f"{name}: components not in the Adobe Glyph List: {missing}")
    return "".join(chars)


def build_table(ttf_path: Path) -> dict:
    font = TTFont(str(ttf_path))
    order = font.getGlyphOrder()
    name_table = font["name"]
    glyphs: dict[str, str] = {}
    for gid, name in enumerate(order):
        if name in SPACER_NAMES:
            glyphs[str(gid)] = ""
        elif name.endswith(LIGA_SUFFIX):
            glyphs[str(gid)] = liga_text(name)
    return {
        "font": name_table.getDebugName(4),
        "version": name_table.getDebugName(5),
        "num_glyphs": len(order),
        "source": ttf_path.name,
        "glyphs": glyphs,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    ttf, out = Path(argv[1]), Path(argv[2])
    table = build_table(ttf)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n")
    print(f"{out}: {table['font']} {table['version']}, {table['num_glyphs']} glyphs, "
          f"{len(table['glyphs'])} overrides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
