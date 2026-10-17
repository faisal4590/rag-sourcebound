# The one [Sn] marker parser that Stages 16, 17, and 18 share. Issue #22, Part 2.
"""Block markers in an answer.

`[S1]` is one marker, `[S1][S3]` two, `[S1, S3]` two (issue #22, open question 3). `(S1)`,
`Source 1`, and a bare `S1` are no marker. Labels come back in reading order, duplicates kept,
so a citation count is the length of the list and a set gives the distinct labels.

A bracket group that looks like a citation but is not one of the accepted forms (`[S1-S9]`,
`[s9]`, `[S 9]`, `[S9 ]`) is malformed: `find_malformed` returns it so that Stage 17 can reject
it instead of letting stray citation text reach the client.
"""

import re

MARKER_GROUP = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
LABEL = re.compile(r"S\d+")
BRACKET_GROUP = re.compile(r"\[[^\[\]]*\]")
LOOKS_LIKE_LABEL = re.compile(r"\bS\s*\d", re.IGNORECASE)


def find_markers(text: str) -> list[str]:
    """Every label cited in `text`, in order: `"[S1][S2, S3]"` -> `["S1", "S2", "S3"]`."""
    return [
        label for group in MARKER_GROUP.finditer(text) for label in LABEL.findall(group.group(1))
    ]


def has_marker(text: str) -> bool:
    return MARKER_GROUP.search(text) is not None


def find_malformed(text: str) -> list[str]:
    """Bracket groups that hold something like a label but are not an accepted marker."""
    return [
        group.group(0)
        for group in BRACKET_GROUP.finditer(text)
        if LOOKS_LIKE_LABEL.search(group.group(0)) and not MARKER_GROUP.fullmatch(group.group(0))
    ]
