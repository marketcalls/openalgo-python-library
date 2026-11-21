"""The canonical text of what a strategy's version id hashes (18-multi-strategy 12.1): the
one rule for its values, and the JSON text every hash of a version id is taken over.

A value is a parameter's, and, for a class whose source cannot be read (`_version_code`),
a class attribute's, a default argument's or that of a callable whose code Python does not
keep. `canonical_value` reads each the same way: its JSON-ready copy (`_manifest.plain`: a
date as its ISO text, a tuple as a list, a number, a string, a bool or None as itself, any
other object as its text, such as "datetime.time(9, 20)"), a set's items sorted (a set's
own order depends on the process's hash seed), and an object whose text names its memory
address (which differs from run to run) by that text without the address, its type alone.
It says which values it knew by their type alone, so the start banner names them: a
change to such a value does not change the id.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .._manifest import plain

#: The memory address in the text of an object with no text of its own ("<Model object
#: at 0x...>"): it differs from run to run, so the id leaves it out.
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def canonical_json(value: Any) -> str:
    """The canonical JSON text of `value`: keys sorted, no spaces, ASCII."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_value(value: Any) -> tuple[Any, bool]:
    """`value` as JSON types that are the same in every process (module docstring), and
    whether a part of it was known by its type alone."""
    if isinstance(value, Mapping):
        parts = {str(key): canonical_value(item) for key, item in value.items()}
        return {key: text for key, (text, _) in parts.items()}, any(
            typed for _, typed in parts.values()
        )
    if isinstance(value, list | tuple | set | frozenset):
        items = [canonical_value(item) for item in value]
        texts = [text for text, _ in items]
        if isinstance(value, set | frozenset):
            texts.sort(key=canonical_json)
        return texts, any(typed for _, typed in items)
    text = plain(value)
    if isinstance(text, str) and not isinstance(value, str):
        bare = _ADDRESS.sub("", text)
        return bare, bare != text
    return text, False


__all__ = ["canonical_json", "canonical_value"]
