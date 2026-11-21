"""The compiled code of a strategy class whose source cannot be read, hashed for its version
id (18-multi-strategy 12.1; 2.4.0 item P5-3), and the canonical JSON text every hash of a
version id is taken over.

A class defined in a notebook cell, or built with `exec` or `type()`, has no source file
`inspect` can read, but Python keeps its compiled code. So that an edit of such a class's
code gives it a new version id, `compiled_code_hash` hashes, for the class and each of its
own base classes up to `bt.Strategy` (in that order):

- its qualified name;
- each function defined on it, in name order: a method, a static or class method, a
  property's getter, setter and deleter, a cached property's function; for each, its code
  object's bytecode, constants (a nested function's or a lambda's code object read the same
  way, a frozenset's items sorted), names, variable names, argument counts and flags, its
  default argument values, and the functions it wraps (`__wrapped__`) or closes over (a
  decorator's inner function and the method it calls);
- each plain attribute, in name order: str, int, float, complex, bool, bytes and None, and
  tuples, lists, sets, dicts and dataclass values (such as `bt.points(3)`) holding only
  those, each value as its type's name and its text. Python's own entries (`__module__`,
  `__doc__` and the like) and any other value are not hashed.

Line numbers are not part of compiled code, so moving the class down its cell or adding a
comment keeps the hash, while an edit of a method's body, a constant, a default argument
or a plain attribute changes it. Compiled code depends on the Python version, so the hashed
text names it (`python_version`, "CPython 3.12"): the hash is the same in every process of
that version, whatever its hash seed (every set's items are sorted), and another version
gives another. A class holding any other callable (a builtin, an object called as a method)
has code Python does not keep: `compiled_code_hash` is then None.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import platform
import sys
from types import CodeType, EllipsisType, FunctionType, NoneType
from typing import Any

from ..strategy import Strategy

#: The start of a compiled code hash, which stands in a version id where a source hash would.
COMPILED_PREFIX = "compiled:sha256:"
#: The types whose values are hashed as their type's name and their text (`repr`, the same
#: in every process): the exact types only, so a subclass's own text is never relied on.
_SCALARS = (NoneType, EllipsisType, bool, int, float, complex, str, bytes)


def canonical_json(value: Any) -> str:
    """The canonical JSON text of `value`: keys sorted, no spaces, ASCII."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def python_version() -> str:
    """The Python compiled code depends on: its implementation and minor version, such as
    "CPython 3.12"."""
    return f"{platform.python_implementation()} {sys.version_info.major}.{sys.version_info.minor}"


def compiled_code_hash(strategy: type) -> str | None:
    """`COMPILED_PREFIX` and the SHA-256, in hexadecimal, of the canonical JSON text of
    `strategy`'s compiled code (module docstring), or None when a member of the class holds
    a callable whose code Python does not keep."""
    classes = []
    for klass in strategy.__mro__:
        if klass is Strategy:
            break
        described = _class(klass)
        if described is None:
            return None
        classes.append(described)
    text = canonical_json({"classes": classes, "python": python_version()})
    return COMPILED_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _class(klass: type) -> dict[str, Any] | None:
    """`klass`'s own functions and plain attributes as JSON types (module docstring), None
    when a member's code cannot be read."""
    functions: dict[str, Any] = {}
    attributes: dict[str, Any] = {}
    for name, value in sorted(vars(klass).items(), key=lambda item: item[0]):
        held = _functions(value)
        if held is None:
            return None
        if held:
            codes = [_function(function, set()) for function in held]
            if None in codes:
                return None
            functions[name] = codes
        elif not (name.startswith("__") and name.endswith("__")):
            plain = _encoded(value)
            if plain is not None:
                attributes[name] = plain
    return {"attributes": attributes, "functions": functions, "name": klass.__qualname__}


def _functions(value: Any) -> list[FunctionType] | None:
    """The Python functions a member of a class holds, in a fixed order (module docstring):
    none for a member that holds no code (a value, a class), None for a callable whose code
    Python does not keep."""
    if isinstance(value, staticmethod | classmethod):
        return _functions(value.__func__)
    if isinstance(value, property):
        accessors = (value.fget, value.fset, value.fdel)
        held = [_functions(each) for each in accessors if each is not None]
        return None if None in held else [function for each in held for function in each]
    if isinstance(value, functools.cached_property):
        return _functions(value.func)
    if isinstance(value, FunctionType):
        return [value]
    return [] if isinstance(value, type) or not callable(value) else None


def _function(function: FunctionType, outer: set[int]) -> dict[str, Any] | None:
    """`function`'s compiled code, default values and the functions it wraps or closes
    over, as JSON types; None when a constant of its code, or of one of those functions',
    is of a type compiled code does not hold. `outer` are the functions being read around
    it: one met again (a function closing over itself) is named, not read again."""
    if id(function) in outer:
        return {"again": function.__qualname__}
    code = _encoded(function.__code__)
    inner = [_function(each, outer | {id(function)}) for each in _inner(function)]
    if code is None or None in inner:
        return None
    keyword = function.__kwdefaults__ or {}
    defaults = {
        "keyword": {name: _default(value) for name, value in keyword.items()},
        "positional": [_default(value) for value in function.__defaults__ or ()],
    }
    return {"code": code, "defaults": defaults, "inner": inner}


def _inner(function: FunctionType) -> list[FunctionType]:
    """The functions `function` wraps (`__wrapped__`, set by `functools.wraps`) or closes
    over, in that order (its free variables' order)."""
    held = [vars(function).get("__wrapped__")]
    for cell in function.__closure__ or ():
        try:
            held.append(cell.cell_contents)
        # A free variable not bound yet holds nothing to hash.
        except ValueError:
            continue
    return [each for each in held if isinstance(each, FunctionType)]


def _default(value: Any) -> Any:
    """A default argument's value as JSON types (`_encoded`); one that is not plain by its
    type's name alone, since its text may name a memory address."""
    encoded = _encoded(value)
    return ["object", type(value).__qualname__] if encoded is None else encoded


def _encoded(value: Any) -> Any:
    """`value` as JSON types that are the same in every process, tagged with its type's
    name (module docstring), or None for a value of any other type."""
    if type(value) in _SCALARS:
        return [type(value).__name__, repr(value)]
    if isinstance(value, CodeType):
        code = _code(value)
        return None if code is None else ["code", code]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = [_encoded(getattr(value, field.name)) for field in dataclasses.fields(value)]
        return None if None in fields else [type(value).__qualname__, fields]
    if isinstance(value, dict):
        pairs = [[_encoded(key), _encoded(item)] for key, item in value.items()]
        return None if any(None in pair for pair in pairs) else _sorted("dict", pairs)
    if isinstance(value, tuple | list | set | frozenset):
        items = [_encoded(item) for item in value]
        if None in items:
            return None
        if isinstance(value, set | frozenset):
            return _sorted(type(value).__name__, items)
        return [type(value).__name__, items]
    return None


def _sorted(tag: str, items: list[Any]) -> list[Any]:
    """An unordered container's encoded items in the order of their canonical text, which
    does not depend on the process's hash seed, after `tag`."""
    return [tag, sorted(items, key=canonical_json)]


def _code(code: CodeType) -> dict[str, Any] | None:
    """A code object as JSON types: its name, bytecode (`co_code`, which holds no line
    numbers and no state of the running interpreter), constants, names, variable names,
    argument counts and flags; None when a constant cannot be encoded."""
    constants = [_encoded(constant) for constant in code.co_consts]
    if None in constants:
        return None
    return {
        "arguments": [code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount],
        "bytecode": code.co_code.hex(),
        "constants": constants,
        "flags": code.co_flags,
        "name": code.co_name,
        "names": list(code.co_names),
        "variables": [list(code.co_varnames), list(code.co_cellvars), list(code.co_freevars)],
    }


__all__ = ["COMPILED_PREFIX", "canonical_json", "compiled_code_hash", "python_version"]
