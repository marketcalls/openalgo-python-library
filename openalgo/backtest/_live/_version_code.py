"""The compiled code of a strategy class whose source cannot be read, hashed for its version
id (18-multi-strategy 12.1; 2.4.0 item P5-3).

A class defined in a notebook cell, or built with `exec` or `type()`, has no source file
`inspect` can read, but Python keeps its compiled code. So that an edit of such a class's
code gives it a new version id, `compiled_code` hashes, for the class and each of its own
base classes up to `bt.Strategy` (in that order), its qualified name and each member of its
own, in name order:

- a member that holds Python functions: a method, a static or class method, a property's
  getter, setter and deleter, a cached property's function, and the function a wrapper
  object keeps as `__wrapped__` (a method cached with `functools.lru_cache` or
  `functools.cache`). For each function, its code object's bytecode, constants (a nested
  function's or a lambda's code object read the same way, a tuple's items in order, a
  frozenset's sorted, any other constant, such as the slice Python 3.14 keeps for
  `x[1:]`, as its type's name and its text), names, variable names, argument counts and
  flags, its default argument values, and the functions it wraps (`__wrapped__`) or
  closes over (a decorator's inner function and the method it calls);
- a member that holds another callable, whose code Python does not keep: a builtin (such
  as `staticmethod(math.floor)`) by its module and name, any other (an object called as
  a method) as a value;
- any other member as a value, such as 5, ("SBIN", "INFY"), bt.points(3) or
  datetime.time(9, 20). Python's own entries (`__module__`, `__doc__` and the like) are
  not hashed.

A value is hashed by the rule a parameter is (`_version_values.canonical_value`); one known
by its type alone (its text names a memory address) is named in `CompiledCode.by_type`, so
the start banner names it: a default argument as "sized's default model", any other by its
member's name.

Line numbers are not part of compiled code, so moving the class down its cell or adding a
comment keeps the hash, while an edit of a method's body, a constant, a default argument
or a class attribute changes it. Compiled code depends on the Python version, so the hashed
text names it (`python_version`, "CPython 3.12"): the hash is the same in every process of
that version, whatever its hash seed (every set's items are sorted), and another version
gives another. A class whose code cannot be read at all, one that holds callables none of
which is a Python function (every one a builtin or an object called as a method), gets
None: its version id hashes its name instead (`_version_id`).
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import platform
import sys
from dataclasses import dataclass, field
from types import BuiltinFunctionType, CodeType, FunctionType, ModuleType
from typing import Any

from ..strategy import Strategy
from ._version_values import canonical_json, canonical_value

#: The start of a compiled code hash, which stands in a version id where a source hash would.
COMPILED_PREFIX = "compiled:sha256:"


@dataclass(frozen=True)
class CompiledCode:
    """A strategy class's compiled code, hashed (module docstring): `code_hash` is
    `COMPILED_PREFIX` and the SHA-256, in hexadecimal, of its canonical JSON text, and
    `by_type` names the values in it known by their type alone, in the order read."""

    code_hash: str
    by_type: tuple[str, ...] = ()


@dataclass
class _Reading:
    """What reading a class's members has found so far: the values known by their type
    alone (`by_type`), and how many callables were Python functions (`functions`) and how
    many held code Python does not keep (`opaque`)."""

    by_type: list[str] = field(default_factory=list)
    functions: int = 0
    opaque: int = 0


def python_version() -> str:
    """The Python compiled code depends on: its implementation and minor version, such as
    "CPython 3.12"."""
    return f"{platform.python_implementation()} {sys.version_info.major}.{sys.version_info.minor}"


def compiled_code(strategy: type) -> CompiledCode | None:
    """`strategy`'s compiled code, hashed (module docstring), or None when its code
    cannot be read at all."""
    reading = _Reading()
    classes = []
    for klass in strategy.__mro__:
        if klass is Strategy:
            break
        classes.append(_class(klass, reading))
    if reading.opaque and not reading.functions:
        return None
    text = canonical_json({"classes": classes, "python": python_version()})
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return CompiledCode(COMPILED_PREFIX + digest, tuple(dict.fromkeys(reading.by_type)))


def _class(klass: type, reading: _Reading) -> dict[str, Any]:
    """`klass`'s own members as JSON types (module docstring): those holding callables
    under `functions`, the others under `attributes`."""
    functions: dict[str, Any] = {}
    attributes: dict[str, Any] = {}
    for name, value in sorted(vars(klass).items(), key=lambda item: item[0]):
        held = _callables(value)
        if held:
            functions[name] = [_callable(name, each, reading) for each in held]
        elif not (name.startswith("__") and name.endswith("__")):
            attributes[name] = _value(name, value, reading)
    return {"attributes": attributes, "functions": functions, "name": klass.__qualname__}


def _callables(value: Any) -> list[Any]:
    """The callables a member of a class holds, in a fixed order (module docstring): the
    functions a descriptor holds, the function a wrapper object keeps as `__wrapped__` in
    its place, and none for a member that holds no callable (a value or a class)."""
    if isinstance(value, staticmethod | classmethod):
        return _callables(value.__func__)
    if isinstance(value, property):
        accessors = (value.fget, value.fset, value.fdel)
        return [function for accessor in accessors for function in _callables(accessor)]
    if isinstance(value, functools.cached_property):
        return _callables(value.func)
    if isinstance(value, type) or not callable(value):
        return []
    if isinstance(value, FunctionType):
        return [value]
    # A function's own `__wrapped__` is read with it (`_inner`), so unwrapping stops there.
    wrapped = inspect.unwrap(value, stop=lambda each: isinstance(each, FunctionType))
    return [value] if wrapped is value else _callables(wrapped)


def _callable(member: str, value: Any, reading: _Reading) -> Any:
    """A callable `member` holds as JSON types: a Python function's compiled code, a
    builtin's module and name, any other as a value (module docstring)."""
    if isinstance(value, FunctionType):
        reading.functions += 1
        return _function(member, value, set(), reading)
    reading.opaque += 1
    if isinstance(value, BuiltinFunctionType) and isinstance(value.__self__, ModuleType):
        return ["builtin", f"{value.__module__}.{value.__qualname__}"]
    return ["callable", _value(member, value, reading)]


def _value(name: str, value: Any, reading: _Reading) -> Any:
    """`value` by the rule a parameter's is hashed by, its `name` noted when it is known by
    its type alone."""
    text, typed = canonical_value(value)
    if typed:
        reading.by_type.append(name)
    return text


def _function(member: str, function: FunctionType, outer: set[int], reading: _Reading) -> Any:
    """`function`'s compiled code, default values and the functions it wraps or closes
    over, as JSON types. `outer` are the functions being read around it: one met again (a
    function closing over itself) is named, not read again."""
    if id(function) in outer:
        return {"again": function.__qualname__}
    around = outer | {id(function)}
    inner = [_function(member, each, around, reading) for each in _inner(function)]
    defaults = {
        argument: _value(f"{member}'s default {argument}", value, reading)
        for argument, value in _defaults(function).items()
    }
    return {"code": _code(function.__code__), "defaults": defaults, "inner": inner}


def _defaults(function: FunctionType) -> dict[str, Any]:
    """`function`'s default argument values by argument name: the positional ones are
    those of its last positional arguments."""
    code = function.__code__
    positional = function.__defaults__ or ()
    names = code.co_varnames[code.co_argcount - len(positional) : code.co_argcount]
    return dict(zip(names, positional, strict=True)) | (function.__kwdefaults__ or {})


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


def _code(code: CodeType) -> dict[str, Any]:
    """A code object as JSON types: its name, bytecode (`co_code`, which holds no line
    numbers and no state of the running interpreter), constants, names, variable names,
    argument counts and flags."""
    return {
        "arguments": [code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount],
        "bytecode": code.co_code.hex(),
        "constants": [_constant(constant) for constant in code.co_consts],
        "flags": code.co_flags,
        "name": code.co_name,
        "names": list(code.co_names),
        "variables": [list(code.co_varnames), list(code.co_cellvars), list(code.co_freevars)],
    }


def _constant(value: Any) -> Any:
    """A constant of compiled code as JSON types, tagged with its type's name (module
    docstring). Compiled code holds only values Python can write to a .pyc file: none
    names a memory address, so the text of any other than a code object, a tuple or a
    frozenset is the same in every process."""
    if isinstance(value, CodeType):
        return ["code", _code(value)]
    if isinstance(value, tuple):
        return ["tuple", [_constant(each) for each in value]]
    if isinstance(value, frozenset):
        return ["frozenset", sorted((_constant(each) for each in value), key=canonical_json)]
    return [type(value).__name__, repr(value)]


__all__ = ["COMPILED_PREFIX", "CompiledCode", "compiled_code", "python_version"]
