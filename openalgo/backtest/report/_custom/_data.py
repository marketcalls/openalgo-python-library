"""What each kind of user block accepts as data, checked and put in one shape before it
is drawn (`_render`): a block's data given as a value is checked when the block is made,
and data a callable returns is checked when the page is rendered, by the same
functions. Every refusal is a ConfigError that names the block (`where`)."""

import datetime as dt
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ...errors import ConfigError
from ._theme import MAX_SERIES

#: How a tile shows: a number, text as it is, or nothing known ("n/a").
TILE_VALUE = (int, float, np.integer, np.floating, str, type(None))


def called(data: object, subject: object, where: str) -> object:
    """The block's data: `data` itself, or what it returns when called with `subject`
    (the result being rendered), once. Raises ConfigError, the cause chained, when the
    call raises."""
    if not callable(data):
        return data
    try:
        return data(subject)
    except Exception as error:
        raise ConfigError(
            f"{where}: its data function raised {type(error).__name__}: {error}"
        ) from error


def _numeric(frame: pd.DataFrame, where: str) -> pd.DataFrame:
    """`frame` as floats; ConfigError naming the first column that is not numbers."""
    _unique(frame, where)
    for column in frame.columns:
        dtype = frame[column].dtype
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            raise ConfigError(f"{where}: column {column!r} is not numbers ({dtype})")
    return frame.astype(float)


def _unique(frame: pd.DataFrame, where: str) -> None:
    """ConfigError when two columns share a name: each is drawn or listed by its name."""
    repeated = frame.columns[frame.columns.duplicated()].unique().tolist()
    if repeated:
        raise ConfigError(f"{where}: the column names repeat ({repeated}); give each its own")


def series_frame(data: object, where: str, label: str) -> tuple[pd.DataFrame, bool]:
    """A Series or DataFrame of numbers as (a DataFrame, whether it draws several
    series); a Series is one column named `label`. At most `MAX_SERIES` columns."""
    if isinstance(data, pd.Series):
        return _numeric(data.to_frame(name=label), where), False
    if not isinstance(data, pd.DataFrame):
        raise ConfigError(
            f"{where}: data is a pandas Series or DataFrame, got {type(data).__name__}"
        )
    if len(data.columns) > MAX_SERIES:
        raise ConfigError(
            f"{where}: at most {MAX_SERIES} series in one chart, got {len(data.columns)} columns"
        )
    if len(data.columns) == 0:
        raise ConfigError(f"{where}: the DataFrame has no column to draw")
    return _numeric(data, where), len(data.columns) > 1


def dated(frame: pd.DataFrame, where: str) -> pd.DataFrame:
    """`frame` indexed by a DatetimeIndex: dates or datetimes (an index of
    `datetime.date` is converted); ConfigError for any other index."""
    index = frame.index
    if isinstance(index, pd.DatetimeIndex):
        return frame
    if len(index) and all(isinstance(value, dt.date) for value in index):
        try:
            return frame.set_axis(pd.DatetimeIndex(list(index)))
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{where}: its dates do not form one index: {error}") from error
    if not len(index):
        return frame.set_axis(pd.DatetimeIndex([]))
    raise ConfigError(
        f"{where}: a line is drawn over dates, so its index is dates or datetimes, got "
        f"{type(index).__name__} of {index.dtype}"
    )


def numbers(data: object, where: str) -> np.ndarray:
    """A Series or a one-dimensional array of numbers as floats."""
    values = data.to_numpy() if isinstance(data, pd.Series) else data
    if isinstance(values, list | tuple):
        values = np.asarray(values)
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise ConfigError(f"{where}: data is a Series or a one-dimensional array of numbers")
    if not np.issubdtype(values.dtype, np.number) or values.dtype == bool:
        raise ConfigError(f"{where}: the values are not numbers ({values.dtype})")
    return values.astype(float)


class Points(NamedTuple):
    """A scatter's points: x and y values, and each point's group (None: one group)."""

    x: np.ndarray
    y: np.ndarray
    group: np.ndarray | None


def _column(frame: pd.DataFrame, name: object, where: str, role: str) -> pd.Series:
    if name not in frame.columns:
        raise ConfigError(
            f"{where}: {role} names column {name!r}, which the data does not have; its "
            f"columns are {list(frame.columns)}"
        )
    return frame[name]


def points(data: object, where: str, x: str, y: str, group: str | None) -> Points:
    """The x, y and group columns of a scatter's DataFrame, x and y as numbers; at most
    `MAX_SERIES` groups, a missing group value read as "n/a"."""
    if not isinstance(data, pd.DataFrame):
        raise ConfigError(f"{where}: data is a pandas DataFrame, got {type(data).__name__}")
    xy = _numeric(
        pd.DataFrame({"x": _column(data, x, where, "x"), "y": _column(data, y, where, "y")}),
        where,
    )
    labels = None
    if group is not None:
        column = _column(data, group, where, "group")
        labels = column.astype(object).where(column.notna(), "n/a").astype(str).to_numpy()
        found = len(set(labels))
        if found > MAX_SERIES:
            raise ConfigError(f"{where}: at most {MAX_SERIES} groups, column {group!r} has {found}")
    return Points(xy["x"].to_numpy(), xy["y"].to_numpy(), labels)


def frame_of(data: object, where: str) -> pd.DataFrame:
    """A DataFrame, as it is, its column names each its own."""
    if not isinstance(data, pd.DataFrame):
        raise ConfigError(f"{where}: data is a pandas DataFrame, got {type(data).__name__}")
    _unique(data, where)
    return data


def heat(data: object, where: str) -> pd.DataFrame:
    """A DataFrame of numbers, rows by columns."""
    return _numeric(frame_of(data, where), where)


def tile_values(data: object, where: str) -> dict[str, Any]:
    """A mapping (or Series) of label to a number or text."""
    if isinstance(data, pd.Series):
        data = data.to_dict()
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: data maps each tile's label to its value")
    for label, value in data.items():
        if isinstance(value, bool) or not isinstance(value, TILE_VALUE):
            raise ConfigError(
                f"{where}: tile {label!r} holds {type(value).__name__}; a tile shows a number "
                "or text"
            )
    return {str(label): value for label, value in data.items()}


def text_of(data: object, where: str) -> str:
    """Text, as it is."""
    if not isinstance(data, str):
        raise ConfigError(f"{where}: data is text, got {type(data).__name__}")
    return data


__all__ = [
    "Points",
    "called",
    "dated",
    "frame_of",
    "heat",
    "numbers",
    "points",
    "series_frame",
    "text_of",
    "tile_values",
]
