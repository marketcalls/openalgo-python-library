//! The plain data of a live session's saved state (`venue_state`): a tree of null,
//! booleans, whole numbers, numbers, text, lists and maps with text keys. These are the
//! shapes a JSON document holds, so a state written as JSON and read back is the same
//! state: the bindings turn the tree into Python's `None`, `bool`, `int`, `float`, `str`,
//! `list` and `dict` and back, and nothing in it is a tuple or a number that is not
//! finite.
//!
//! `Read` reads a saved tree with the path of every value it reads, so a state that
//! cannot be restored is refused naming the field (`LiveStateError::Malformed`).

use super::error::LiveStateError;

/// One value of a saved state (module docs).
#[derive(Clone, Debug, PartialEq)]
pub enum StateValue {
    /// No value (JSON `null`).
    Null,
    /// A boolean.
    Bool(bool),
    /// A whole number: an order id, a slice index, UTC nanoseconds, a count.
    Int(i64),
    /// A number: rupees, units, points, a percent. Always finite.
    Float(f64),
    /// Text: a code such as "MIS" or "leg_stop", a symbol, a tag.
    Text(String),
    /// A list, in order.
    List(Vec<StateValue>),
    /// A map with text keys, in the order they were written.
    Map(Vec<(String, StateValue)>),
}

impl From<bool> for StateValue {
    fn from(value: bool) -> StateValue {
        StateValue::Bool(value)
    }
}

impl From<i64> for StateValue {
    fn from(value: i64) -> StateValue {
        StateValue::Int(value)
    }
}

impl From<i32> for StateValue {
    fn from(value: i32) -> StateValue {
        StateValue::Int(i64::from(value))
    }
}

impl From<u32> for StateValue {
    fn from(value: u32) -> StateValue {
        StateValue::Int(i64::from(value))
    }
}

impl From<u16> for StateValue {
    fn from(value: u16) -> StateValue {
        StateValue::Int(i64::from(value))
    }
}

impl From<u64> for StateValue {
    /// An order id. Ids count a session's orders and never reach `i64::MAX`, so the
    /// saturation never acts.
    fn from(value: u64) -> StateValue {
        StateValue::Int(i64::try_from(value).unwrap_or(i64::MAX))
    }
}

impl From<usize> for StateValue {
    /// A slice index of one day's grid, far below `i64::MAX`.
    fn from(value: usize) -> StateValue {
        StateValue::Int(i64::try_from(value).unwrap_or(i64::MAX))
    }
}

impl From<f64> for StateValue {
    fn from(value: f64) -> StateValue {
        StateValue::Float(value)
    }
}

impl From<&str> for StateValue {
    fn from(value: &str) -> StateValue {
        StateValue::Text(value.to_string())
    }
}

impl From<String> for StateValue {
    fn from(value: String) -> StateValue {
        StateValue::Text(value)
    }
}

impl<T: Into<StateValue>> From<Option<T>> for StateValue {
    fn from(value: Option<T>) -> StateValue {
        value.map_or(StateValue::Null, Into::into)
    }
}

impl<T: Into<StateValue>> From<Vec<T>> for StateValue {
    fn from(values: Vec<T>) -> StateValue {
        StateValue::List(values.into_iter().map(Into::into).collect())
    }
}

/// A map of `entries`, in order.
pub(super) fn map<const N: usize>(entries: [(&str, StateValue); N]) -> StateValue {
    StateValue::Map(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    )
}

/// A list of `items`, each written by `write`.
pub(super) fn list<T>(
    items: impl IntoIterator<Item = T>,
    write: impl Fn(T) -> StateValue,
) -> StateValue {
    StateValue::List(items.into_iter().map(write).collect())
}

/// A value of a saved tree being read, with its path for errors ("positions[0].tranches").
#[derive(Clone, Debug)]
pub(super) struct Read<'a> {
    value: &'a StateValue,
    path: String,
}

/// The root of a path into a same-day state: the call that returned it.
pub(super) const LIVE_ROOT: &str = "live_state";

/// The root of a path into a carried state: the call that returned it (`carry`).
pub(super) const CARRIED_ROOT: &str = "carried_state";

impl<'a> Read<'a> {
    /// The root of a saved state.
    pub(super) fn root(value: &'a StateValue) -> Read<'a> {
        Read {
            value,
            path: LIVE_ROOT.to_string(),
        }
    }

    /// The root of a carried state: its errors name `carried_state` (`carry`).
    pub(super) fn carried_root(value: &'a StateValue) -> Read<'a> {
        Read {
            value,
            path: CARRIED_ROOT.to_string(),
        }
    }

    /// The error for this value: `reason` says what is wrong with it.
    pub(super) fn malformed(&self, reason: impl Into<String>) -> LiveStateError {
        LiveStateError::Malformed {
            path: self.path.clone(),
            reason: reason.into(),
        }
    }

    fn entries(&self) -> Result<&'a [(String, StateValue)], LiveStateError> {
        match self.value {
            StateValue::Map(entries) => Ok(entries),
            _ => Err(self.malformed("is not a map")),
        }
    }

    /// The value of `key`, which must be present (it may be null).
    ///
    /// Errors: `Malformed` when this is not a map or `key` is missing.
    pub(super) fn field(&self, key: &str) -> Result<Read<'a>, LiveStateError> {
        let entries = self.entries()?;
        let found = entries.iter().find(|(known, _)| known == key);
        let Some((_, value)) = found else {
            return Err(self.malformed(format!("has no field {key:?}")));
        };
        Ok(Read {
            value,
            path: format!("{}.{key}", self.path),
        })
    }

    /// The value of `key`, `None` when it is null.
    ///
    /// Errors: those of `field`.
    pub(super) fn maybe(&self, key: &str) -> Result<Option<Read<'a>>, LiveStateError> {
        let read = self.field(key)?;
        Ok((*read.value != StateValue::Null).then_some(read))
    }

    /// True when this value is null.
    pub(super) fn is_null(&self) -> bool {
        *self.value == StateValue::Null
    }

    /// True when this value is the text `text`.
    pub(super) fn is_text(&self, text: &str) -> bool {
        matches!(self.value, StateValue::Text(known) if known == text)
    }

    /// The items of a list, in order.
    ///
    /// Errors: `Malformed` when this is not a list.
    pub(super) fn items(&self) -> Result<Vec<Read<'a>>, LiveStateError> {
        let StateValue::List(items) = self.value else {
            return Err(self.malformed("is not a list"));
        };
        Ok(items
            .iter()
            .enumerate()
            .map(|(index, value)| Read {
                value,
                path: format!("{}[{index}]", self.path),
            })
            .collect())
    }

    /// A finite number (a whole number is taken as one).
    ///
    /// Errors: `Malformed` for anything else.
    pub(super) fn float(&self) -> Result<f64, LiveStateError> {
        let number = match *self.value {
            StateValue::Float(number) => number,
            StateValue::Int(whole) => whole as f64,
            _ => return Err(self.malformed("is not a number")),
        };
        if !number.is_finite() {
            return Err(self.malformed("is not a finite number"));
        }
        Ok(number)
    }

    /// A whole number in the range of `T`.
    ///
    /// Errors: `Malformed` for anything else.
    pub(super) fn whole<T: TryFrom<i64>>(&self) -> Result<T, LiveStateError> {
        let StateValue::Int(whole) = *self.value else {
            return Err(self.malformed("is not a whole number"));
        };
        T::try_from(whole).map_err(|_| self.malformed(format!("{whole} is out of range")))
    }

    /// A boolean.
    ///
    /// Errors: `Malformed` for anything else.
    pub(super) fn boolean(&self) -> Result<bool, LiveStateError> {
        match *self.value {
            StateValue::Bool(value) => Ok(value),
            _ => Err(self.malformed("is not true or false")),
        }
    }

    /// Text.
    ///
    /// Errors: `Malformed` for anything else.
    pub(super) fn text(&self) -> Result<&'a str, LiveStateError> {
        match self.value {
            StateValue::Text(text) => Ok(text),
            _ => Err(self.malformed("is not text")),
        }
    }

    /// The item of `all` whose `code` this text is.
    ///
    /// Errors: `Malformed` for text that names none of them, or a value that is not text.
    pub(super) fn code<T: Copy>(
        &self,
        all: &[T],
        code: fn(T) -> &'static str,
    ) -> Result<T, LiveStateError> {
        let text = self.text()?;
        all.iter()
            .copied()
            .find(|item| code(*item) == text)
            .ok_or_else(|| {
                let codes: Vec<&str> = all.iter().map(|item| code(*item)).collect();
                self.malformed(format!("{text:?} is not one of {}", codes.join(", ")))
            })
    }

    /// The value `parse` reads from this text (a code such as "MIS" or "BUY").
    ///
    /// Errors: `Malformed` for text `parse` does not know, or a value that is not text.
    pub(super) fn parsed<T>(&self, parse: impl Fn(&str) -> Option<T>) -> Result<T, LiveStateError> {
        let text = self.text()?;
        parse(text).ok_or_else(|| self.malformed(format!("{text:?} is not a known code")))
    }

    /// The number of field `key`. Errors: those of `field` and `float`.
    pub(super) fn f64_of(&self, key: &str) -> Result<f64, LiveStateError> {
        self.field(key)?.float()
    }

    /// The number of field `key`, `None` when null. Errors: those of `field` and `float`.
    pub(super) fn opt_f64_of(&self, key: &str) -> Result<Option<f64>, LiveStateError> {
        self.maybe(key)?.map(|read| read.float()).transpose()
    }

    /// The whole number of field `key`. Errors: those of `field` and `whole`.
    pub(super) fn whole_of<T: TryFrom<i64>>(&self, key: &str) -> Result<T, LiveStateError> {
        self.field(key)?.whole()
    }

    /// The whole number of field `key`, `None` when null. Errors: those of `field` and
    /// `whole`.
    pub(super) fn opt_whole_of<T: TryFrom<i64>>(
        &self,
        key: &str,
    ) -> Result<Option<T>, LiveStateError> {
        self.maybe(key)?.map(|read| read.whole()).transpose()
    }

    /// The boolean of field `key`. Errors: those of `field` and `boolean`.
    pub(super) fn bool_of(&self, key: &str) -> Result<bool, LiveStateError> {
        self.field(key)?.boolean()
    }

    /// The items of the list in field `key`. Errors: those of `field` and `items`.
    pub(super) fn items_of(&self, key: &str) -> Result<Vec<Read<'a>>, LiveStateError> {
        self.field(key)?.items()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn saved() -> StateValue {
        map([
            ("units", 100.0.into()),
            ("order", 7u64.into()),
            ("mark", Option::<f64>::None.into()),
            ("tranches", list([1.5, 2.5], StateValue::from)),
        ])
    }

    #[test]
    fn a_field_is_read_with_its_type() {
        let value = saved();
        let root = Read::root(&value);
        assert_eq!(root.f64_of("units"), Ok(100.0));
        assert_eq!(root.whole_of::<u64>("order"), Ok(7));
        assert_eq!(root.opt_f64_of("mark"), Ok(None));
        let tranches: Vec<f64> = root
            .items_of("tranches")
            .and_then(|items| items.iter().map(Read::float).collect())
            .unwrap_or_default();
        assert_eq!(tranches, [1.5, 2.5]);
    }

    #[test]
    fn a_value_that_cannot_be_read_is_refused_naming_its_path() {
        let value = saved();
        let root = Read::root(&value);
        // (what, the error's path, its reason)
        let table = [
            (root.f64_of("price"), "live_state", "has no field \"price\""),
            (
                root.f64_of("tranches"),
                "live_state.tranches",
                "is not a number",
            ),
            (
                root.items_of("tranches")
                    .and_then(|items| items[1].whole::<u32>().map(f64::from)),
                "live_state.tranches[1]",
                "is not a whole number",
            ),
        ];
        for (read, path, reason) in table {
            let expected = LiveStateError::Malformed {
                path: path.to_string(),
                reason: reason.to_string(),
            };
            assert_eq!(read, Err(expected));
        }
    }

    #[test]
    fn a_number_that_is_not_finite_or_a_whole_number_out_of_range_is_refused() {
        let value = map([("nan", f64::NAN.into()), ("big", i64::MAX.into())]);
        let root = Read::root(&value);
        assert!(root.f64_of("nan").is_err());
        assert!(root.whole_of::<u32>("big").is_err());
        assert_eq!(root.whole_of::<i64>("big"), Ok(i64::MAX));
    }
}
