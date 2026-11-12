//! A small JSON reader for test fixtures: the crate has no dependencies, and the risk
//! vectors are JSON. It reads objects, arrays, strings with the common escapes, numbers,
//! booleans and null; anything else is an error naming the byte offset.

use std::collections::BTreeMap;

/// A JSON value. Objects keep their keys sorted, which no fixture depends on.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Json {
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    List(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

impl Json {
    /// The member `key` of an object; `Null` when absent or not an object.
    pub(super) fn get(&self, key: &str) -> &Json {
        match self {
            Json::Object(members) => members.get(key).unwrap_or(&Json::Null),
            _ => &Json::Null,
        }
    }

    pub(super) fn number(&self) -> Option<f64> {
        match self {
            Json::Number(value) => Some(*value),
            _ => None,
        }
    }

    pub(super) fn text(&self) -> Option<&str> {
        match self {
            Json::Text(value) => Some(value),
            _ => None,
        }
    }

    pub(super) fn boolean(&self) -> Option<bool> {
        match self {
            Json::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub(super) fn list(&self) -> &[Json] {
        match self {
            Json::List(items) => items,
            _ => &[],
        }
    }
}

/// Parse a whole document.
pub(super) fn parse(text: &str) -> Result<Json, String> {
    let mut reader = Reader {
        bytes: text.as_bytes(),
        at: 0,
    };
    let value = reader.value()?;
    reader.blank();
    if reader.at == reader.bytes.len() {
        Ok(value)
    } else {
        Err(format!("trailing text at byte {}", reader.at))
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn blank(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.at += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        self.blank();
        if self.peek() == Some(byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", byte as char, self.at))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.blank();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(Json::Text),
            Some(b't') => self.word("true", Json::Bool(true)),
            Some(b'f') => self.word("false", Json::Bool(false)),
            Some(b'n') => self.word("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(format!("unexpected byte at {}", self.at)),
        }
    }

    fn word(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(format!("expected {word} at byte {}", self.at))
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        while matches!(
            self.peek(),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.at += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).map_err(|e| e.to_string())?;
        text.parse::<f64>()
            .map(Json::Number)
            .map_err(|_| format!("bad number {text} at byte {start}"))
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            let byte = self.peek().ok_or("unterminated string")?;
            self.at += 1;
            match byte {
                b'"' => break,
                b'\\' => {
                    let escaped = self.peek().ok_or("unterminated escape")?;
                    self.at += 1;
                    out.push(match escaped {
                        b'n' => b'\n',
                        b't' => b'\t',
                        b'r' => b'\r',
                        b'"' | b'\\' | b'/' => escaped,
                        _ => return Err(format!("unsupported escape at byte {}", self.at)),
                    });
                }
                _ => out.push(byte),
            }
        }
        String::from_utf8(out).map_err(|e| e.to_string())
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.blank();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::List(items));
        }
        loop {
            items.push(self.value()?);
            self.blank();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::List(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.at)),
            }
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut members = BTreeMap::new();
        self.blank();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.blank();
            let key = self.string()?;
            self.expect(b':')?;
            members.insert(key, self.value()?);
            self.blank();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(members));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.at)),
            }
        }
    }
}

#[test]
fn the_reader_reads_every_kind_of_value() {
    let parsed = parse(r#"{"a": [1, -2.5e1, true, null], "b": "x\"y"}"#);
    let mut members = BTreeMap::new();
    members.insert(
        "a".to_string(),
        Json::List(vec![
            Json::Number(1.0),
            Json::Number(-25.0),
            Json::Bool(true),
            Json::Null,
        ]),
    );
    members.insert("b".to_string(), Json::Text("x\"y".to_string()));
    assert_eq!(parsed, Ok(Json::Object(members)));
    assert!(parse("[1, 2").is_err());
}
