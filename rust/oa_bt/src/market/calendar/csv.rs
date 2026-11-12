//! A reader for the small comma-separated tables embedded in the calendar.
//!
//! Format: one record per line; a line that starts with `#` is a comment and blank
//! lines are skipped; the first remaining line is the header and must match the
//! expected one exactly; fields are separated by commas; a field may be wrapped in
//! double quotes, inside which a comma is text and a doubled quote is one quote.

use std::fmt;

/// A problem in one of the embedded tables: which table, which line, and what.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableError {
    /// File name of the table, for example "regular_hours.csv".
    pub table: &'static str,
    /// Line number in the file, from 1; 0 when the problem is not tied to a line.
    pub line: usize,
    pub problem: String,
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} line {}: {}. Fix the row in the table file; it is checked when the \
             calendar is first used",
            self.table, self.line, self.problem
        )
    }
}

impl std::error::Error for TableError {}

/// One data line of a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    /// Line number in the file, from 1.
    pub(crate) line: usize,
    pub(crate) fields: Vec<String>,
}

/// Split one line into fields.
fn split_fields(line: &str) -> Result<Vec<String>, String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match (ch, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                chars.next();
                field.push('"');
            }
            ('"', true) => quoted = false,
            ('"', false) if field.is_empty() => quoted = true,
            ('"', false) => return Err("a quote inside an unquoted field".to_string()),
            (',', false) => fields.push(std::mem::take(&mut field)),
            (other, _) => field.push(other),
        }
    }
    if quoted {
        return Err("a quoted field is not closed".to_string());
    }
    fields.push(field);
    Ok(fields)
}

/// Read the data records of a table whose header must be `header`.
///
/// Errors: a missing or different header, an unbalanced quote, or a record whose
/// number of fields differs from the header's.
pub(crate) fn read(
    table: &'static str,
    text: &str,
    header: &[&str],
) -> Result<Vec<Record>, TableError> {
    let fail = |line: usize, problem: String| TableError {
        table,
        line,
        problem,
    };
    let mut records = Vec::new();
    let mut header_seen = false;
    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        let content = raw.trim();
        if content.is_empty() || content.starts_with('#') {
            continue;
        }
        let fields = split_fields(content).map_err(|problem| fail(line, problem))?;
        if !header_seen {
            if fields != header {
                return Err(fail(
                    line,
                    format!("the header must be {}", header.join(",")),
                ));
            }
            header_seen = true;
            continue;
        }
        if fields.len() != header.len() {
            let problem = format!(
                "{}, expected {}",
                crate::model::counted(fields.len(), "field", "fields"),
                header.len()
            );
            return Err(fail(line, problem));
        }
        records.push(Record { line, fields });
    }
    if header_seen {
        Ok(records)
    } else {
        Err(fail(0, "the table has no header line".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: [&str; 3] = ["name", "value", "note"];

    fn fields(text: &str) -> Vec<Vec<String>> {
        let records = read("test.csv", text, &HEADER).unwrap();
        records.into_iter().map(|record| record.fields).collect()
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let text = "# a comment\n\nname,value,note\n# another\na,1,x\n";
        assert_eq!(fields(text), vec![vec!["a", "1", "x"]]);
    }

    #[test]
    fn a_quoted_field_keeps_commas_and_doubled_quotes() {
        let text = "name,value,note\na,1,\"see page 2, \"\"Timings\"\"\"\n";
        assert_eq!(
            fields(text),
            vec![vec!["a", "1", "see page 2, \"Timings\""]]
        );
    }

    #[test]
    fn empty_fields_are_kept() {
        assert_eq!(fields("name,value,note\na,,\n"), vec![vec!["a", "", ""]]);
    }

    #[test]
    fn records_carry_their_line_number() {
        let records = read("test.csv", "# c\nname,value,note\n\na,1,x\n", &HEADER).unwrap();
        assert_eq!(records[0].line, 4);
    }

    #[test]
    fn a_wrong_header_is_an_error() {
        let error = read("test.csv", "name,value\n", &HEADER).unwrap_err();
        assert_eq!((error.table, error.line), ("test.csv", 1));
        assert!(
            error.problem.contains("name,value,note"),
            "{}",
            error.problem
        );
    }

    #[test]
    fn a_record_with_the_wrong_number_of_fields_is_an_error() {
        let error = read("test.csv", "name,value,note\na,1\n", &HEADER).unwrap_err();
        assert_eq!(error.line, 2);
        assert_eq!(error.problem, "2 fields, expected 3");
    }

    #[test]
    fn an_unclosed_quote_is_an_error() {
        let error = read("test.csv", "name,value,note\na,1,\"open\n", &HEADER).unwrap_err();
        assert_eq!(error.problem, "a quoted field is not closed");
    }

    #[test]
    fn a_table_without_a_header_is_an_error() {
        let error = read("test.csv", "# only a comment\n", &HEADER).unwrap_err();
        assert_eq!(error.line, 0);
    }
}
