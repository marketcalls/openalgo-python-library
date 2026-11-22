//! Tests of the fund (18-multi-strategy KR5 to KR9), one behaviour per test, expected
//! values worked by hand in each module's docs: the equality of an engine in a fund with
//! its run alone (`alone`), contention for the account (`contention`), the account's
//! daily loss limit (`breach`) on the round's session day across exchanges
//! (`session_day`), the account's daily profit target and trade cap (`daily_limits`), the
//! merged time line (`time_line`) and what the fund refuses (`errors`). The account rule's
//! shared table is run in `rules`.

mod kit;

mod alone;
mod breach;
mod contention;
mod daily_limits;
mod errors;
mod session_day;
mod time_line;
