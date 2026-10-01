//! Shared helpers for the integration tests.
//!
//! Every test wants one of three things: the WGSL a source lowers to, the fact
//! that it lowers to *something* Naga accepts, or the message it is rejected
//! with.

#![allow(dead_code)]

#[cfg(feature = "wgsl")]
use synaga::to_wgsl;
use synaga::{parse_str, validate, validate_unbound};

/// Lower, validate, and emit WGSL. Panics with the reason on any failure.
#[cfg(feature = "wgsl")]
pub fn roundtrip(src: &str) -> String {
    let module = parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    let info = validate(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
    to_wgsl(&module, &info).unwrap_or_else(|e| panic!("wgsl: {e}\n{src}"))
}

/// Like `roundtrip`, for a module whose `@group`/`@binding` the host assigns.
#[cfg(feature = "wgsl")]
pub fn roundtrip_unbound(src: &str) -> String {
    let module = parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    let info = validate_unbound(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
    to_wgsl(&module, &info).unwrap_or_else(|e| panic!("wgsl: {e}\n{src}"))
}

/// Lower and validate, ignoring the generated WGSL.
pub fn validate_only(src: &str) {
    let module = parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    validate(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
}

/// Like `validate_only`, for a module whose `@group`/`@binding` the host
/// assigns.
pub fn validate_only_unbound(src: &str) {
    let module = parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    validate_unbound(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
}

/// A module's debug dump with every byte range blanked out.
///
/// Two spellings of the same shader differ in length, so the spans their nodes
/// carry differ too. A span is a position in a file rather than a property of
/// the module, so tests that ask "do these two spellings mean the same thing"
/// compare the dump without them. Spans themselves are covered by the tests in
/// `tests/build.rs`, which read a module's spans back.
pub fn without_spans(dump: &str) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(dump.len());
    for line in dump.lines() {
        let trimmed = line.trim_start();
        let is_offset = ["start: ", "end: "]
            .iter()
            .find_map(|prefix| trimmed.strip_prefix(prefix))
            .is_some_and(|rest| rest.trim_end_matches(',').trim().parse::<u64>().is_ok());
        if is_offset {
            let indent = &line[..line.len() - trimmed.len()];
            let name = if trimmed.starts_with("start") {
                "start"
            } else {
                "end"
            };
            let _ = writeln!(out, "{indent}{name}: 0,");
        } else {
            let _ = writeln!(out, "{line}");
        }
    }
    out
}

/// The message `src` is rejected with. Panics if it is accepted instead.
pub fn reject(src: &str) -> String {
    match parse_str(src) {
        Ok(_) => panic!("expected an error, but this was accepted:\n{src}"),
        Err(e) => e.to_string(),
    }
}
