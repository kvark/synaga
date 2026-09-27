//! Shared helpers for the integration tests.
//!
//! Every test wants one of three things: the WGSL a source lowers to, the fact
//! that it lowers to *something* Naga accepts, or the message it is rejected
//! with.

#![allow(dead_code)]

use synaga::{parse_str, to_wgsl, validate, validate_unbound};

/// Lower, validate, and emit WGSL. Panics with the reason on any failure.
pub fn roundtrip(src: &str) -> String {
    let module = parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    let info = validate(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
    to_wgsl(&module, &info).unwrap_or_else(|e| panic!("wgsl: {e}\n{src}"))
}

/// Like `roundtrip`, for a module whose `@group`/`@binding` the host assigns.
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

/// The message `src` is rejected with. Panics if it is accepted instead.
pub fn reject(src: &str) -> String {
    match parse_str(src) {
        Ok(_) => panic!("expected an error, but this was accepted:\n{src}"),
        Err(e) => e.to_string(),
    }
}
