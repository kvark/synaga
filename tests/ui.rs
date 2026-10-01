//! Compiles the compiletest fixtures and checks the diagnostics they produce.
//!
//! Most of the suite asserts on a `naga::Module` or on a message. These cover
//! what that cannot reach: the errors `rustc` itself prints, which are what a
//! host sees for a wrong `#[derive]` or a layout its structs do not agree on.
//!
//! The fixtures are plain source files rather than crates with build scripts,
//! which is all `trybuild` can compile — so the layout fixture spells out the
//! body the build script writes, and `tests/build.rs` checks that what the build
//! script writes has that shape.

#[test]
fn the_diagnostics_are_the_ones_we_wrote() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compiletests/*.rs");
}
