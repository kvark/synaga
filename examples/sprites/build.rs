//! Compiles the shaders in `src/shaders/` to Naga modules before the crate is.
//!
//! Every resource says where it binds, which is what the defaults expect. A
//! host that assigns bindings itself, as Blade does, would write its
//! resources `= binding()` and say `.bindings(Bindings::Host)` here.
//!
//! `src/shaders/mod.rs` has `synaga_shader::check_layout!()`, which is where
//! `rustc` checks this crate's own layout of the structs it shares with the
//! shaders. The build reads `mod.rs` for it, and warns if it goes missing.

fn main() {
    synaga::build::Shaders::new().run();
}
