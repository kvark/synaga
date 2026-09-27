//! Compiles the shaders in `src/shaders/` to Naga modules before the crate is.
//!
//! Every resource says where it binds, which is what the defaults expect. A
//! host that assigns bindings itself, as Blade does, would write its
//! resources `= binding()` and say `.bindings(Bindings::Host)` here.

fn main() {
    synaga::build::Shaders::new().run();
}
