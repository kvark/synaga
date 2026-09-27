//! Compiles the shaders in `src/shaders/` to Naga modules before the crate is.

fn main() {
    synaga::build::Shaders::new()
        // Bindings are assigned by the host at pipeline creation, so the
        // shaders leave `#[group]`/`#[binding]` off.
        .bindings(synaga::build::Bindings::Host)
        .run();
}
