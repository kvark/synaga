//! The shaders are compiled twice: `rustc` checks `src/shaders/` as ordinary
//! Rust, and `build.rs` reads the same files and serializes a Naga module.

mod shaders;

mod ir {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

fn main() {
    for (name, bytes) in ir::ALL {
        println!("--- {name} ({} bytes) ---", bytes.len());
        println!("{}", String::from_utf8_lossy(bytes));
    }
}
