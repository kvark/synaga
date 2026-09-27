//! The shaders are compiled twice: `rustc` checks `src/shaders/` as ordinary
//! Rust, and `build.rs` reads the same files and writes a Naga module for each.

mod shaders;

mod shader_ir {
    synaga_shader::include_ir!();
}

fn main() {
    for (name, ir) in shader_ir::ALL {
        let module: naga::Module = ir.decode().unwrap_or_else(|err| panic!("{name}: {err}"));
        let entry_points: Vec<String> = module
            .entry_points
            .iter()
            .map(|entry| format!("{:?} `{}`", entry.stage, entry.name))
            .collect();
        println!(
            "{name}: {} bytes, {}",
            ir.bytes().len(),
            entry_points.join(", ")
        );
    }
}
