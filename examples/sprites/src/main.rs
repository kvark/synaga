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
        // What a host builds its bind group layouts from. The numbers are the
        // shaders' own, `shaders::common::FRAME` included.
        let bindings: Vec<String> = module
            .global_variables
            .iter()
            .filter_map(|(_, var)| {
                let at = var.binding.as_ref()?;
                Some(format!(
                    "`{}` at {}.{}",
                    var.name.as_deref()?,
                    at.group,
                    at.binding
                ))
            })
            .collect();
        println!(
            "{name}: {} bytes, {}; binds {}",
            ir.bytes().len(),
            entry_points.join(", "),
            bindings.join(", ")
        );
    }
}
