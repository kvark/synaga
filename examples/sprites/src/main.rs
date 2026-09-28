//! The shaders are compiled twice: `rustc` checks `src/shaders/` as ordinary
//! Rust, and `build.rs` reads the same files and writes a Naga module for each.

mod shaders;

mod shader_ir {
    synaga_shader::include_ir!();
}

use synaga_shader::vec2;

fn main() {
    // The host fills in the shaders' own structs and uploads their bytes.
    let mut sprite = shaders::sprite::Locals {
        position: [10.0, 20.0].into(),
        velocity: vec2(30.0, 0.0),
        color: 0xFF00_FFFF,
        _pad: 0,
    };
    sprite.position.x += sprite.velocity.x * 0.5;
    let uploads = [
        ("locals", bytemuck::bytes_of(&sprite).len()),
        ("globals", size_of::<shaders::common::Globals>()),
    ];

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
        // What the host uploads is as long as what the shader reads.
        for (global, size) in uploads {
            let Some((_, var)) = module
                .global_variables
                .iter()
                .find(|(_, var)| var.name.as_deref() == Some(global))
            else {
                continue;
            };
            if let naga::TypeInner::Struct { span, .. } = module.types[var.ty].inner {
                assert_eq!(size, span as usize, "`{global}` in {name}");
            }
        }
    }
    println!("uploaded a sprite at {:?}", sprite.position);
}
