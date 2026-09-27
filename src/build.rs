//! Build-script support: shader modules as part of the crate, compiled to
//! Naga modules before the crate that uses them is.
//!
//! The shape this is for:
//!
//! ```text
//! src/
//!   main.rs
//!   shaders/
//!     mod.rs         <- `pub mod common; pub mod bunnymark;`
//!     common.rs      <- helpers
//!     bunnymark.rs   <- `use super::common::*;` and an entry point or two
//! build.rs
//! ```
//!
//! ```ignore
//! // build.rs
//! fn main() {
//!     synaga::build::Shaders::new().run();
//! }
//! ```
//!
//! ```ignore
//! // src/main.rs
//! mod shaders; // `rustc` type-checks the sources
//! mod shader_ir {
//!     synaga_shader::include_ir!(); // and this is what they compile to
//! }
//!
//! let module: naga::Module = shader_ir::BUNNYMARK.decode().unwrap();
//! ```
//!
//! # What gets compiled
//!
//! Every file in the directory with an entry point is a shader, and becomes
//! one module. The other files are what those use: a shader's `use
//! super::common::*` or `common::luminance(..)` is what brings `common.rs`
//! along, the same way it tells `rustc` where to look. A file no shader
//! reaches is left alone. `mod.rs` lists the modules for `rustc` and is not
//! read here.
//!
//! What an entry point does not reach is pruned from its module, so a helper
//! file can hold everything the shaders need between them.
//!
//! `#[cfg(...)]` and `cfg!(...)` are settled the way `rustc` settles them for
//! the crate being built: `cfg!(debug_assertions)` holds in a debug build.
//!
//! The modules are bincode, behind a short header; see `synaga_shader::ir`.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use syn::visit::Visit;

use crate::{Cfg, Error, Source};

/// The major version of the Naga this crate builds against, recorded in each
/// module's header. Kept in step with `Cargo.toml` by a test.
pub const NAGA_MAJOR: u8 = 30;

/// The first bytes of each module: `SYNAGA`, the format, and the Naga major
/// version. `synaga_shader::ir` reads the same.
const HEADER: [u8; 8] = [b'S', b'Y', b'N', b'A', b'G', b'A', 1, NAGA_MAJOR];

/// Who assigns `@group` and `@binding`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Bindings {
    /// The shader says where each resource binds, as a WGSL shader does: it
    /// is initialised with `group(0).binding(1)`. A resource that does not
    /// say is an error.
    #[default]
    Explicit,
    /// The host assigns them at pipeline creation, matching globals up by
    /// name — what [Blade](https://github.com/kvark/blade) does. Every
    /// resource is initialised with `binding()`, one that says where it binds
    /// is an error, and validation is told not to ask.
    Host,
}

/// What went wrong, and where.
#[derive(Debug)]
pub struct BuildError {
    /// The source file the failure belongs to, if it belongs to one.
    pub path: Option<PathBuf>,
    pub kind: BuildErrorKind,
}

#[derive(Debug)]
pub enum BuildErrorKind {
    Io(std::io::Error),
    /// The source is not in the dialect.
    Transpile(Error),
    /// It is in the dialect, but the module it builds is not a valid shader.
    Validate(crate::ValidationError),
    /// The module is fine but cannot be written out.
    Emit(String),
    /// `OUT_DIR` was not set, so this is not running under Cargo.
    NotABuildScript,
}

impl std::fmt::Display for BuildError {
    /// Formatted as `path:line:column: message`, which is the shape Cargo and
    /// editors already know how to turn into a jump.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(path) = &self.path {
            write!(f, "{}", path.display())?;
            if let BuildErrorKind::Transpile(err) = &self.kind {
                if let Some((line, column)) = err.location() {
                    write!(f, ":{line}:{column}")?;
                }
            }
            write!(f, ": ")?;
        }
        match &self.kind {
            BuildErrorKind::Io(err) => write!(f, "{err}"),
            // The position prefix already said which line, so the item's own
            // wrapper is unwrapped here to keep from saying it twice.
            BuildErrorKind::Transpile(Error::At { item, source, .. }) => {
                write!(f, "{item}: {source}")
            }
            BuildErrorKind::Transpile(err) => write!(f, "{err}"),
            BuildErrorKind::Validate(err) => write!(f, "{err}"),
            BuildErrorKind::Emit(msg) => write!(f, "{msg}"),
            BuildErrorKind::NotABuildScript => {
                write!(f, "OUT_DIR is not set; this belongs in a build script")
            }
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            BuildErrorKind::Io(err) => Some(err),
            BuildErrorKind::Transpile(err) => Some(err),
            BuildErrorKind::Validate(err) => Some(err),
            _ => None,
        }
    }
}

/// One compiled shader.
#[derive(Debug)]
pub struct Shader {
    /// The file with its entry points.
    pub source_path: PathBuf,
    /// Every file that went into it, that one first.
    pub sources: Vec<PathBuf>,
    /// Where the serialized module was written.
    pub output_path: PathBuf,
    /// The constant it is reachable through, e.g. `BUNNYMARK`.
    pub constant: String,
    /// The module's name, which is its file stem, e.g. `bunnymark`.
    pub name: String,
    /// What each entry point is called. The module keeps these names.
    pub entry_points: Vec<EntryPoint>,
}

/// An entry point's name.
///
/// The serialized module keeps the name from the source, so this is also what
/// a pipeline asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPoint {
    /// What the shader calls it.
    pub name: String,
}

/// Compiles a directory of shader modules during a build script.
#[derive(Debug)]
pub struct Shaders {
    dir: PathBuf,
    module_name: String,
    prune: bool,
    bindings: Bindings,
    capabilities: naga::valid::Capabilities,
    cfg: Option<Cfg>,
}

impl Default for Shaders {
    fn default() -> Self {
        Self::new()
    }
}

impl Shaders {
    /// Compile `src/shaders`, with explicit bindings and no extra
    /// capabilities.
    pub fn new() -> Self {
        Self {
            dir: PathBuf::from("src/shaders"),
            module_name: "shaders.rs".into(),
            prune: true,
            bindings: Bindings::Explicit,
            capabilities: naga::valid::Capabilities::empty(),
            cfg: None,
        }
    }

    /// Where the shader modules live. Relative paths are taken from the
    /// crate root, as Cargo runs a build script there.
    pub fn dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.dir = dir.as_ref().to_path_buf();
        self
    }

    /// Drop declarations no entry point reaches. On by default.
    ///
    /// This is what keeps a helper file from putting an unused uniform in
    /// every shader — which is not merely untidy, since a host that binds
    /// resources by name has to find something to bind it to.
    pub fn prune(mut self, prune: bool) -> Self {
        self.prune = prune;
        self
    }

    /// Name of the generated file in `OUT_DIR`. Defaults to `shaders.rs`.
    pub fn module_name(mut self, name: impl Into<String>) -> Self {
        self.module_name = name.into();
        self
    }

    /// Who assigns `@group` and `@binding`. See [`Bindings`].
    pub fn bindings(mut self, bindings: Bindings) -> Self {
        self.bindings = bindings;
        self
    }

    /// Naga capabilities to validate against, for a shader that needs one the
    /// defaults leave off — `RAY_QUERY`, the binding-array capabilities, and
    /// so on.
    pub fn capabilities(mut self, capabilities: naga::valid::Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Which `cfg` predicates hold. Left alone, it is what Cargo told the
    /// build script about the crate being built; see [`Cfg::from_cargo_env`].
    pub fn cfg(mut self, cfg: Cfg) -> Self {
        self.cfg = Some(cfg);
        self
    }

    /// Compile, or print the reason and exit the build script.
    ///
    /// Failures reach Cargo as `cargo::error=` lines, so they are reported
    /// where a compile error would be rather than buried in build output.
    pub fn run(self) -> Vec<Shader> {
        match self.emit() {
            Ok(shaders) => shaders,
            Err(err) => {
                // One line per `cargo::error=`; the directive has no escape for
                // a newline.
                for line in err.to_string().lines() {
                    println!("cargo::error={line}");
                }
                std::process::exit(1);
            }
        }
    }

    /// Compile, handing back what was written.
    pub fn emit(self) -> Result<Vec<Shader>, BuildError> {
        let out_dir = std::env::var_os("OUT_DIR").ok_or(BuildError {
            path: None,
            kind: BuildErrorKind::NotABuildScript,
        })?;
        self.emit_to(Path::new(&out_dir))
    }

    /// Compile into `out_dir`, without needing Cargo's environment.
    ///
    /// `emit` is what a build script wants; this is for testing the same path
    /// without one.
    pub fn emit_to(&self, out_dir: &Path) -> Result<Vec<Shader>, BuildError> {
        let cfg = self.cfg.clone().unwrap_or_else(Cfg::from_cargo_env);

        // Anything added, removed or edited in the directory changes the
        // output, so the directory itself is watched as well as each file.
        println!("cargo::rerun-if-changed={}", self.dir.display());

        let files = self.read_files()?;
        let stems: Vec<&str> = files.iter().map(|f| f.stem.as_str()).collect();
        let mut shaders = Vec::new();
        let mut generated = String::from(
            "// Generated by synaga. Do not edit.\n\
             //\n\
             // Each constant is a serialized Naga module; see `synaga_shader::ir`.\n",
        );

        for (index, file) in files.iter().enumerate() {
            if !file.has_entry_point {
                continue;
            }
            let sources = reachable(&files, &stems, index);
            let (bytes, entry_points) = self.compile(&files, &sources, &cfg)?;

            let output_path = out_dir.join(format!("{}.naga", file.stem));
            std::fs::write(&output_path, &bytes).map_err(|e| BuildError {
                path: Some(output_path.clone()),
                kind: BuildErrorKind::Io(e),
            })?;

            let constant = constant_name(&file.stem);
            let _ = writeln!(
                generated,
                "\n/// Naga module compiled from `{}`.\n\
                 pub const {constant}: ::synaga_shader::ir::Ir = \
                 ::synaga_shader::ir::Ir::new(include_bytes!({:?}));",
                file.path.display(),
                output_path.display().to_string(),
            );
            shaders.push(Shader {
                source_path: file.path.clone(),
                sources: sources.iter().map(|&i| files[i].path.clone()).collect(),
                output_path,
                constant,
                name: file.stem.clone(),
                entry_points,
            });
        }

        // A host that hands every shader to the same place should not have to
        // repeat the list; it is exactly what the directory already said.
        let _ = writeln!(
            generated,
            "\n/// Every shader here, as `(module name, module)`.\n\
             #[allow(dead_code)]\n\
             pub const ALL: [(&str, ::synaga_shader::ir::Ir); {}] = [{}];",
            shaders.len(),
            shaders
                .iter()
                .map(|s| format!("({:?}, {}), ", s.name, s.constant))
                .collect::<String>(),
        );

        let module_path = out_dir.join(&self.module_name);
        std::fs::write(&module_path, generated).map_err(|e| BuildError {
            path: Some(module_path.clone()),
            kind: BuildErrorKind::Io(e),
        })?;
        Ok(shaders)
    }

    /// Every `.rs` file in the directory but `mod.rs`, read and looked over,
    /// in a stable order.
    fn read_files(&self) -> Result<Vec<File>, BuildError> {
        let dir_error = |e: std::io::Error| BuildError {
            path: Some(self.dir.clone()),
            kind: BuildErrorKind::Io(e),
        };
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&self.dir)
            .map_err(dir_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(dir_error)?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|e| e == "rs"))
            // `mod.rs` lists the shader modules for Rust; it is not one.
            .filter(|path| path.file_name().is_some_and(|n| n != "mod.rs"))
            .collect();
        // Directory order is arbitrary; the generated file should not be.
        paths.sort();

        let mut files = Vec::with_capacity(paths.len());
        for path in paths {
            println!("cargo::rerun-if-changed={}", path.display());
            let text = std::fs::read_to_string(&path).map_err(|e| BuildError {
                path: Some(path.clone()),
                kind: BuildErrorKind::Io(e),
            })?;
            let syntax: syn::File = syn::parse_str(&text).map_err(|e| BuildError {
                path: Some(path.clone()),
                kind: BuildErrorKind::Transpile(e.into()),
            })?;
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let mut names = Mentions::default();
            names.visit_file(&syntax);
            files.push(File {
                has_entry_point: has_entry_point(&syntax),
                mentions: names.0,
                path,
                stem,
                text,
            });
        }
        Ok(files)
    }

    /// The serialized module, and what each entry point is called.
    fn compile(
        &self,
        files: &[File],
        sources: &[usize],
        cfg: &Cfg,
    ) -> Result<(Vec<u8>, Vec<EntryPoint>), BuildError> {
        let root = &files[sources[0]].path;
        let at = |kind| BuildError {
            path: Some(root.clone()),
            kind,
        };
        let texts: Vec<Source> = sources
            .iter()
            .map(|&i| Source {
                name: Some(&files[i].stem),
                text: &files[i].text,
            })
            .collect();
        let mut module =
            crate::parse_expecting(&texts, cfg, Some(self.bindings)).map_err(|err| BuildError {
                // Blame the file the failure is in, which need not be the shader.
                path: Some(files[sources[err.index]].path.clone()),
                kind: BuildErrorKind::Transpile(err.error),
            })?;

        let flags = match self.bindings {
            Bindings::Explicit => naga::valid::ValidationFlags::all(),
            Bindings::Host => {
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS
            }
        };
        crate::validate_with(&module, flags, self.capabilities)
            .map_err(|err| at(BuildErrorKind::Validate(err)))?;

        // Pruning needs a module already known to be valid. It drops the
        // validation info, so a pruned module is checked again.
        if self.prune {
            naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
            crate::validate_with(&module, flags, self.capabilities)
                .map_err(|err| at(BuildErrorKind::Validate(err)))?;
        }

        let mut bytes = HEADER.to_vec();
        bincode::serde::encode_into_std_write(&module, &mut bytes, bincode::config::standard())
            .map_err(|err| at(BuildErrorKind::Emit(err.to_string())))?;
        Ok((bytes, entry_points(&module)))
    }
}

/// A source file in the shader directory.
struct File {
    path: PathBuf,
    stem: String,
    text: String,
    has_entry_point: bool,
    /// Every name the file uses as a module: the heads of its paths and the
    /// segments of its `use` trees.
    mentions: BTreeSet<String>,
}

/// File `root`, then every sibling it reaches through its paths, and every
/// sibling those reach, in the order they were found.
fn reachable(files: &[File], stems: &[&str], root: usize) -> Vec<usize> {
    let mut order = vec![root];
    let mut next = 0;
    while next < order.len() {
        let file = &files[order[next]];
        next += 1;
        for name in &file.mentions {
            if let Some(found) = stems.iter().position(|stem| stem == name) {
                if !order.contains(&found) {
                    order.push(found);
                }
            }
        }
    }
    order
}

/// Does the file have a function marked `#[entry_point]`?
fn has_entry_point(file: &syn::File) -> bool {
    file.items.iter().any(|item| match item {
        syn::Item::Fn(func) => func
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("entry_point")),
        _ => false,
    })
}

/// Collects the names a file reaches other modules through. `a::b::c` uses
/// `a` and `b` as modules; `c` is the item. `use` trees count every segment,
/// since `use super::common;` names a module in its last one.
#[derive(Default)]
struct Mentions(BTreeSet<String>);

impl<'ast> Visit<'ast> for Mentions {
    fn visit_path(&mut self, path: &'ast syn::Path) {
        let segments = &path.segments;
        for segment in segments.iter().take(segments.len().saturating_sub(1)) {
            self.0.insert(segment.ident.to_string());
        }
        syn::visit::visit_path(self, path);
    }

    fn visit_use_tree(&mut self, tree: &'ast syn::UseTree) {
        match tree {
            syn::UseTree::Path(path) => {
                self.0.insert(path.ident.to_string());
            }
            syn::UseTree::Name(name) => {
                self.0.insert(name.ident.to_string());
            }
            syn::UseTree::Rename(rename) => {
                self.0.insert(rename.ident.to_string());
            }
            syn::UseTree::Glob(_) | syn::UseTree::Group(_) => {}
        }
        syn::visit::visit_use_tree(self, tree);
    }
}

/// What each entry point is called.
///
/// The serialized module keeps the name from the source. A pipeline asks for
/// that name.
pub fn entry_point_names(module: &naga::Module) -> Vec<EntryPoint> {
    entry_points(module)
}

fn entry_points(module: &naga::Module) -> Vec<EntryPoint> {
    module
        .entry_points
        .iter()
        .map(|entry| EntryPoint {
            name: entry.name.clone(),
        })
        .collect()
}

/// `bunnymark` -> `BUNNYMARK`, `post_proc` -> `POST_PROC`.
fn constant_name(stem: &str) -> String {
    let mut name: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    // A constant cannot open with a digit.
    if name.starts_with(|c: char| c.is_ascii_digit()) {
        name.insert(0, '_');
    }
    name
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_recorded_naga_version_is_the_one_depended_on() {
        let manifest = include_str!("../Cargo.toml");
        let wanted = format!("naga = {{ version = \"{}\"", super::NAGA_MAJOR);
        assert!(manifest.contains(&wanted), "update NAGA_MAJOR");
    }
}
