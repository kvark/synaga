//! Build-script support: shaders as ordinary-looking source files, turned into
//! WGSL before the crate that uses them is compiled.
//!
//! The shape this is for:
//!
//! ```text
//! src/
//!   main.rs
//!   shaders/
//!     common.rs      <- helpers shared by the rest
//!     bunnymark.rs   <- one shader module
//! build.rs
//! ```
//!
//! ```ignore
//! // build.rs — a build script really is a `fn main`
//! fn main() {
//!     synaga::build::Shaders::new().prelude("common.rs").run();
//! }
//! ```
//!
//! ```ignore
//! // src/main.rs
//! mod shaders {
//!     include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
//! }
//!
//! let wgsl: &str = shaders::BUNNYMARK;
//! ```
//!
//! # These files are not part of your crate
//!
//! A shader module mentions `vec3`, `texture_2d<f32>`, `#[vertex]` — none of
//! which are Rust. So the files must not be reachable from your crate root: no
//! `mod shaders;` next to `mod` declarations that *are* compiled. Cargo never
//! looks at a file nothing declares, so `src/shaders/` sitting there is fine,
//! and the `.rs` extension still buys syntax highlighting and brace matching.
//!
//! The cost is honest: `rustc` never sees these files, so they get no
//! borrow checking and no type inference beyond what this crate does, and
//! `cargo fmt` skips them for the same reason. If you would rather they not
//! look like crate sources at all, point [`Shaders::dir`] somewhere else —
//! `shaders/` beside `src/` reads more plainly.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::Error;

/// Who assigns `@group` and `@binding`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Bindings {
    /// The shader writes them itself, with `#[group]` and `#[binding]`.
    #[default]
    Explicit,
    /// The host assigns them at pipeline creation, matching globals up by
    /// name — what [Blade](https://github.com/kvark/blade) does. Globals are
    /// expected to carry no binding attributes, and validation is told not to
    /// ask for any.
    Host,
}

/// What went wrong, and where.
#[derive(Debug)]
pub struct BuildError {
    /// The shader module being compiled, if the failure belongs to one.
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
    /// The module is fine but cannot be written in the chosen output format.
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
    /// The source it came from.
    pub source_path: PathBuf,
    /// Where the WGSL was written.
    pub output_path: PathBuf,
    /// The constant it is reachable through, e.g. `BUNNYMARK`.
    pub constant: String,
    /// The module's name, which is its file stem, e.g. `bunnymark`.
    pub name: String,
    /// What each entry point is called, in the source and in the output.
    pub entry_points: Vec<EntryPoint>,
}

/// An entry point, and the name a pipeline has to ask for to get it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPoint {
    /// What the shader calls it.
    pub name: String,
    /// What the generated WGSL calls it, which is usually the same.
    pub emitted_name: String,
}

impl EntryPoint {
    /// Whether Naga had to give it a different name on the way out.
    pub fn renamed(&self) -> bool {
        self.name != self.emitted_name
    }
}

/// Compiles a directory of shader modules during a build script.
#[derive(Debug)]
pub struct Shaders {
    dir: PathBuf,
    prelude: Vec<PathBuf>,
    module_name: String,
    prune: bool,
    bindings: Bindings,
    capabilities: naga::valid::Capabilities,
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
            prelude: Vec::new(),
            module_name: "shaders.rs".into(),
            prune: true,
            bindings: Bindings::Explicit,
            capabilities: naga::valid::Capabilities::empty(),
        }
    }

    /// Where the shader modules live. Relative paths are taken from the
    /// crate root, as Cargo runs a build script there.
    pub fn dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.dir = dir.as_ref().to_path_buf();
        self
    }

    /// A file of shared declarations available to every shader module, in
    /// place of the `#include` a WGSL project would write. Relative to
    /// [`Shaders::dir`]. A file named here is not compiled on its own.
    ///
    /// What a module does not use is pruned from its output, so a prelude can
    /// hold everything the set of shaders needs between them without every
    /// shader carrying all of it. See [`Shaders::prune`].
    pub fn prelude(mut self, path: impl AsRef<Path>) -> Self {
        self.prelude.push(path.as_ref().to_path_buf());
        self
    }

    /// Drop declarations no entry point reaches. On by default.
    ///
    /// This is what keeps a prelude from putting an unused uniform in every
    /// shader — which is not merely untidy, since a host that binds resources
    /// by name has to find something to bind it to.
    ///
    /// It has no effect on a module with no entry points, where there is no
    /// root to measure "reached" from and pruning would empty the module.
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

    /// Compile, or print the reason and exit the build script.
    ///
    /// Failures reach Cargo as `cargo::error=` lines, so they are reported
    /// where a compile error would be rather than buried in build output.
    pub fn run(self) -> Vec<Shader> {
        match self.emit() {
            Ok(shaders) => {
                // Silently answering to a different name than the source says
                // is exactly the surprise this build step exists to avoid.
                for shader in &shaders {
                    for entry in shader.entry_points.iter().filter(|e| e.renamed()) {
                        println!(
                            "cargo::warning={}: entry point `{}` is `{}` in the generated \
                             WGSL, because Naga reserves names it may need to uniquify. \
                             Create the pipeline with `{}`, or rename it.",
                            shader.source_path.display(),
                            entry.name,
                            entry.emitted_name,
                            entry.emitted_name,
                        );
                    }
                }
                shaders
            }
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
        let at = |path: &Path| {
            let path = path.to_path_buf();
            move |e: std::io::Error| BuildError {
                path: Some(path.clone()),
                kind: BuildErrorKind::Io(e),
            }
        };

        // Anything added, removed or edited in the directory changes the
        // output, so the directory itself is watched as well as each file.
        println!("cargo::rerun-if-changed={}", self.dir.display());

        let mut prelude = Vec::new();
        for name in &self.prelude {
            let path = self.dir.join(name);
            println!("cargo::rerun-if-changed={}", path.display());
            let text = std::fs::read_to_string(&path).map_err(at(&path))?;
            prelude.push((path, text));
        }

        let mut shaders = Vec::new();
        let mut generated = String::from(
            "// Generated by synaga. Do not edit.\n\
             //\n\
             // Each constant is the WGSL for the shader module it is named after.\n",
        );

        for source_path in self.sources()? {
            println!("cargo::rerun-if-changed={}", source_path.display());
            let source = std::fs::read_to_string(&source_path).map_err(at(&source_path))?;
            let (wgsl, entry_points) = self.compile(&source_path, &prelude, &source)?;

            let stem = source_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let output_path = out_dir.join(format!("{stem}.wgsl"));
            std::fs::write(&output_path, &wgsl).map_err(at(&output_path))?;

            let constant = constant_name(&stem);
            let _ = writeln!(
                generated,
                "\n/// WGSL generated from `{}`.\n\
                 pub const {constant}: &str = include_str!({:?});",
                source_path.display(),
                output_path.display().to_string(),
            );
            shaders.push(Shader {
                source_path,
                output_path,
                constant,
                name: stem,
                entry_points,
            });
        }

        // A host that hands every shader to the same place should not have to
        // repeat the list; it is exactly what the directory already said.
        let _ = writeln!(
            generated,
            "\n/// Every shader here, as `(module name, WGSL)`.\n\
             #[allow(dead_code)]\n\
             pub const ALL: [(&str, &str); {}] = [{}];",
            shaders.len(),
            shaders
                .iter()
                .map(|s| format!("({:?}, {}), ", s.name, s.constant))
                .collect::<String>(),
        );

        let module_path = out_dir.join(&self.module_name);
        std::fs::write(&module_path, generated).map_err(at(&module_path))?;
        Ok(shaders)
    }

    /// Shader modules in the directory, in a stable order, minus the prelude.
    fn sources(&self) -> Result<Vec<PathBuf>, BuildError> {
        let dir_error = |e: std::io::Error| BuildError {
            path: Some(self.dir.clone()),
            kind: BuildErrorKind::Io(e),
        };
        let prelude: Vec<PathBuf> = self.prelude.iter().map(|p| self.dir.join(p)).collect();
        let mut sources: Vec<PathBuf> = std::fs::read_dir(&self.dir)
            .map_err(dir_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(dir_error)?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|e| e == "rs"))
            // `mod.rs` lists the shader modules for Rust; it is not one.
            .filter(|path| path.file_name().is_some_and(|n| n != "mod.rs"))
            .filter(|path| !prelude.contains(path))
            .collect();
        // Directory order is arbitrary; the generated file should not be.
        sources.sort();
        Ok(sources)
    }

    /// The WGSL, and what each entry point is called in it.
    fn compile(
        &self,
        path: &Path,
        prelude: &[(PathBuf, String)],
        source: &str,
    ) -> Result<(String, Vec<EntryPoint>), BuildError> {
        let at = |kind| BuildError {
            path: Some(path.to_path_buf()),
            kind,
        };
        // Parsed as separate sources rather than concatenated text, so a line
        // number in an error still points into the file it came from.
        let texts: Vec<&str> = prelude
            .iter()
            .map(|(_, text)| text.as_str())
            .chain(std::iter::once(source))
            .collect();
        let mut module = crate::parse_all(texts.iter().copied()).map_err(|err| {
            // Blame the prelude file when the prelude is what failed.
            let blamed = prelude
                .get(err.index)
                .map(|(path, _)| path.clone())
                .unwrap_or_else(|| path.to_path_buf());
            BuildError {
                path: Some(blamed),
                kind: BuildErrorKind::Transpile(err.error),
            }
        })?;

        let flags = match self.bindings {
            Bindings::Explicit => naga::valid::ValidationFlags::all(),
            Bindings::Host => {
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS
            }
        };
        let info = crate::validate_with(&module, flags, self.capabilities)
            .map_err(|err| at(BuildErrorKind::Validate(err)))?;

        // Pruning needs a module already known to be valid, and invalidates the
        // info that proved it, so validation runs on either side of it.
        let info = if self.prune && !module.entry_points.is_empty() {
            naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
            crate::validate_with(&module, flags, self.capabilities)
                .map_err(|err| at(BuildErrorKind::Validate(err)))?
        } else {
            info
        };

        let wgsl = crate::to_wgsl(&module, &info)
            .map_err(|err| at(BuildErrorKind::Emit(err.to_string())))?;
        Ok((wgsl, entry_points(&module)))
    }
}

/// What each entry point is called on both sides of the WGSL backend.
///
/// Naga reserves identifiers it might need to uniquify. A name that only gained
/// the `_` it appends to something already ending in a digit is written back,
/// so `atrous3x3` stays `atrous3x3`. A real collision still comes out as
/// `name_1`. A host creates a pipeline by entry point name, so that difference
/// has to reach the caller.
///
/// The MSL, HLSL and GLSL backends hand back `entry_point_names` for exactly
/// this. The WGSL one returns only a string: its `names` map is private and
/// `finish` drops it. But the namer that fills that map is public, and the
/// backend's own `reset` is these six arguments, so this asks the same question
/// of the same code rather than reading the output text back.
pub fn entry_point_names(module: &naga::Module) -> Vec<EntryPoint> {
    entry_points(module)
}

fn entry_points(module: &naga::Module) -> Vec<EntryPoint> {
    let mut names = naga::FastHashMap::default();
    naga::proc::Namer::default().reset(
        module,
        &naga::keywords::wgsl::RESERVED_SET,
        &naga::keywords::wgsl::BUILTIN_IDENTIFIER_SET,
        // An identifier must not start with two underscores.
        naga::proc::CaseInsensitiveKeywordSet::empty(),
        &["__", "_naga"],
        &mut names,
    );

    module
        .entry_points
        .iter()
        .enumerate()
        .map(|(index, entry)| EntryPoint {
            name: entry.name.clone(),
            emitted_name: crate::restore_digit_suffix(
                names[&naga::proc::NameKey::EntryPoint(index as u16)].clone(),
            ),
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
