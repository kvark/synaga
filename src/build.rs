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
//! reaches is left alone. `mod.rs` lists the modules for `rustc` and is not a
//! shader; it is read only for `check_layout!`, below.
//!
//! What an entry point does not reach is pruned from its module, so a helper
//! file can hold everything the shaders need between them.
//!
//! `#[cfg(...)]` and `cfg!(...)` are settled the way `rustc` settles them for
//! the crate being built: `cfg!(debug_assertions)` holds in a debug build.
//!
//! The modules are bincode, behind a short header; see `synaga_shader::ir`.
//!
//! # Checking what the host shares
//!
//! Next to the module listing goes a file of `size_of` and `offset_of!`
//! assertions, one for each struct the host shares and each field of it that
//! `mod.rs` can name, which `synaga_shader::check_layout!()` in `mod.rs`
//! includes. The transpiler works out `rustc`'s layout of those structs from
//! their types and refuses one the GPU lays out differently; the assertions
//! have `rustc` confirm it on the target being built.
//!
//! The build looks for that `check_layout!` where `rustc` looks for the
//! module: `mod.rs` in the directory, or the file named after the directory
//! beside it, as in `src/shaders.rs`. When there are shared structs and it is
//! not there, the build warns.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use syn::visit::Visit;

use crate::{Cfg, Error, Source};

/// The major version of the Naga this crate builds against, recorded in each
/// module's header and asserted against
/// [`synaga_shader::ir::NAGA_MAJOR`] in the file it generates. Kept in step
/// with `Cargo.toml`, and with the shader crate, by tests.
pub const NAGA_MAJOR: u8 = synaga_shader::ir::NAGA_MAJOR;

/// The generated file's name unless [`Shaders::module_name`] says otherwise,
/// and so what `include_ir!()` and `check_layout!()` with no file include.
const DEFAULT_MODULE_NAME: &str = "shaders.rs";

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
    /// Where in that file, for a failure Naga blamed on a span. A transpile
    /// error carries its own position, which is finer.
    pub at: Option<(usize, usize)>,
    pub kind: BuildErrorKind,
}

impl BuildError {
    fn new(path: Option<PathBuf>, kind: BuildErrorKind) -> Self {
        Self {
            path,
            at: None,
            kind,
        }
    }

    /// The position to print after the path, preferring the finer of the two.
    fn location(&self) -> Option<(usize, usize)> {
        if let BuildErrorKind::Transpile(err) = &self.kind {
            if let Some(at) = err.location() {
                return Some(at);
            }
        }
        self.at
    }
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
    /// Every fault found, one per line.
    ///
    /// A build script that stops at the first failure makes fixing a shader a
    /// sequence of rebuilds, which is why a run reports all of them.
    Many(Errors),
}

impl std::fmt::Display for BuildError {
    /// Formatted as `path:line:column: message`, which is the shape Cargo and
    /// editors already know how to turn into a jump.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(path) = &self.path {
            write!(f, "{}", path.display())?;
            if let Some((line, column)) = self.location() {
                write!(f, ":{line}:{column}")?;
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
            BuildErrorKind::Many(errors) => write!(f, "{errors}"),
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
    /// The module itself, when [`Shaders::keep_modules`] asked for it.
    ///
    /// Holding every module costs a second copy of each, so it is off unless
    /// asked for. A build script that wants to print or re-validate a shader
    /// asks rather than decoding the bytes it just wrote. Read it with
    /// [`Shader::module`].
    pub(crate) module: Option<naga::Module>,
}

impl Shader {
    /// The module itself, so a build script can print it, validate it again, or
    /// hand it to a Naga backend without decoding the bytes it just wrote.
    ///
    /// `None` unless [`Shaders::keep_modules`] asked for it, which
    /// [`Shaders::wgsl`] does: a script with no use for the module should not
    /// pay to hold a second copy of every one.
    pub fn module(&self) -> Option<&naga::Module> {
        self.module.as_ref()
    }
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
    /// Where to write the WGSL, under `OUT_DIR`, if it is wanted.
    #[cfg(feature = "wgsl")]
    wgsl: Option<PathBuf>,
    /// Keep each module in [`Shader`], for a caller that wants to look at one.
    keep_modules: bool,
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
            module_name: DEFAULT_MODULE_NAME.into(),
            prune: true,
            bindings: Bindings::Explicit,
            capabilities: naga::valid::Capabilities::empty(),
            cfg: None,
            #[cfg(feature = "wgsl")]
            wgsl: None,
            keep_modules: false,
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
    ///
    /// The layout assertions go next to it, named after it: `shaders_layout.rs`,
    /// or `gpu_layout.rs` for `gpu.rs`.
    pub fn module_name(mut self, name: impl Into<String>) -> Self {
        self.module_name = name.into();
        self
    }

    /// Who assigns `@group` and `@binding`. See [`Bindings`].
    pub fn bindings(mut self, bindings: Bindings) -> Self {
        self.bindings = bindings;
        self
    }

    /// Write each module beside its bytes as WGSL, in a `wgsl/` directory under
    /// `out_dir`, and keep the modules so a caller can reach them.
    ///
    /// The IR is the product, but it is not readable. A shader whose module is
    /// not what you expected is far easier to diagnose as text, and a `.wgsl`
    /// next to the `.naga` is also something a test can compare against.
    /// Requires the `wgsl` feature, which is off by default.
    ///
    /// ```no_run
    /// synaga::build::Shaders::new().wgsl().run();
    /// ```
    #[cfg(feature = "wgsl")]
    pub fn wgsl(mut self) -> Self {
        self.wgsl = Some(std::path::PathBuf::from("wgsl"));
        self.keep_modules = true;
        self
    }

    /// The directory [`Shaders::wgsl`] writes into, under `OUT_DIR`.
    ///
    /// Needs the `wgsl` feature; without it there is no text to write.
    #[cfg(feature = "wgsl")]
    pub fn wgsl_dir(mut self, dir: impl AsRef<std::path::Path>) -> Self {
        self.wgsl = Some(dir.as_ref().to_path_buf());
        self.keep_modules = true;
        self
    }

    /// Keep each module in the [`Shader`] this hands back, reachable with
    /// [`Shader::module`].
    ///
    /// Off by default: holding every module is a second copy of each, which a
    /// build script that only writes bytes has no use for. Useful for a
    /// script that wants to report on what it built — how many entry points, how
    /// many globals survived pruning, the WGSL of one that looks wrong.
    pub fn keep_modules(mut self, keep: bool) -> Self {
        self.keep_modules = keep;
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
        let out_dir = std::env::var_os("OUT_DIR")
            .ok_or(BuildError::new(None, BuildErrorKind::NotABuildScript))?;
        self.emit_to(Path::new(&out_dir))
    }

    /// Compile into `out_dir`, without needing Cargo's environment.
    ///
    /// `emit` is what a build script wants; this is for testing the same path
    /// without one.
    pub fn emit_to(&self, out_dir: &Path) -> Result<Vec<Shader>, BuildError> {
        let cfg = self.cfg.clone().unwrap_or_else(Cfg::from_cargo_env);
        self.declare_rebuild_triggers();

        let mut errors = Errors::default();
        let files = self.read_files(&mut errors)?;
        report_unreached(&files, &mut errors)?;
        let stems: Vec<&str> = files.iter().map(|f| f.stem.as_str()).collect();
        let mut shaders = Vec::new();
        let mut shared = BTreeSet::new();
        // The reader is the host's own `naga`, reached through a dependency the
        // build script does not control, and a mismatched pair decodes to a
        // runtime panic at the first `decode()`. Asserted at compile time
        // instead, where the error names the version rather than the bytes.
        let mut generated = String::from(
            "// Generated by synaga. Do not edit.\n\
             //\n\
             // Each constant is a serialized Naga module; see `synaga_shader::ir`.\n\
             //\n\
             // This checks that synaga and synaga-shader agree on Naga's major\n\
             // version. The number below is the writer's; the constant is the\n\
             // shader crate's supported version. It cannot inspect the host's\n\
             // own Naga dependency, which must resolve to the writer's package.\n\
             const _: () = assert!(\n",
        );
        let _ = write!(
            generated,
            "    ::synaga_shader::ir::NAGA_MAJOR == {NAGA_MAJOR},\n\
             \x20   \"synaga wrote these modules with Naga {NAGA_MAJOR}, but this crate's \
             synaga-shader reads them with a different Naga major version; the synaga \
             build-dependency and the synaga-shader dependency must be the same \
             version, and their `naga`s must resolve to the same package\"\n\
             );\n"
        );

        for (index, file) in files.iter().enumerate() {
            if !file.has_entry_point {
                continue;
            }
            let sources = reachable(&files, &stems, index);
            let Compiled {
                bytes,
                entry_points,
                shared: structs,
                module,
                #[cfg_attr(not(feature = "wgsl"), allow(unused_variables))]
                info,
            } = match self.compile(&files, &sources, &cfg, &mut errors) {
                Ok(compiled) => compiled,
                // Recorded; the rest of the shaders still get a turn, so one
                // build reports every fault rather than the first.
                Err(_) => continue,
            };
            shared.extend(structs);
            // Used by the WGSL error path below, and the root of what a failure
            // in this module belongs to.
            #[cfg_attr(not(feature = "wgsl"), allow(unused_variables))]
            let root = files[sources[0]].path.clone();

            let output_path = out_dir.join(format!("{}.naga", file.stem));
            write_if_changed(&output_path, &bytes)
                .map_err(|e| BuildError::new(Some(output_path.clone()), BuildErrorKind::Io(e)))?;

            // The IR is what a host runs, but it is not readable. When asked,
            // the same module is written as text beside it, which is what makes
            // a surprising module diagnosable and testable.
            #[cfg(feature = "wgsl")]
            if let Some(dir) = &self.wgsl {
                let text = crate::to_wgsl(&module, &info).map_err(|e| {
                    BuildError::new(
                        Some(root.clone()),
                        BuildErrorKind::Emit(format!("cannot print this module as WGSL: {e}")),
                    )
                })?;
                let target = out_dir.join(dir).join(format!("{}.wgsl", file.stem));
                write_if_changed(&target, text.as_bytes())
                    .map_err(|e| BuildError::new(Some(target.clone()), BuildErrorKind::Io(e)))?;
            }
            let _ = &module;

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
                module: self.keep_modules.then_some(module),
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
        write_if_changed(&module_path, generated.as_bytes())
            .map_err(|e| BuildError::new(Some(module_path.clone()), BuildErrorKind::Io(e)))?;

        let layout_path = out_dir.join(layout_name(&self.module_name));
        let layout = layout_checks(&shared);
        write_if_changed(&layout_path, layout.as_bytes())
            .map_err(|e| BuildError::new(Some(layout_path.clone()), BuildErrorKind::Io(e)))?;

        // The layout checks are the one thing standing between a shader and a
        // silently misread uniform, and nothing includes them unless the crate
        // says `synaga_shader::check_layout!()`. Said out loud, since the
        // transpiler's own check is a model of `rustc` and not `rustc`.
        //
        // Only when the build is not already failing: a tree with a dozen
        // mistakes in it should hear what is wrong with its shaders before it
        // hears what is wrong with the build script's own output.
        if errors.is_empty() {
            if let Some(warning) = self.layout_warning(shared.len(), &layout_path, &cfg) {
                println!("cargo::warning={warning}");
            }
        }

        // Reported once everything is built, so a shader that is broken for one
        // reason is not also reported as unreached, and one run lists every
        // fault in the tree rather than the first.
        if !errors.is_empty() {
            return Err(BuildError::new(None, BuildErrorKind::Many(errors)));
        }
        Ok(shaders)
    }

    /// Everything the output depends on, declared to Cargo.
    ///
    /// Which files, which directory, and — the one that is easy to miss — the
    /// `CARGO_CFG_*` variables the `cfg` comes from. A feature enabled or
    /// disabled changes which items exist, and Cargo does not re-run a build
    /// script for an environment variable unless the script says so.
    fn declare_rebuild_triggers(&self) {
        // Anything added, removed or edited under the directory changes the
        // output. Cargo watches a directory path recursively.
        println!("cargo::rerun-if-changed={}", self.dir.display());
        // Whether `cfg` is left to Cargo decides what it holds, and the caller
        // may have set it by hand instead — in which case nothing below applies.
        if self.cfg.is_none() {
            for (key, _) in std::env::vars() {
                if key.starts_with("CARGO_CFG_") {
                    println!("cargo::rerun-if-env-changed={key}");
                }
            }
        }
        // The file that lists the shader modules is read for `check_layout!`,
        // and one beside the directory is not under it.
        if let Some(file) = self.module_file() {
            println!("cargo::rerun-if-changed={}", file.display());
        }
    }

    /// The file that lists the shader modules for `rustc`: `mod.rs` in the
    /// directory, or the file named after the directory beside it, the two
    /// places `mod shaders;` looks.
    fn module_file(&self) -> Option<PathBuf> {
        let mod_rs = self.dir.join("mod.rs");
        if mod_rs.is_file() {
            return Some(mod_rs);
        }
        let name = self.dir.file_name()?.to_str()?;
        let beside = self.dir.with_file_name(format!("{name}.rs"));
        beside.is_file().then_some(beside)
    }

    /// What to say about the layout checks, if anything: there are shared
    /// structs, and the file that lists the shader modules does not include
    /// the checks for them.
    ///
    /// Quiet when that file cannot be read or parsed. `rustc` says what is
    /// wrong with it, and whether it includes the checks is not known.
    fn layout_warning(&self, shared: usize, layout_path: &Path, cfg: &Cfg) -> Option<String> {
        if shared == 0 {
            return None;
        }
        let layout = layout_name(&self.module_name);
        let call = if layout == layout_name(DEFAULT_MODULE_NAME) {
            "synaga_shader::check_layout!();".to_owned()
        } else {
            format!("synaga_shader::check_layout!({layout:?});")
        };
        let found = format!(
            "synaga: {shared} shared struct(s) in {} have `rustc` layout checks in {}",
            self.dir.display(),
            layout_path.display(),
        );
        let otherwise = "or the layout is checked only by synaga's own model of `rustc`";
        let Some(file) = self.module_file() else {
            return Some(format!(
                "{found}, which nothing includes — list the shader modules in {}, with `{call}` beside them, {otherwise}",
                self.dir.join("mod.rs").display(),
            ));
        };
        let source = std::fs::read_to_string(&file).ok()?;
        if includes_layout(&source, &layout, cfg)? {
            return None;
        }
        Some(format!(
            "{found}, which {} does not include — add `{call}` to it, {otherwise}",
            file.display(),
        ))
    }

    /// Every `.rs` file in the directory but `mod.rs`, read and looked over, in
    /// a stable order, and any file nothing can read or parse reported.
    fn read_files(&self, errors: &mut Errors) -> Result<Vec<File>, BuildError> {
        let dir_error =
            |e: std::io::Error| BuildError::new(Some(self.dir.clone()), BuildErrorKind::Io(e));
        let paths = collect_paths(&self.dir, &dir_error)?;

        let mut files = Vec::with_capacity(paths.len());
        for path in paths {
            println!("cargo::rerun-if-changed={}", path.display());
            // A file that cannot be read or parsed is reported and skipped, so
            // one broken file does not hide the state of the others.
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(e) => {
                    errors.push(BuildError::new(Some(path), BuildErrorKind::Io(e)));
                    continue;
                }
            };
            let syntax: syn::File = match syn::parse_str(&text) {
                Ok(syntax) => syntax,
                Err(e) => {
                    errors.push(BuildError::new(
                        Some(path),
                        BuildErrorKind::Transpile(e.into()),
                    ));
                    continue;
                }
            };
            if let Err(error) = crate::cfg::reject_cfg_attr(&syntax) {
                errors.push(BuildError::new(
                    Some(path),
                    BuildErrorKind::Transpile(error),
                ));
                continue;
            }
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

    /// The serialized module, what each entry point is called, and what it
    /// shares with the host.
    ///
    /// `errors` collects what is wrong with this module, so one pass reports
    /// every fault in it rather than the first.
    fn compile(
        &self,
        files: &[File],
        sources: &[usize],
        cfg: &Cfg,
        errors: &mut Errors,
    ) -> Result<Compiled, BuildError> {
        let root = &files[sources[0]].path;
        let at = |kind| BuildError::new(Some(root.clone()), kind);
        let texts: Vec<Source> = sources
            .iter()
            .map(|&i| Source {
                name: Some(&files[i].stem),
                text: &files[i].text,
            })
            .collect();
        let parsed = crate::parse_shared(&texts, cfg, Some(self.bindings));
        let (mut module, shared) = match parsed {
            Ok(parsed) => parsed,
            Err(err) => {
                // One failure stops this module: without a module there is
                // nothing to validate, and every later complaint would be about
                // a module that was never built. What is wrong with the *other*
                // shaders is still reported, by the caller.
                let error = err.error;
                let blamed = files[sources[err.index]].path.clone();
                return Err(errors.push(BuildError::new(
                    Some(blamed),
                    BuildErrorKind::Transpile(error),
                )));
            }
        };

        let flags = match self.bindings {
            Bindings::Explicit => naga::valid::ValidationFlags::all(),
            Bindings::Host => {
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS
            }
        };
        // Naga names what is wrong and its own chain says why; the span says
        // where, as a byte range in one of the module's files, so it is turned
        // back into a `file:line:column` before it is lost.
        let check = |module: &naga::Module| {
            crate::validate_with(module, flags, self.capabilities).map_err(|err| {
                let (path, at) = where_of(err.span(), files, sources, root);
                let mut error = BuildError::new(Some(path), BuildErrorKind::Validate(err));
                error.at = at;
                error
            })
        };
        let info = check(&module).map_err(|error| errors.push(error))?;

        // Pruning needs a module already known to be valid. It drops the
        // validation info, so a pruned module is checked again.
        let info = if self.prune {
            naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
            check(&module).map_err(|error| errors.push(error))?
        } else {
            info
        };

        let mut bytes = HEADER.to_vec();
        bincode::serde::encode_into_std_write(&module, &mut bytes, bincode::config::standard())
            .map_err(|err| at(BuildErrorKind::Emit(err.to_string())))?;
        Ok(Compiled {
            bytes,
            entry_points: entry_points(&module),
            shared,
            module,
            info,
        })
    }
}

/// What went wrong, across every module, so one build reports all of it.
///
/// A build script that stops at the first failure makes fixing a shader a
/// sequence of rebuilds; a tree the size of Blade's has a dozen mistakes in it
/// at a time. Each is reported on its own `cargo::error=` line, with the file
/// and position it belongs to.
#[derive(Debug, Default)]
pub struct Errors(Vec<String>);

impl Errors {
    /// Keep `error` for the report and hand it back for the early return.
    ///
    /// A shader that is also a helper of another one is compiled into both, so
    /// the same fault arrives twice; saying it once is the point of listing
    /// them all.
    fn push(&mut self, error: BuildError) -> BuildError {
        let message = error.to_string();
        if !self.0.contains(&message) {
            self.0.push(message);
        }
        error
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Display for Errors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, error) in self.0.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{error}")?;
        }
        Ok(())
    }
}

/// Which file a Naga span belongs to, and where in it.
///
/// A module is built from several files, and a span is a byte range in whichever
/// one the node was written in. Each file is searched for a node covering the
/// range, which is unambiguous because every span the lowering emits is the
/// range of some `syn` node in one of them.
fn where_of(
    span: Option<naga::Span>,
    files: &[File],
    sources: &[usize],
    root: &Path,
) -> (PathBuf, Option<(usize, usize)>) {
    let Some(range) = span.and_then(|span| span.to_range()) else {
        return (root.to_path_buf(), None);
    };
    for &i in sources {
        if let Some(at) = line_column(&files[i].text, range.clone()) {
            return (files[i].path.clone(), Some(at));
        }
    }
    (root.to_path_buf(), None)
}

/// The 1-based line and column of a byte range in `text`.
fn line_column(text: &str, range: std::ops::Range<usize>) -> Option<(usize, usize)> {
    if range.start >= text.len() {
        return None;
    }
    let head = &text[..range.start];
    let line = head.matches('\n').count() + 1;
    let column = head.rsplit('\n').next().map(str::len).unwrap_or(0) + 1;
    Some((line, column))
}

/// One shader, compiled.
struct Compiled {
    /// The serialized module.
    bytes: Vec<u8>,
    entry_points: Vec<EntryPoint>,
    /// The structs it shares with the host, as the GPU lays them out.
    shared: Vec<crate::lower::SharedStruct>,
    /// The module itself, for a caller that wants to read or print it.
    module: naga::Module,
    /// Its validation info, which printing the module as WGSL needs.
    info: naga::valid::ModuleInfo,
}

/// Write `bytes` to `path`, leaving the file alone if it already says that.
///
/// `OUT_DIR` outlives a single build, and rewriting a file whose contents did
/// not change makes everything that tracks it by mtime — Cargo's own
/// dependency tracking, a file watcher, an incremental build — do the work
/// again for nothing.
fn write_if_changed(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Ok(existing) = std::fs::read(path) {
        if existing == bytes {
            return Ok(());
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)
}

/// Report a file that no shader reaches.
///
/// A file nothing reaches is compiled by nothing: `rustc` checks it as a shader
/// while its contents are in no module, which is exactly the gap this whole
/// design exists to close. A helper meant only for the CPU is legitimate, but it
/// belongs outside the shader directory, where nothing claims it is a shader.
///
/// A file that is an entry point of its own is never unreached, and neither is
/// one whose failure has already been reported for another reason.
fn report_unreached(files: &[File], errors: &mut Errors) -> Result<(), BuildError> {
    if !files.iter().any(|file| file.has_entry_point) {
        // Nothing is a shader, so every file is a helper and nothing is wrong.
        return Ok(());
    }
    let stems: Vec<&str> = files.iter().map(|f| f.stem.as_str()).collect();
    let mut reached = vec![false; files.len()];
    for (index, file) in files.iter().enumerate() {
        if !file.has_entry_point {
            continue;
        }
        for reached_index in reachable(files, &stems, index) {
            reached[reached_index] = true;
        }
    }
    for (index, file) in files.iter().enumerate() {
        if reached[index] {
            continue;
        }
        errors.push(BuildError::new(
            Some(file.path.clone()),
            BuildErrorKind::Transpile(Error::UnreachedFile(
                file.path.display().to_string(),
                file.stem.clone(),
            )),
        ));
    }
    Ok(())
}

/// Every `.rs` file under `dir`, in a stable order.
///
/// Subdirectories are searched too, since a shader tree that grows past a
/// screenful of files wants them. A file is a module named after its stem, so
/// two files of one stem in different directories would be ambiguous; that is
/// an error rather than a silent choice.
fn collect_paths(
    dir: &Path,
    io_error: &dyn Fn(std::io::Error) -> BuildError,
) -> Result<Vec<PathBuf>, BuildError> {
    let entries = std::fs::read_dir(dir).map_err(io_error)?;
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry.map_err(io_error)?.path();
        if path.is_dir() {
            paths.extend(collect_paths(&path, io_error)?);
            continue;
        }
        if path.extension().is_some_and(|e| e == "rs")
            // `mod.rs` lists the shader modules for Rust; it is not one.
            && !path.file_name().is_some_and(|n| n == "mod.rs")
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
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
///
/// Nested `mod { .. }` counts, and it has to: an entry point inside one is not
/// something the transpiler will compile, so treating the file as a shader with
/// no entry point in it would skip it silently. Counting it here means the file
/// is built, and the lowering refuses the `mod` by name.
fn has_entry_point(file: &syn::File) -> bool {
    fn any(items: &[syn::Item]) -> bool {
        items.iter().any(|item| match item {
            syn::Item::Fn(func) => func
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("entry_point")),
            syn::Item::Mod(item_mod) => item_mod
                .content
                .as_ref()
                .is_some_and(|(_, items)| any(items)),
            _ => false,
        })
    }
    any(&file.items)
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

/// The file the layout checks for `module_name` go in: `shaders_layout.rs`
/// for `shaders.rs`.
pub fn layout_name(module_name: &str) -> String {
    let stem = module_name.strip_suffix(".rs").unwrap_or(module_name);
    format!("{stem}_layout.rs")
}

/// Does `source` say `check_layout!` for the file `layout`, as one of its own
/// items that `cfg` keeps? `None` if it does not parse.
///
/// Only the file's own items count: inside a nested `mod`, the checks'
/// `self::` paths would start from the wrong module.
fn includes_layout(source: &str, layout: &str, cfg: &Cfg) -> Option<bool> {
    let file: syn::File = syn::parse_str(source).ok()?;
    Some(file.items.iter().any(|item| {
        let syn::Item::Macro(item) = item else {
            return false;
        };
        let named_check_layout = item
            .mac
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "check_layout");
        // A `cfg` this cannot settle is left to `rustc` rather than warned
        // about.
        if !named_check_layout || !cfg.keeps(&item.attrs).unwrap_or(true) {
            return false;
        }
        let included = if item.mac.tokens.is_empty() {
            layout_name(DEFAULT_MODULE_NAME)
        } else {
            match syn::parse2::<syn::LitStr>(item.mac.tokens.clone()) {
                Ok(name) => name.value(),
                Err(_) => return false,
            }
        };
        included == layout
    }))
}

/// `rustc`'s layout of each shared struct, asserted to be the GPU's: the
/// layout check in the transpiler works out Rust's from the types, and this
/// asks `rustc` itself, on the target being built.
///
/// The whole body sits in one `const _: ()`, so the file is a no-op when it is
/// not included — which is why the build warns when there is something to check
/// and nothing has included it.
fn layout_checks(shared: &BTreeSet<crate::lower::SharedStruct>) -> String {
    let mut out = String::from(
        "// Generated by synaga. Do not edit.\n\
         //\n\
         // `rustc`'s layout of each struct the shaders share with the host, checked\n\
         // against the GPU's. `synaga_shader::check_layout!` includes this in the\n\
         // module that lists the shader modules.\n\
         //\n\
         // `self::` keeps a module from being read as a crate of the same name,\n\
         // which a crate that warns about `unused_qualifications` would call\n\
         // unnecessary.\n\
         #[allow(unused_qualifications)]\n\
         const _: () = {\n",
    );
    for s in shared {
        let path = format!("self::{}::{}", s.module, s.name);
        if let Some(size) = s.size {
            let _ = writeln!(
                out,
                "    assert!(::core::mem::size_of::<{path}>() == {size}, \
                 \"`{}::{}` is {size} bytes on the GPU\");",
                s.module, s.name
            );
        }
        // Alignment follows from the size and the offsets for every type a
        // shader holds, but a `#[repr(C, align(N))]` can raise it on its own,
        // and a struct nested in another is placed by it.
        let _ = writeln!(
            out,
            "    assert!(::core::mem::align_of::<{path}>() == {}, \
             \"`{}::{}` is aligned {} on the GPU\");",
            s.align, s.module, s.name, s.align
        );
        for (field, offset) in &s.fields {
            let _ = writeln!(
                out,
                "    assert!(::core::mem::offset_of!({path}, {field}) == {offset}, \
                 \"`{}::{}::{field}` is at byte {offset} on the GPU\");",
                s.module, s.name
            );
        }
    }
    out.push_str("};\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recorded_naga_version_is_the_one_depended_on() {
        let manifest = include_str!("../Cargo.toml");
        let wanted = format!("naga = {{ version = \"{}\"", super::NAGA_MAJOR);
        assert!(manifest.contains(&wanted), "update NAGA_MAJOR");
    }

    fn includes_default(source: &str) -> Option<bool> {
        includes_layout(source, "shaders_layout.rs", &Cfg::new())
    }

    #[test]
    fn check_layout_is_found_however_it_is_spelled() {
        for source in [
            "pub mod sprite;\nsynaga_shader::check_layout!();",
            "use synaga_shader::check_layout;\ncheck_layout!();",
            "synaga_shader::check_layout!(\"shaders_layout.rs\");",
            "//! Shaders.\n#![allow(non_upper_case_globals)]\npub mod sprite;\nsynaga_shader::check_layout! {}",
        ] {
            assert_eq!(includes_default(source), Some(true), "{source}");
        }
    }

    #[test]
    fn check_layout_counts_only_for_this_build_s_file() {
        assert_eq!(includes_default("pub mod sprite;"), Some(false));
        // Another build's file, and the default one for a build that renamed
        // its own.
        assert_eq!(
            includes_default("synaga_shader::check_layout!(\"gpu_layout.rs\");"),
            Some(false)
        );
        assert_eq!(
            includes_layout(
                "synaga_shader::check_layout!();",
                "gpu_layout.rs",
                &Cfg::new()
            ),
            Some(false)
        );
        // In a nested module, its paths would start from the wrong place.
        assert_eq!(
            includes_default("mod checks { synaga_shader::check_layout!(); }"),
            Some(false)
        );
        // Under a `cfg` that does not hold, `rustc` never sees it.
        let gated = "#[cfg(feature = \"checks\")]\nsynaga_shader::check_layout!();";
        assert_eq!(includes_default(gated), Some(false));
        let cfg = Cfg::new().with_value("feature", "checks");
        assert_eq!(
            includes_layout(gated, "shaders_layout.rs", &cfg),
            Some(true)
        );
        // Broken, it is `rustc`'s to report.
        assert_eq!(includes_default("pub mod sprite"), None);
    }

    /// An empty `shaders` directory in a directory of its own under the
    /// system's temporary directory.
    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("synaga-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("shaders");
        std::fs::create_dir_all(&dir).expect("create scratch");
        dir
    }

    #[test]
    fn the_layout_warning_looks_where_rustc_looks_for_the_module() {
        let out = Path::new("out/shaders_layout.rs");
        let cfg = Cfg::new();
        let dir = scratch("module_file");
        let shaders = Shaders::new().dir(&dir);

        // Nothing shared, nothing to check.
        assert_eq!(shaders.layout_warning(0, out, &cfg), None);
        // Nothing lists the modules where `rustc` would look.
        let warning = shaders.layout_warning(2, out, &cfg).expect("a warning");
        assert!(warning.contains("which nothing includes"), "{warning}");
        assert!(
            warning.contains(&*dir.join("mod.rs").to_string_lossy()),
            "{warning}"
        );

        std::fs::write(dir.join("mod.rs"), "pub mod sprite;\n").expect("write");
        let warning = shaders.layout_warning(2, out, &cfg).expect("a warning");
        assert!(warning.contains("mod.rs does not include"), "{warning}");
        assert!(
            warning.contains("add `synaga_shader::check_layout!();` to it"),
            "{warning}"
        );
        std::fs::write(
            dir.join("mod.rs"),
            "pub mod sprite;\nsynaga_shader::check_layout!();\n",
        )
        .expect("write");
        assert_eq!(shaders.layout_warning(2, out, &cfg), None);

        // `shaders.rs` beside the directory, for a renamed module.
        std::fs::remove_file(dir.join("mod.rs")).expect("remove");
        let beside = dir.with_file_name("shaders.rs");
        std::fs::write(&beside, "pub mod sprite;\n").expect("write");
        let renamed = Shaders::new().dir(&dir).module_name("gpu.rs");
        let warning = renamed.layout_warning(2, out, &cfg).expect("a warning");
        assert!(warning.contains("shaders.rs does not include"), "{warning}");
        assert!(
            warning.contains("add `synaga_shader::check_layout!(\"gpu_layout.rs\");` to it"),
            "{warning}"
        );
        std::fs::write(
            &beside,
            "pub mod sprite;\nsynaga_shader::check_layout!(\"gpu_layout.rs\");\n",
        )
        .expect("write");
        assert_eq!(renamed.layout_warning(2, out, &cfg), None);

        let _ = std::fs::remove_dir_all(dir.parent().expect("a parent"));
    }
}
