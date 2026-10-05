//! Rust → [`naga::Module`].
//!
//! Shader modules are ordinary Rust, type-checked by `rustc` against
//! `synaga-shader`; this crate reads the same files and builds the Naga module
//! they describe. [`build::Shaders`] does that from a build script; [`parse`]
//! does it for sources in hand.
//!
//! Items are what Rust has: in any order, across sibling modules reached by
//! `use` or by path, kept or dropped by `#[cfg]`. Operand typing mirrors
//! Naga's own rules, so anything [`parse_str`] accepts is a module [`validate`]
//! accepts; untyped integer literals take their type from context, as Rust's
//! inference would.

pub mod build;
mod cfg;
mod error;
mod lower;
#[cfg(feature = "wgsl")]
mod ray_wgsl;

pub use cfg::Cfg;
pub use error::{Error, Pos};
pub use naga;

use lower::Context;

/// One file of shader source.
#[derive(Clone, Copy, Debug)]
pub struct Source<'a> {
    /// The module name the other sources reach this one by: the file stem,
    /// `brdf` for `brdf.rs`. `None` for a source nothing names in a path.
    pub name: Option<&'a str>,
    pub text: &'a str,
}

/// Parse a Rust source string into a Naga module.
pub fn parse_str(source: &str) -> Result<naga::Module, Error> {
    parse_all([source]).map_err(|err| err.error)
}

/// Parse several unnamed sources into one module, with no `cfg` set.
///
/// See [`parse`], which this is with every [`Source::name`] left out: a name
/// one source declares is still found from another, as long as exactly one
/// declares it, but a path through a module name has nothing to go by.
pub fn parse_all<'a>(
    sources: impl IntoIterator<Item = &'a str>,
) -> Result<naga::Module, SourceError> {
    let sources: Vec<Source> = sources
        .into_iter()
        .map(|text| Source { name: None, text })
        .collect();
    parse(&sources, &Cfg::new())
}

/// Parse the files of a module tree into one Naga module.
///
/// The files are sibling modules, so `use super::brdf::*` and `brdf::sample()`
/// in one reach the items of the one named `brdf`. Order does not matter, in
/// the files or in the list: every item is known before any is lowered, as in
/// Rust. `cfg` says which `#[cfg(...)]` and `cfg!(...)` hold.
///
/// Parsed separately rather than concatenated, each file keeps its own line
/// numbers, and [`SourceError::index`] says which one went wrong.
pub fn parse(sources: &[Source<'_>], cfg: &Cfg) -> Result<naga::Module, SourceError> {
    parse_expecting(sources, cfg, None)
}

/// [`parse`], holding the module to what `bindings` says about who assigns
/// bindings, so that a resource or a vertex struct that does otherwise is
/// reported where it is written.
pub(crate) fn parse_expecting(
    sources: &[Source<'_>],
    cfg: &Cfg,
    bindings: Option<build::Bindings>,
) -> Result<naga::Module, SourceError> {
    parse_shared(sources, cfg, bindings).map(|(module, _)| module)
}

/// [`parse_expecting`], and the structs the module shares with a host, as the
/// GPU lays them out, for `rustc` to check its own layout against.
pub(crate) fn parse_shared(
    sources: &[Source<'_>],
    cfg: &Cfg,
    bindings: Option<build::Bindings>,
) -> Result<(naga::Module, Vec<lower::SharedStruct>), SourceError> {
    let mut files = Vec::with_capacity(sources.len());
    for (index, source) in sources.iter().enumerate() {
        let file: syn::File = syn::parse_str(source.text).map_err(|e| SourceError {
            index,
            error: Error::from(e),
        })?;
        crate::cfg::reject_cfg_attr(&file).map_err(|error| SourceError { index, error })?;
        files.push((source.name.map(str::to_string), file));
    }
    let mut ctx = Context::new(cfg.clone(), bindings);
    ctx.lower_sources(files).map_err(|error| SourceError {
        index: ctx.failed_source.unwrap_or(0),
        error,
    })?;
    let shared = lower::shared_structs(&ctx);
    Ok((ctx.module, shared))
}

/// An [`Error`], and which of the sources handed to [`parse_all`] it came from.
#[derive(Debug)]
pub struct SourceError {
    /// Index into the sources, in the order they were given.
    pub index: usize,
    pub error: Error,
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for SourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// A Naga validation failure, with the reason it gives.
///
/// Naga puts the useful part of a validation error in the source chain: the top
/// level says only which global or function is invalid. Printing this prints
/// the whole chain, so the actual complaint is visible.
#[derive(Debug)]
pub struct ValidationError(Box<naga::WithSpan<naga::valid::ValidationError>>);

impl ValidationError {
    /// The underlying Naga error, spans included.
    pub fn into_inner(self) -> naga::WithSpan<naga::valid::ValidationError> {
        *self.0
    }

    /// The positions in source files that Naga blames, narrowest first.
    ///
    /// Every expression, statement, global and type the lowering emits is
    /// tagged with the span of the `syn` node it came from, so these are real
    /// positions rather than `Span::UNDEFINED`. Each is a byte range in
    /// whichever file that node was written in, which is what
    /// [`crate::build`] uses to turn a validation failure into a
    /// `file:line:column`. Empty when Naga blamed nothing in particular.
    pub fn spans(&self) -> impl Iterator<Item = (naga::Span, &str)> {
        self.0.spans().map(|(span, what)| (*span, what.as_str()))
    }

    /// The position Naga blames, if it blames one.
    pub fn span(&self) -> Option<naga::Span> {
        self.0.spans().next().map(|(span, _)| *span)
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)?;
        let mut source = std::error::Error::source(&*self.0);
        while let Some(err) = source {
            write!(f, ": {err}")?;
            source = err.source();
        }
        Ok(())
    }
}

impl std::error::Error for ValidationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0)
    }
}

/// Validate `module` with default Naga flags and no extra capabilities.
pub fn validate(module: &naga::Module) -> Result<naga::valid::ModuleInfo, ValidationError> {
    validate_with(
        module,
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
}

/// Validate a module whose resource bindings the host assigns.
///
/// Some engines — Blade among them — leave `@group`/`@binding` out of the
/// shader and fill them in at pipeline creation, matching globals up by name.
/// A module for one of those has globals with no binding, which the default
/// flags reject.
pub fn validate_unbound(module: &naga::Module) -> Result<naga::valid::ModuleInfo, ValidationError> {
    validate_with(
        module,
        naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
        naga::valid::Capabilities::empty(),
    )
}

/// Validate with the flags and capabilities of your choosing.
///
/// Some of the dialect needs a capability the defaults leave off — ray queries
/// need [`Capabilities::RAY_QUERY`] — and a host validating against a real
/// device has its own set anyway.
///
/// [`Capabilities::RAY_QUERY`]: naga::valid::Capabilities::RAY_QUERY
pub fn validate_with(
    module: &naga::Module,
    flags: naga::valid::ValidationFlags,
    capabilities: naga::valid::Capabilities,
) -> Result<naga::valid::ModuleInfo, ValidationError> {
    naga::valid::Validator::new(flags, capabilities)
        .validate(module)
        .map_err(|e| ValidationError(e.into_boxed()))
}

/// `Validator::validate` reports a `WithSpan<E>` in one Naga and a
/// `Box<WithSpan<E>>` in the next, and this is the one place both are accepted,
/// so a host and its transpiler need not be on the same revision to agree on
/// what a module looks like.
trait IntoBoxed<E> {
    fn into_boxed(self) -> Box<naga::WithSpan<E>>;
}

impl<E> IntoBoxed<E> for naga::WithSpan<E> {
    fn into_boxed(self) -> Box<naga::WithSpan<E>> {
        Box::new(self)
    }
}

impl<E> IntoBoxed<E> for Box<naga::WithSpan<E>> {
    fn into_boxed(self) -> Box<naga::WithSpan<E>> {
        self
    }
}

/// Emit WGSL for a validated module.
///
/// This is optional (`feature = "wgsl"`). The module is the product; printing
/// it is a client's choice. Naga's WGSL backend has no spelling for a ray
/// query and panics on one, so a module that traces a ray is rewritten into
/// the builtin calls (`rayQueryInitialize` and the rest) before that backend
/// runs, and the stand-in functions are removed from the text. The result asks
/// for `enable wgpu_ray_query`.
#[cfg(feature = "wgsl")]
pub fn to_wgsl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<String, naga::back::wgsl::Error> {
    if uses_ray_query(module).is_some() {
        let mut module = module.clone();
        ray_wgsl::rewrite(&mut module).map_err(naga::back::wgsl::Error::Custom)?;
        let info = revalidate_ray(&module).map_err(naga::back::wgsl::Error::Custom)?;
        let wgsl =
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())?;
        return Ok(restore_digit_suffix(ray_wgsl::finish_text(wgsl)));
    }
    let wgsl =
        naga::back::wgsl::write_string(module, info, naga::back::wgsl::WriterFlags::empty())?;
    Ok(restore_digit_suffix(wgsl))
}

/// Naga's WGSL writer appends `_` to any identifier that ends in a digit, so a
/// later numeric suffix stays separate. Resource names are matched by the host
/// as written, so an identifier whose only change was that underscore is put back.
#[cfg(feature = "wgsl")]
pub(crate) fn restore_digit_suffix(wgsl: String) -> String {
    let mut out = String::with_capacity(wgsl.len());
    let mut chars = wgsl.chars().peekable();
    while let Some(ch) = chars.next() {
        if is_ident_start(ch) {
            let mut ident = String::new();
            ident.push(ch);
            while let Some(next) = chars.peek().copied() {
                if is_ident_continue(next) {
                    ident.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            if let Some(kept) = ident.strip_suffix('_') {
                if kept.ends_with(|c: char| c.is_ascii_digit()) {
                    out.push_str(kept);
                    continue;
                }
            }
            out.push_str(&ident);
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(feature = "wgsl")]
fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

#[cfg(feature = "wgsl")]
fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

/// The rewritten module still has unbound resources and ray-query types, so
/// validation is the permissive host kind with every capability on.
#[cfg(feature = "wgsl")]
fn revalidate_ray(module: &naga::Module) -> Result<naga::valid::ModuleInfo, String> {
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    naga::valid::Validator::new(flags, naga::valid::Capabilities::all())
        .validate(module)
        .map_err(|err| {
            let mut message = err.to_string();
            let mut source = std::error::Error::source(&err);
            while let Some(next) = source {
                message.push_str(": ");
                message.push_str(&next.to_string());
                source = next.source();
            }
            message
        })
}

/// The name of the first function that traces a ray query, if any does.
#[cfg(feature = "wgsl")]
fn uses_ray_query(module: &naga::Module) -> Option<String> {
    fn in_block(block: &naga::Block) -> bool {
        block.iter().any(|stmt| match stmt {
            naga::Statement::RayQuery { .. } => true,
            naga::Statement::Block(inner) => in_block(inner),
            naga::Statement::If { accept, reject, .. } => in_block(accept) || in_block(reject),
            naga::Statement::Loop {
                body, continuing, ..
            } => in_block(body) || in_block(continuing),
            naga::Statement::Switch { cases, .. } => cases.iter().any(|c| in_block(&c.body)),
            _ => false,
        })
    }
    let functions = module
        .functions
        .iter()
        .map(|(_, f)| (f.name.clone().unwrap_or_default(), &f.body));
    let entries = module
        .entry_points
        .iter()
        .map(|e| (e.name.clone(), &e.function.body));
    functions
        .chain(entries)
        .find(|(_, body)| in_block(body))
        .map(|(name, _)| name)
}

#[cfg(all(test, feature = "wgsl"))]
mod tests {
    use super::*;

    #[test]
    fn add_f32() {
        let module = parse_str("fn add(a: f32, b: f32) -> f32 { a + b }").unwrap();
        let info = validate(&module).expect("naga validation");
        let wgsl = to_wgsl(&module, &info).expect("wgsl");
        assert!(wgsl.contains("fn add"), "{wgsl}");
        assert!(wgsl.contains('+'), "{wgsl}");
    }
}
