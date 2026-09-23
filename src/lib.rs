//! Native Rust → [`naga::Module`] frontend.
//!
//! v0 dialect: free functions, scalars (`f32` / `u32` / `i32` / `bool`),
//! vectors (`vec2`/`vec3`/`vec4` and `vecN<T>`),
//! matrices (`mat2`/`mat3`/`mat4` and `matCxR`), literals, unary/binary operators,
//! `as` casts, `let`, assignment, `if`/`else`, `loop`/`while`, `return`, and
//! `#[vertex]`/`#[fragment]`/`#[compute]` entry points,
//! `#[group]`/`#[binding]` globals, and named structs — including structs that
//! carry `#[location]` / `#[builtin]` bindings on their fields, as vertex
//! outputs and fragment inputs do.
//! No references, methods, or generics yet.
//!
//! Operand typing mirrors Naga's own rules, so anything [`parse_str`] accepts
//! is a module [`validate`] accepts; untyped integer literals take their type
//! from context, as Rust's inference would.

pub mod build;
mod error;
mod lower;
mod ray_wgsl;

pub use error::Error;
pub use naga;

use lower::Context;

/// Parse a Rust source string into a Naga module.
pub fn parse_str(source: &str) -> Result<naga::Module, Error> {
    parse_all([source]).map_err(|err| err.error)
}

/// Parse several sources into one module, in order, as if concatenated.
///
/// This is how a shared prelude is combined with the module that uses it.
/// Concatenating the text first would work too, but then every line number in
/// an error from the second file is off by the length of the first; parsed
/// separately, each keeps its own, and [`SourceError::index`] says which one
/// went wrong.
pub fn parse_all<'a>(
    sources: impl IntoIterator<Item = &'a str>,
) -> Result<naga::Module, SourceError> {
    let mut ctx = Context::new();
    for (index, source) in sources.into_iter().enumerate() {
        let at = |error: Error| SourceError { index, error };
        let file: syn::File = syn::parse_str(source).map_err(|e| at(Error::from(e)))?;
        ctx.lower_file(file).map_err(at)?;
    }
    Ok(ctx.module)
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
        .map_err(|e| ValidationError(Box::new(e)))
}

/// Emit WGSL for a validated module.
///
/// Naga's WGSL backend has no spelling for a ray query and panics on one. A
/// module that traces a ray is rewritten into the builtin calls WGSL does
/// spell (`rayQueryInitialize` and the rest) before that backend runs, and the
/// stand-in functions are removed from the text. The result asks for
/// `enable wgpu_ray_query`, which is what a WGSL frontend needs to read those
/// calls back as ray queries.
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

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

/// The rewritten module still has unbound resources and ray-query types, so
/// validation is the permissive host kind with every capability on.
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

#[cfg(test)]
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
