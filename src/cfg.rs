//! `#[cfg(...)]` and `cfg!(...)`.
//!
//! `rustc` drops an item whose `cfg` does not hold and turns `cfg!(...)` into
//! `true` or `false`. The transpiler reads the same source, so it has to make
//! the same call, which means knowing what `rustc` was told: in a build script,
//! Cargo passes that along as `CARGO_CFG_*` variables.

use std::collections::HashSet;

use crate::Error;

/// Which configuration predicates hold.
///
/// Empty by default, as for a crate built with no features, no
/// `debug_assertions`, and no target: every `cfg` naming one of those is false.
/// A build script wants [`Cfg::from_cargo_env`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cfg {
    set: HashSet<(String, Option<String>)>,
    /// Predicates Cargo does not tell a build script about. Treating one as
    /// false would silently pick a different shader from the Rust source.
    unavailable: HashSet<String>,
}

/// These may differ between the normal library and its test harness, which
/// share a build script's output. `PROFILE=debug` holds for both `cargo build`
/// and `cargo test`, and cannot decide them.
const RUSTC_ONLY: [&str; 3] = ["test", "doctest", "miri"];

/// `cfg_attr` can introduce an entry point, change a field's binding, or alter
/// a shared layout. Ignoring it would accept different Rust and GPU programs.
/// Check before even looking for entry points, including in helper-only trees.
pub(crate) fn reject_cfg_attr(file: &syn::File) -> Result<(), Error> {
    use syn::visit::Visit;

    #[derive(Default)]
    struct ConditionalAttribute(Option<proc_macro2::Span>);
    impl<'ast> Visit<'ast> for ConditionalAttribute {
        fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
            if self.0.is_none() && attr.path().is_ident("cfg_attr") {
                self.0 = Some(attr.pound_token.span);
            }
        }
    }
    let mut visitor = ConditionalAttribute::default();
    visitor.visit_file(file);
    match visitor.0 {
        Some(span) => {
            let at = span.start();
            Err(Error::Pos {
                line: at.line,
                column: at.column + 1,
                source: Box::new(Error::CfgAttr),
            })
        }
        None => Ok(()),
    }
}

impl Cfg {
    /// Nothing holds.
    pub fn new() -> Self {
        Self::default()
    }

    /// What Cargo says holds. Kept as an alias of [`Cfg::from_cargo_env`].
    ///
    /// Earlier versions guessed `test` from `PROFILE=debug`, which is also
    /// the profile of an ordinary build. Rustc-only predicates now require an
    /// explicit choice, as described in [`Cfg::from_cargo_env`].
    pub fn from_cargo_env_agreeing_with_rustc() -> Self {
        Self::from_cargo_env()
    }

    /// What Cargo says holds for the crate being built: `debug_assertions` in a
    /// debug build, `feature = "..."` for each enabled feature, `target_os` and
    /// the rest. Only meaningful inside a build script.
    ///
    /// Cargo does not pass `test`, `doctest` or `miri`. Evaluating those is an
    /// error unless the caller chooses with [`Cfg::with`] or [`Cfg::without`].
    /// Prefer a Cargo feature when the shader and host must share the choice.
    pub fn from_cargo_env() -> Self {
        let mut cfg = Self::new();
        for (key, value) in std::env::vars() {
            let Some(name) = key.strip_prefix("CARGO_CFG_") else {
                continue;
            };
            let name = name.to_ascii_lowercase();
            if value.is_empty() {
                cfg.set.insert((name, None));
            } else {
                // Multi-valued options, `feature` and `target_feature` among
                // them, arrive comma-separated.
                for part in value.split(',') {
                    cfg.set.insert((name.clone(), Some(part.to_string())));
                }
            }
        }
        cfg.unavailable.extend(
            RUSTC_ONLY
                .iter()
                .filter(|&&name| !cfg.set.contains(&(name.into(), None)))
                .map(|&name| name.into()),
        );
        cfg
    }

    /// Make `name` hold, as `--cfg name` would.
    pub fn with(mut self, name: &str) -> Self {
        self.unavailable.remove(name);
        self.set.insert((name.to_string(), None));
        self
    }

    /// Make a bare predicate false, including one Cargo cannot decide.
    pub fn without(mut self, name: &str) -> Self {
        self.unavailable.remove(name);
        self.set.remove(&(name.to_string(), None));
        self
    }

    /// Make `name = "value"` hold, as `--cfg name="value"` would.
    pub fn with_value(mut self, name: &str, value: &str) -> Self {
        self.set.insert((name.to_string(), Some(value.to_string())));
        self
    }

    /// Does the predicate inside `cfg(...)` hold?
    pub(crate) fn eval(&self, meta: &syn::Meta) -> Result<bool, Error> {
        let unsupported = || Error::UnsupportedCfg(quote_meta(meta));
        match meta {
            syn::Meta::Path(path) => {
                let name = path.get_ident().ok_or_else(unsupported)?.to_string();
                if self.unavailable.contains(&name) {
                    return Err(Error::UnavailableCfg(name));
                }
                Ok(self.set.contains(&(name, None)))
            }
            syn::Meta::NameValue(pair) => {
                let name = pair.path.get_ident().ok_or_else(unsupported)?.to_string();
                let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(value),
                    ..
                }) = &pair.value
                else {
                    return Err(unsupported());
                };
                Ok(self.set.contains(&(name, Some(value.value()))))
            }
            syn::Meta::List(list) => {
                let args = list
                    .parse_args_with(
                        syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                    )
                    .map_err(|_| unsupported())?;
                let mut values = args.iter().map(|arg| self.eval(arg));
                if list.path.is_ident("all") {
                    values.try_fold(true, |all, value| Ok(all && value?))
                } else if list.path.is_ident("any") {
                    values.try_fold(false, |any, value| Ok(any || value?))
                } else if list.path.is_ident("not") && args.len() == 1 {
                    Ok(!self.eval(&args[0])?)
                } else {
                    Err(unsupported())
                }
            }
        }
    }

    /// Does an item with these attributes exist? It does unless one of them
    /// is a `#[cfg(...)]` that does not hold.
    pub(crate) fn keeps(&self, attrs: &[syn::Attribute]) -> Result<bool, Error> {
        for attr in attrs {
            if attr.path().is_ident("cfg") {
                let meta: syn::Meta = attr
                    .parse_args()
                    .map_err(|_| Error::UnsupportedCfg(quote_meta(&attr.meta)))?;
                if !self.eval(&meta)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

fn quote_meta(meta: &syn::Meta) -> String {
    quote::ToTokens::to_token_stream(meta).to_string()
}
