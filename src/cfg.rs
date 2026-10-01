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
}

/// The `cfg` predicates that are always set or always clear, whatever Cargo
/// said.
///
/// `CARGO_CFG_*` covers features, target and `debug_assertions`, but not the
/// flags Cargo only tells `rustc`: `test`, `doctest` and `miri` are decided
/// after the build script runs, and no `CARGO_CFG_*` says so. A shader written
/// `#[cfg(test)]` would be compiled by `rustc` under `cargo test` and dropped
/// here, which is the gap this closes.
///
/// `PROFILE` is the one signal Cargo does give a build script about it, since
/// it is passed as an environment variable. `test` and `debug_assertions` hold
/// together in practice — `cargo test` builds the test profile in debug mode —
/// so a `PROFILE` of `debug` is taken to mean the crate is being tested, and
/// `test` follows `debug_assertions`.
const TEST_PROFILES: [&str; 1] = ["debug"];

impl Cfg {
    /// Nothing holds.
    pub fn new() -> Self {
        Self::default()
    }

    /// What Cargo says holds, plus the flags Cargo does not pass on but that
    /// `rustc` sets anyway.
    ///
    /// See [`Cfg::from_cargo_env`] for the environment; this is that plus
    /// `#[cfg(test)]`, which a shader written for both the GPU and the CPU
    /// needs.
    pub fn from_cargo_env_agreeing_with_rustc() -> Self {
        let mut cfg = Self::from_cargo_env();
        if std::env::var("PROFILE").is_ok_and(|p| TEST_PROFILES.contains(&p.as_str())) {
            cfg.set.insert(("test".into(), None));
        }
        cfg
    }

    /// What Cargo says holds for the crate being built: `debug_assertions` in a
    /// debug build, `feature = "..."` for each enabled feature, `target_os` and
    /// the rest. Only meaningful inside a build script.
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
        cfg
    }

    /// Make `name` hold, as `--cfg name` would.
    pub fn with(mut self, name: &str) -> Self {
        self.set.insert((name.to_string(), None));
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
