//! Items and the names they are reached by.
//!
//! Rust does not care in what order a module declares its items, or which
//! file of the crate an item is in, as long as a `use` or a path says where to
//! look. A shader module is Rust, so the transpiler follows the same rules.
//! Every item of every source is indexed before anything is lowered, and an
//! item is lowered when something first needs it. A name is resolved the way
//! `rustc` resolves it: the module's own items first, then what it imports by
//! name, then what it imports with a glob.
//!
//! One difference: an unqualified name nothing imports still resolves, if
//! exactly one source declares it. A source handed over without a module
//! name, as in a test or through [`crate::parse_all`], has no other way to
//! reach its neighbours. `rustc` has already refused anything that relies on
//! this in a real crate.

use std::collections::HashMap;

use naga::{Function, Handle, Type};
use syn::{Item, UseTree};

use crate::{Cfg, Error};

/// Rust keeps types apart from values, so a struct and a function may share a
/// name.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Ns {
    Type,
    Value,
}

/// What an item became, once lowered.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Lowered {
    Function(Handle<Function>),
    EntryPoint,
    Type(Handle<Type>),
    /// Index into `Context::consts`.
    Const(usize),
    /// Statics bind by name as the function starts; nothing else to record.
    Static,
    /// `use`, `mod` and the like, which declare nothing to lower.
    Nothing,
}

pub(crate) enum State {
    Pending,
    InProgress,
    Done(Lowered),
}

pub(crate) struct Entry {
    pub source: usize,
    /// `None` for an item nothing refers to by name: an `extern` block, say.
    pub name: Option<(Ns, String)>,
    /// Taken out while the item is lowered, so the lowering can borrow the
    /// context.
    pub item: Option<Item>,
    pub state: State,
}

impl Entry {
    pub fn is_static_or_const(&self) -> bool {
        matches!(
            self.item,
            Some(Item::Static(_) | Item::Const(_) | Item::ForeignMod(_))
        )
    }
}

#[derive(Default)]
struct SourceScope {
    /// The module name the source is reached by: its file stem.
    name: Option<String>,
    /// `use a::b::c as d;` maps `d` to `[a, b, c]`.
    imports: HashMap<String, Vec<String>>,
    /// `use a::b::*;` adds `[a, b]`.
    globs: Vec<Vec<String>>,
}

#[derive(Default)]
pub(crate) struct Scope {
    sources: Vec<SourceScope>,
    pub entries: Vec<Entry>,
    by_name: HashMap<(Ns, String), Vec<usize>>,
    /// `Mode::Variance` is the discriminant, as a `u32`.
    variants: HashMap<(String, String), u32>,
    /// The enums, by name, which are a `u32` on the GPU.
    pub enums: HashMap<String, super::nominal::EnumInfo>,
    /// The `bitflags!` sets, by name, which are too.
    pub flags: HashMap<String, super::nominal::FlagsInfo>,
    /// What each `type` alias stands for, as written, so that a check on the
    /// Rust type can see through it after the alias is lowered.
    aliases: HashMap<String, syn::Type>,
}

impl Scope {
    /// The `u32` discriminant of `Enum::Variant`, if that enum was declared.
    pub fn enum_variant(&self, enumeration: &str, variant: &str) -> Option<u32> {
        self.variants
            .get(&(enumeration.to_string(), variant.to_string()))
            .copied()
    }

    /// The pointer-sized integer in `ty`, through arrays, type arguments and
    /// aliases. `usize` is as wide as a pointer on the host and a `u32` on the
    /// GPU, so a host that shares one cannot agree with the GPU on it.
    pub fn pointer_sized(&self, ty: &syn::Type) -> Option<&'static str> {
        self.pointer_sized_in(ty, 0)
    }

    fn pointer_sized_in(&self, ty: &syn::Type, depth: usize) -> Option<&'static str> {
        if depth > 16 {
            return None;
        }
        match ty {
            syn::Type::Path(path) => {
                let last = path.path.segments.last()?;
                match last.ident.to_string().as_str() {
                    "usize" => return Some("usize"),
                    "isize" => return Some("isize"),
                    name => {
                        let aliased = self.aliases.get(name);
                        if let Some(found) =
                            aliased.and_then(|t| self.pointer_sized_in(t, depth + 1))
                        {
                            return Some(found);
                        }
                    }
                }
                let syn::PathArguments::AngleBracketed(args) = &last.arguments else {
                    return None;
                };
                args.args.iter().find_map(|arg| match arg {
                    syn::GenericArgument::Type(inner) => self.pointer_sized_in(inner, depth + 1),
                    _ => None,
                })
            }
            syn::Type::Array(array) => self.pointer_sized_in(&array.elem, depth + 1),
            syn::Type::Slice(slice) => self.pointer_sized_in(&slice.elem, depth + 1),
            syn::Type::Paren(inner) => self.pointer_sized_in(&inner.elem, depth + 1),
            syn::Type::Group(inner) => self.pointer_sized_in(&inner.elem, depth + 1),
            _ => None,
        }
    }

    /// Index the items of `files`, dropping any whose `#[cfg]` does not hold.
    pub fn index(files: Vec<(Option<String>, syn::File)>, cfg: &Cfg) -> Result<Self, IndexError> {
        let mut scope = Scope::default();
        for (source, (name, file)) in files.into_iter().enumerate() {
            scope.sources.push(SourceScope {
                name,
                ..Default::default()
            });
            for item in file.items {
                let keep = cfg
                    .keeps(item_attrs(&item))
                    .map_err(|error| IndexError { source, error })?;
                if !keep {
                    continue;
                }
                if let Item::Use(item_use) = &item {
                    let scope_of = &mut scope.sources[source];
                    collect_use(&item_use.tree, Vec::new(), scope_of);
                    continue;
                }
                if let Item::Enum(enumeration) = &item {
                    let variants =
                        enum_variants(enumeration).map_err(|error| IndexError { source, error })?;
                    let info = super::nominal::enum_info(enumeration, &variants);
                    let name = enumeration.ident.to_string();
                    for (variant, value) in variants {
                        scope.variants.insert((name.clone(), variant), value);
                    }
                    scope.enums.insert(name, info);
                    continue;
                }
                // `bitflags!` declares sets, which are types the shader reads
                // rather than items it lowers. Any other macro is an error when
                // it is lowered.
                if let Item::Macro(item_macro) = &item {
                    if super::nominal::is_bitflags(&item_macro.mac) {
                        let sets = super::nominal::parse_bitflags(&item_macro.mac)
                            .map_err(|error| IndexError { source, error })?;
                        scope.flags.extend(sets);
                        continue;
                    }
                }
                if let Item::Type(alias) = &item {
                    scope
                        .aliases
                        .insert(alias.ident.to_string(), (*alias.ty).clone());
                }
                let name = item_name(&item);
                if let Some(key) = &name {
                    let clash = scope.by_name.get(key).is_some_and(|found| {
                        found.iter().any(|&i| scope.entries[i].source == source)
                    });
                    if clash {
                        return Err(IndexError {
                            source,
                            error: duplicate(&item, key.1.clone()),
                        });
                    }
                    scope
                        .by_name
                        .entry(key.clone())
                        .or_default()
                        .push(scope.entries.len());
                }
                scope.entries.push(Entry {
                    source,
                    name,
                    item: Some(item),
                    state: State::Pending,
                });
            }
        }
        scope.find_flags_newtypes()?;
        Ok(scope)
    }

    /// `bitflags! { impl Flags: u32 { .. } }` puts the flags on a struct the
    /// sources declare. Its `#[repr]` and derives are on that struct, which
    /// has to be a `u32` and nothing else.
    fn find_flags_newtypes(&mut self) -> Result<(), IndexError> {
        for (name, info) in self.flags.iter_mut().filter(|(_, info)| info.external) {
            let found = self.entries.iter().find_map(|entry| match &entry.item {
                Some(Item::Struct(item)) if item.ident == name.as_str() => {
                    Some((entry.source, item))
                }
                _ => None,
            });
            let Some((source, item)) = found else {
                return Err(IndexError {
                    source: 0,
                    error: Error::Bitflags(format!(
                        "`impl {name}: u32` is for a `struct {name}(u32)` the shader declares"
                    )),
                });
            };
            let is_u32 =
                |ty: &syn::Type| matches!(ty, syn::Type::Path(p) if p.path.is_ident("u32"));
            let newtype = match &item.fields {
                syn::Fields::Unnamed(fields) => {
                    fields.unnamed.len() == 1 && is_u32(&fields.unnamed[0].ty)
                }
                _ => false,
            };
            if !newtype {
                return Err(IndexError {
                    source,
                    error: Error::FlagsRepr(name.clone()),
                });
            }
            info.transparent = super::nominal::has_repr(&item.attrs, "transparent");
            info.derives_default = super::structure::derives_default(&item.attrs);
        }
        Ok(())
    }

    fn source_named(&self, name: &str) -> Option<usize> {
        self.sources
            .iter()
            .position(|s| s.name.as_deref() == Some(name))
    }

    /// The module `name` stands for inside source `from`: a sibling source,
    /// possibly under a name a `use` gave it.
    fn module(&self, from: usize, name: &str) -> Option<usize> {
        if let Some(target) = self.sources[from].imports.get(name) {
            let target = strip_relative(target);
            if let [.., last] = target {
                if let Some(found) = self.source_named(last) {
                    return Some(found);
                }
            }
        }
        self.source_named(name)
    }

    /// Is `path`, seen from source `from`, headed by a module rather than a
    /// type? `brdf::sample` is an item in a module; `vec3::splat` is a
    /// function on a type.
    pub fn is_module_path(&self, from: usize, path: &[String]) -> bool {
        let rest = strip_relative(path);
        rest.len() >= 2 && self.module(from, &rest[rest.len() - 2]).is_some()
            || rest.len() < path.len()
    }

    /// Where `name`, as seen from source `from`, comes from outside the
    /// sources: `PI` after `use core::f32::consts::PI`, or after a `use` of a
    /// module that has that `use`. The path is the outside one.
    pub fn external_path(&self, from: usize, name: &str) -> Option<Vec<String>> {
        self.external_path_in(from, name, 0)
    }

    fn external_path_in(&self, from: usize, name: &str, depth: usize) -> Option<Vec<String>> {
        if depth > 16 {
            return None;
        }
        let scope = &self.sources[from];
        if let Some(target) = scope.imports.get(name) {
            let (last, modules) = strip_relative(target).split_last()?;
            return match modules.last().and_then(|m| self.module(from, m)) {
                Some(source) => self.external_path_in(source, last, depth + 1),
                None => Some(target.clone()),
            };
        }
        scope.globs.iter().find_map(|glob| {
            match strip_relative(glob)
                .last()
                .and_then(|m| self.module(from, m))
            {
                Some(source) => self.external_path_in(source, name, depth + 1),
                None => Some(glob.iter().cloned().chain([name.to_string()]).collect()),
            }
        })
    }

    /// Which entry `path` names in namespace `ns`, as seen from source `from`.
    pub fn resolve(&self, from: usize, ns: Ns, path: &[String]) -> Result<Option<usize>, Error> {
        self.resolve_in(from, ns, path, 0, true)
    }

    /// `anywhere` allows the last resort, a name only one source declares.
    /// It is for a name as the source wrote it; looking inside a module a
    /// path or an import points at has to find the item there or nowhere.
    fn resolve_in(
        &self,
        from: usize,
        ns: Ns,
        path: &[String],
        depth: usize,
        anywhere: bool,
    ) -> Result<Option<usize>, Error> {
        // A `use` cycle is a `rustc` error; this only keeps it from hanging.
        if depth > 16 {
            return Ok(None);
        }
        let relative = strip_relative(path);
        let ours = relative.len() < path.len();
        let path = relative;
        let Some((name, modules)) = path.split_last() else {
            return Ok(None);
        };
        let key = (ns, name.clone());
        let found = self.by_name.get(&key).map(Vec::as_slice).unwrap_or(&[]);
        if let Some(module) = modules.last() {
            if let Some(source) = self.module(from, module) {
                return self.resolve_in(source, ns, std::slice::from_ref(name), depth + 1, false);
            }
            // `super::common::f` is one of ours even when the sources were
            // handed over without names, so it is found by its own name. A
            // path into another crate, such as `synaga_shader::vec3`, is for
            // the caller to make sense of.
            return match (ours, found) {
                (true, [one]) => Ok(Some(*one)),
                (true, [_, _, ..]) => Err(Error::AmbiguousName(name.clone())),
                _ => Ok(None),
            };
        }

        if let Some(&own) = found.iter().find(|&&i| self.entries[i].source == from) {
            return Ok(Some(own));
        }
        let scope = &self.sources[from];
        if let Some(target) = scope.imports.get(name) {
            return self.resolve_in(from, ns, target, depth + 1, anywhere);
        }
        let mut through_globs = Vec::new();
        for glob in &scope.globs {
            let glob = strip_relative(glob);
            let Some(module) = glob.last().and_then(|m| self.module(from, m)) else {
                continue;
            };
            if let Some(i) =
                self.resolve_in(module, ns, std::slice::from_ref(name), depth + 1, false)?
            {
                if !through_globs.contains(&i) {
                    through_globs.push(i);
                }
            }
        }
        match through_globs[..] {
            [one] => return Ok(Some(one)),
            [_, _, ..] => return Err(Error::AmbiguousName(name.clone())),
            [] => {}
        }
        match found {
            _ if !anywhere => Ok(None),
            [] => Ok(None),
            [one] => Ok(Some(*one)),
            _ => Err(Error::AmbiguousName(name.clone())),
        }
    }
}

/// A failure while indexing, and which source it came from.
pub(crate) struct IndexError {
    pub source: usize,
    pub error: Error,
}

/// `crate::`, `self::` and `super::` say where to start looking. Sources are
/// siblings, so every start leads to the same set of them.
fn strip_relative(path: &[String]) -> &[String] {
    let skip = path
        .iter()
        .take_while(|s| matches!(s.as_str(), "crate" | "self" | "super"))
        .count();
    &path[skip..]
}

fn collect_use(tree: &UseTree, mut prefix: Vec<String>, scope: &mut SourceScope) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_use(&path.tree, prefix, scope);
        }
        UseTree::Name(name) => {
            let ident = name.ident.to_string();
            // `use a::b::{self}` imports the module `b` itself.
            if ident == "self" {
                if let Some(last) = prefix.last().cloned() {
                    scope.imports.insert(last, prefix);
                }
                return;
            }
            prefix.push(ident.clone());
            scope.imports.insert(ident, prefix);
        }
        UseTree::Rename(rename) => {
            let ident = rename.ident.to_string();
            if ident != "self" {
                prefix.push(ident);
            }
            scope.imports.insert(rename.rename.to_string(), prefix);
        }
        UseTree::Glob(_) => scope.globs.push(prefix),
        UseTree::Group(group) => {
            for tree in &group.items {
                collect_use(tree, prefix.clone(), scope);
            }
        }
    }
}

/// Fieldless variants become `u32` discriminants. An omitted one is one past
/// the previous, starting at zero, as in Rust.
fn enum_variants(item: &syn::ItemEnum) -> Result<Vec<(String, u32)>, Error> {
    let mut next = 0u32;
    let mut variants = Vec::with_capacity(item.variants.len());
    for variant in &item.variants {
        if !matches!(variant.fields, syn::Fields::Unit) {
            return Err(Error::UnsupportedItem(format!(
                "enum variant `{}::{}` with fields",
                item.ident, variant.ident
            )));
        }
        if let Some((_, expr)) = &variant.discriminant {
            next = enum_discriminant(expr)?;
        }
        variants.push((variant.ident.to_string(), next));
        next = next.saturating_add(1);
    }
    Ok(variants)
}

fn enum_discriminant(expr: &syn::Expr) -> Result<u32, Error> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(value),
            ..
        }) => value.base10_parse().map_err(Error::from),
        syn::Expr::Paren(inner) => enum_discriminant(&inner.expr),
        syn::Expr::Group(inner) => enum_discriminant(&inner.expr),
        _ => Err(Error::UnsupportedConstExpr(
            "enum discriminant must be an integer literal".into(),
        )),
    }
}

fn item_name(item: &Item) -> Option<(Ns, String)> {
    Some(match item {
        Item::Fn(f) => (Ns::Value, f.sig.ident.to_string()),
        Item::Const(c) => (Ns::Value, c.ident.to_string()),
        Item::Static(s) => (Ns::Value, s.ident.to_string()),
        Item::Struct(s) => (Ns::Type, s.ident.to_string()),
        Item::Type(t) => (Ns::Type, t.ident.to_string()),
        _ => return None,
    })
}

fn duplicate(item: &Item, name: String) -> Error {
    match item {
        Item::Fn(_) => Error::DuplicateFunction(name),
        Item::Const(_) => Error::DuplicateConst(name),
        Item::Static(_) => Error::DuplicateGlobal(name),
        _ => Error::DuplicateStruct(name),
    }
}

pub(super) fn item_attrs(item: &Item) -> &[syn::Attribute] {
    match item {
        Item::Const(i) => &i.attrs,
        Item::Enum(i) => &i.attrs,
        Item::ExternCrate(i) => &i.attrs,
        Item::Fn(i) => &i.attrs,
        Item::ForeignMod(i) => &i.attrs,
        Item::Impl(i) => &i.attrs,
        Item::Macro(i) => &i.attrs,
        Item::Mod(i) => &i.attrs,
        Item::Static(i) => &i.attrs,
        Item::Struct(i) => &i.attrs,
        Item::Trait(i) => &i.attrs,
        Item::TraitAlias(i) => &i.attrs,
        Item::Type(i) => &i.attrs,
        Item::Union(i) => &i.attrs,
        Item::Use(i) => &i.attrs,
        _ => &[],
    }
}
