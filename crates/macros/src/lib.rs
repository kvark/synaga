//! Macros that make the shader dialect legal Rust.
//!
//! `#[location(0)]` and `#[builtin(position)]` mean nothing to `rustc`, which
//! rejects an attribute it cannot resolve. Two things can make one legal. A
//! derive can declare it as a helper attribute, which is how `#[derive(Io)]`
//! lets a struct's fields carry one. An attribute macro on the enclosing item
//! can take it off before `rustc` looks, which is how `#[entry_point]` handles
//! its parameters, where no derive reaches.
//!
//! Neither changes what the code means. The transpiler reads the same source
//! as text and gives the attributes their meaning; `rustc` only has to accept
//! them. So nothing here adds `unsafe`, and nothing turns a lint off except
//! `dead_code` on an entry point, whose caller is the GPU.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, Data, DeriveInput, Fields, FnArg, ItemFn, LitInt, Meta, Token};

/// Attributes on an entry point's parameters.
const PARAM_ATTRS: &[&str] = &["location", "builtin", "flat", "interpolate", "invariant"];

/// Attributes on an entry point itself, beside `#[entry_point]`.
const FN_ATTRS: &[&str] = &["output"];

fn strip(attrs: &mut Vec<syn::Attribute>, names: &[&str]) {
    attrs.retain(|attr| !names.iter().any(|name| attr.path().is_ident(name)));
}

/// A shader entry point: `#[entry_point(vertex)]`, `#[entry_point(fragment)]`,
/// or `#[entry_point(compute, threads(8, 4))]`.
///
/// Its parameters may carry `#[location(N)]` and `#[builtin(name)]`, or be
/// named after the builtin they are: `global_invocation_id: Vec3<u32>`. A
/// bare value it returns is a vertex shader's position or a fragment shader's
/// `location(0)`, unless `#[output(..)]` on the function says otherwise:
/// `#[output(builtin(frag_depth))]`.
#[proc_macro_attribute]
pub fn entry_point(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(input as ItemFn);
    let args = parse_macro_input!(args with Punctuated::<Meta, Token![,]>::parse_terminated);
    let error = check_stage(&args).err().map(|err| err.to_compile_error());

    strip(&mut item.attrs, FN_ATTRS);
    for arg in &mut item.sig.inputs {
        match arg {
            FnArg::Typed(pat) => strip(&mut pat.attrs, PARAM_ATTRS),
            FnArg::Receiver(receiver) => strip(&mut receiver.attrs, PARAM_ATTRS),
        }
    }

    // Nothing on the CPU calls an entry point, so `rustc` would call every one
    // of them dead.
    quote!(
        #error
        #[allow(dead_code)]
        #item
    )
    .into()
}

/// The stage comes first; a compute entry point also says its workgroup size.
/// The transpiler checks all of this again, but saying it here puts the
/// complaint where an editor shows it.
fn check_stage(args: &Punctuated<Meta, Token![,]>) -> syn::Result<()> {
    let mut args = args.iter();
    let stage = match args.next() {
        Some(Meta::Path(path))
            if ["vertex", "fragment", "compute"]
                .iter()
                .any(|stage| path.is_ident(stage)) =>
        {
            path
        }
        Some(other) => {
            return Err(syn::Error::new_spanned(
                other,
                "expected `vertex`, `fragment` or `compute`",
            ))
        }
        None => {
            return Err(syn::Error::new(
                Span::call_site(),
                "`#[entry_point]` needs a stage: `vertex`, `fragment` or `compute`",
            ))
        }
    };

    let mut threads = None;
    for arg in args {
        match arg {
            Meta::List(list) if list.path.is_ident("threads") => {
                if threads.is_some() {
                    return Err(syn::Error::new_spanned(list, "`threads` is given twice"));
                }
                let sizes =
                    list.parse_args_with(Punctuated::<LitInt, Token![,]>::parse_terminated)?;
                if sizes.is_empty() || sizes.len() > 3 {
                    return Err(syn::Error::new_spanned(
                        list,
                        "`threads` takes one to three sizes",
                    ));
                }
                for size in &sizes {
                    if size.base10_parse::<u32>()? == 0 {
                        return Err(syn::Error::new_spanned(
                            size,
                            "a workgroup size cannot be 0",
                        ));
                    }
                }
                threads = Some(list);
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "unknown `#[entry_point]` argument",
                ))
            }
        }
    }

    match (stage.is_ident("compute"), threads) {
        (true, None) => Err(syn::Error::new_spanned(
            stage,
            "a compute entry point needs its workgroup size: `threads(x, y, z)`",
        )),
        (false, Some(list)) => Err(syn::Error::new_spanned(
            list,
            "`threads` is only for a compute entry point",
        )),
        _ => Ok(()),
    }
}

/// A struct whose fields are all bound with `#[location(N)]` or
/// `#[builtin(name)]`: a vertex output, a fragment input, or a set of render
/// targets.
///
/// It declares those attributes, so `rustc` accepts them on the fields, and
/// checks that every field has one. The `synaga_shader::Io` it implements
/// reads every field, because the reader of an interface field is often the
/// rasterizer or a shader in another module, which `rustc` cannot see, and
/// "never read" would be wrong about most of them.
#[proc_macro_derive(Io, attributes(location, builtin, flat, interpolate, invariant))]
pub fn derive_io(input: TokenStream) -> TokenStream {
    let item = parse_macro_input!(input as DeriveInput);
    match check_io(&item) {
        Ok(fields) => {
            let name = &item.ident;
            // A trait from another crate, so `rustc` counts the impl as used
            // and its body as reading the fields.
            quote!(
                impl ::synaga_shader::Io for #name {
                    fn read_every_field(&self) {
                        let _ = (#(&self.#fields,)*);
                    }
                }
            )
            .into()
        }
        Err(err) => err.to_compile_error().into(),
    }
}

fn check_io(item: &DeriveInput) -> syn::Result<Vec<&syn::Ident>> {
    if !item.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.generics,
            "an `Io` struct cannot be generic",
        ));
    }
    let Data::Struct(data) = &item.data else {
        return Err(syn::Error::new_spanned(
            &item.ident,
            "`Io` is for a struct of bound fields",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            &item.ident,
            "`Io` needs named fields",
        ));
    };
    let mut names = Vec::new();
    for field in &fields.named {
        let bindings = field
            .attrs
            .iter()
            .filter(|attr| attr.path().is_ident("location") || attr.path().is_ident("builtin"))
            .count();
        if bindings != 1 {
            return Err(syn::Error::new_spanned(
                field,
                "each field needs exactly one `#[location(N)]` or `#[builtin(name)]`",
            ));
        }
        names.extend(field.ident.as_ref());
    }
    Ok(names)
}
