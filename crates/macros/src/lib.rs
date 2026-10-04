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

fn strip(attrs: &mut Vec<syn::Attribute>, names: &[&str]) {
    attrs.retain(|attr| !names.iter().any(|name| attr.path().is_ident(name)));
}

/// A shader entry point: `#[entry_point(vertex)]`, `#[entry_point(fragment)]`,
/// or `#[entry_point(compute, threads(8, 4))]`.
///
/// Its parameters may carry `#[location(N)]` and `#[builtin(name)]`, or be
/// named after the builtin they are: `global_invocation_id: Vec3<u32>`. A
/// bare value it returns is a vertex shader's position or a fragment shader's
/// `location(0)`. Anything else is returned in a struct that derives `Io`,
/// whose fields say where they go.
#[proc_macro_attribute]
pub fn entry_point(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(input as ItemFn);
    let args = parse_macro_input!(args with Punctuated::<Meta, Token![,]>::parse_terminated);
    let error = check_stage(&args)
        .and_then(|()| check_no_output(&item))
        .err()
        .map(|err| err.to_compile_error());

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

/// `#[output(..)]` was once how a function said where its result goes. The
/// result's type is where that belongs, and Rust has no attributes on types,
/// so a result that is not the stage's default is a struct of bound fields.
fn check_no_output(item: &ItemFn) -> syn::Result<()> {
    match item
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("output"))
    {
        Some(attr) => Err(syn::Error::new_spanned(
            attr,
            "a bare return value is a vertex shader's position or a fragment shader's \
             `location(0)`; return a struct that derives `Io` for anything else",
        )),
        None => Ok(()),
    }
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

/// A struct the host fills in and uploads, which a shader reads as it is:
/// `Clone`, `Copy`, a `Default` of all zeroes, which is the GPU's default
/// too, and `bytemuck`'s `Zeroable` and `Pod`, in one derive.
///
/// The struct says `#[repr(C)]`, which is also what tells the build to check
/// that the GPU reads each field where `rustc` puts it. `Pod` is checked as
/// `bytemuck`'s derive checks it: every field is `Pod`, and there is no
/// padding. It needs `synaga-shader`'s `bytemuck` feature.
///
/// A struct that holds an enum cannot be `Pod`, since not every `u32` is one
/// of its variants. `#[shared(no_uninit)]` makes it `bytemuck`'s `NoUninit`
/// instead, which is all an upload needs, with every field `NoUninit` and
/// `Zeroable`. The host then cannot read one back from bytes.
#[proc_macro_derive(Shared, attributes(shared))]
pub fn derive_shared(input: TokenStream) -> TokenStream {
    let item = parse_macro_input!(input as DeriveInput);
    match expand_shared(&item) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_shared(item: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    if !item.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.generics,
            "a `Shared` struct cannot be generic",
        ));
    }
    let Data::Struct(data) = &item.data else {
        return Err(syn::Error::new_spanned(
            &item.ident,
            "`Shared` is for a struct",
        ));
    };
    if !has_c_repr(&item.attrs) {
        return Err(syn::Error::new_spanned(
            &item.ident,
            "a `Shared` struct needs `#[repr(C)]`, the layout the host and the GPU agree on",
        ));
    }
    let name = &item.ident;
    let fields: Vec<&syn::Type> = data.fields.iter().map(|field| &field.ty).collect();
    let bytemuck = quote!(::synaga_shader::__private::bytemuck);
    // What the struct is to `bytemuck`, and a check that every field can make
    // it that, whose name is what `rustc` reports for a field that cannot.
    let (data_impl, field_check) = if no_uninit(&item.attrs)? {
        (
            quote! {
                // SAFETY: `#[repr(C)]`, every field `NoUninit`, and no padding
                // between or after them, all checked below.
                unsafe impl #bytemuck::NoUninit for #name {}
            },
            quote! {
                fn every_field_is_no_uninit_and_zeroable() {
                    fn no_uninit_and_zeroable<T: #bytemuck::NoUninit + #bytemuck::Zeroable>() {}
                    #( no_uninit_and_zeroable::<#fields>(); )*
                }
            },
        )
    } else {
        (
            quote! {
                // SAFETY: `#[repr(C)]`, every field `Pod`, and no padding
                // between or after them, all checked below.
                unsafe impl #bytemuck::Pod for #name {}
            },
            quote! {
                fn every_field_is_pod() {
                    fn pod<T: #bytemuck::Pod>() {}
                    #( pod::<#fields>(); )*
                }
            },
        )
    };
    Ok(quote! {
        impl ::core::clone::Clone for #name {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl ::core::marker::Copy for #name {}
        impl ::core::default::Default for #name {
            #[inline]
            fn default() -> Self {
                #bytemuck::Zeroable::zeroed()
            }
        }
        // SAFETY: every field is `Zeroable`, which `Pod` includes, as
        // checked below.
        unsafe impl #bytemuck::Zeroable for #name {}
        #data_impl
        impl ::synaga_shader::Shared for #name {}
        const _: () = {
            #field_check
            assert!(
                ::core::mem::size_of::<#name>() == 0 #( + ::core::mem::size_of::<#fields>() )*,
                "a `Shared` struct has no padding: add a field for it, as the build's layout check suggests",
            );
        };
    })
}

/// Whether a `Shared` struct says `#[shared(no_uninit)]`.
fn no_uninit(attrs: &[syn::Attribute]) -> syn::Result<bool> {
    let mut found = false;
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("shared")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("no_uninit") {
                found = true;
                Ok(())
            } else {
                Err(meta.error("`#[shared(..)]` takes `no_uninit`"))
            }
        })?;
    }
    Ok(found)
}

/// `#[repr(C)]` or `#[repr(transparent)]`, with anything beside it.
fn has_c_repr(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .filter(|attr| attr.path().is_ident("repr"))
        .any(|attr| {
            let mut found = false;
            let _ = attr.parse_nested_meta(|meta| {
                found |= meta.path.is_ident("C") || meta.path.is_ident("transparent");
                Ok(())
            });
            found
        })
}
