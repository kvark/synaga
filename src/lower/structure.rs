use naga::{
    proc::Layouter, Block, Expression, Function, Handle, Span, StructMember, Type, TypeInner,
};
use syn::{Fields, ItemStruct};

use super::emit::emit;
use super::env::Env;
use super::expr::lower_expr_hinted;
use super::{Context, Typed};
use crate::Error;

pub(super) fn lower_struct_item(
    ctx: &mut Context,
    item: ItemStruct,
) -> Result<Handle<Type>, Error> {
    if !item.generics.params.is_empty() {
        return Err(Error::UnsupportedItem(format!(
            "generic struct `{}`",
            item.ident
        )));
    }
    let name = item.ident.to_string();
    let repr = host_repr(&item.attrs, &name)?;
    let derives_default = derives_default(&item.attrs);
    let named = match item.fields {
        Fields::Named(fields) => fields,
        Fields::Unnamed(_) => return Err(Error::UnsupportedItem("tuple struct".into())),
        Fields::Unit => return Err(Error::UnsupportedItem("unit struct".into())),
    };
    if named.named.is_empty() {
        return Err(Error::EmptyStruct(name));
    }

    let mut member_tys = Vec::new();
    let mut member_names = Vec::new();
    let mut member_bindings = Vec::new();
    for field in named.named {
        let fname = field
            .ident
            .as_ref()
            .ok_or_else(|| Error::UnsupportedItem("tuple field".into()))?
            .to_string();
        if member_names.iter().any(|n| n == &fname) {
            return Err(Error::DuplicateField(fname));
        }
        let ty = ctx.lower_type(&field.ty)?;
        let mut binding = super::entry::parse_io_binding(&field.attrs)?;
        if let Some(binding) = binding.as_mut() {
            super::entry::apply_default_interpolation(ctx, ty, binding);
        }
        member_names.push(fname);
        member_tys.push(ty);
        member_bindings.push(binding);
    }
    // A struct is either plain data or a shader interface, never half of each:
    // Naga rejects a partly bound entry-point struct, with a worse message.
    let bound = member_bindings.iter().filter(|b| b.is_some()).count();
    if bound != 0 && bound != member_bindings.len() {
        return Err(Error::MixedStructBindings(name));
    }

    let mut layouter = Layouter::default();
    layouter
        .update(ctx.module.to_ctx())
        .map_err(|e| Error::UnsupportedType(e.to_string()))?;

    let mut offset = 0u32;
    let mut struct_align = naga::proc::Alignment::ONE;
    let mut members = Vec::new();
    let fields = member_names
        .iter()
        .zip(member_tys.iter().copied())
        .zip(member_bindings);
    for ((fname, ty), binding) in fields {
        let layout = layouter[ty];
        offset = layout.alignment.round_up(offset);
        if layout.alignment > struct_align {
            struct_align = layout.alignment;
        }
        members.push(StructMember {
            name: Some(fname.clone()),
            ty,
            binding,
            offset,
        });
        offset += layout.size;
    }
    let span = struct_align.round_up(offset);

    let handle = ctx.module.types.insert(
        Type {
            name: Some(name),
            inner: TypeInner::Struct { members, span },
        },
        Span::UNDEFINED,
    );
    // An interface struct's fields are bindings, so it has no bytes to share.
    if let Some(repr) = repr.filter(|_| bound == 0) {
        ctx.host_reprs.insert(handle, repr);
    }
    if derives_default {
        ctx.derived_defaults.insert(handle);
    }
    Ok(handle)
}

/// Does a struct with `attrs` derive `Default`?
fn derives_default(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("derive"))
        .any(|attr| {
            let mut found = false;
            // A derive list that does not parse is `rustc`'s to report.
            let _ = attr.parse_nested_meta(|meta| {
                let last = meta.path.segments.last();
                found |= last.is_some_and(|s| s.ident == "Default");
                Ok(())
            });
            found
        })
}

/// The struct in `ty` whose `default()` may not be zero, if there is one.
///
/// `T::default()` lowers to the zero value, which is what a derived `Default`
/// gives when every field's is zero too, as a scalar's, vector's and
/// matrix's are. A struct's `Default` written by hand is in the host, where
/// the shader cannot see it, and need not be zero: a skinned vertex's weights
/// may default to all on its first joint.
pub(super) fn unseen_default(ctx: &Context, ty: Handle<Type>) -> Option<String> {
    match ctx.module.types[ty].inner {
        TypeInner::Array { base, .. } => unseen_default(ctx, base),
        TypeInner::Struct { ref members, .. } => {
            if !ctx.derived_defaults.contains(&ty) {
                return Some(ctx.module.types[ty].name.clone().unwrap_or_default());
            }
            members.iter().find_map(|m| unseen_default(ctx, m.ty))
        }
        _ => None,
    }
}

/// Check every `#[repr(C)]` struct in `ty`, the type of a buffer: the GPU
/// reads what the host wrote there, so the two have to agree on where each
/// field is. That is the only place they meet. A struct that only reaches
/// the GPU as a vertex's attributes is laid out by the vertex buffer's
/// format, which the host gives.
pub(super) fn check_shared(ctx: &mut Context, ty: Handle<Type>) -> Result<(), Error> {
    match ctx.module.types[ty].inner {
        TypeInner::Array { base, .. } | TypeInner::BindingArray { base, .. } => {
            check_shared(ctx, base)
        }
        TypeInner::Struct { ref members, span } => {
            let members = members.clone();
            // A field's own layout is settled before the struct's.
            for member in &members {
                check_shared(ctx, member.ty)?;
            }
            if let Some(&repr) = ctx.host_reprs.get(&ty) {
                if !ctx.host_layouts.contains_key(&ty) {
                    let name = ctx.module.types[ty].name.clone().unwrap_or_default();
                    let layout = check_host_layout(ctx, &name, &members, span, repr)?;
                    ctx.host_layouts.insert(ty, layout);
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// How `rustc` lays out a type, in bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostLayout {
    size: u32,
    align: u32,
}

/// What a struct's `#[repr]` says, when it says `C`: the host shares the
/// struct, so its layout has to be the GPU's. `align(N)` raises the
/// alignment, which also pads the size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostRepr {
    align: u32,
}

fn host_repr(attrs: &[syn::Attribute], name: &str) -> Result<Option<HostRepr>, Error> {
    let mut c = false;
    let mut align = 1;
    for attr in attrs.iter().filter(|a| a.path().is_ident("repr")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("C") {
                c = true;
            } else if meta.path.is_ident("align") {
                let content;
                syn::parenthesized!(content in meta.input);
                let n: syn::LitInt = content.parse()?;
                align = align.max(n.base10_parse()?);
            } else if meta.path.is_ident("packed") {
                return Err(meta.error("the GPU has no packed layout"));
            }
            Ok(())
        })
        .map_err(|e| Error::HostLayout(name.into(), e.to_string()))?;
    }
    Ok(c.then_some(HostRepr { align }))
}

/// `members`, laid out the way `rustc` lays out a `#[repr(C)]` struct, have
/// to land where the GPU puts them. A difference would have the host write
/// a field where the shader reads another, which nothing else would catch.
fn check_host_layout(
    ctx: &Context,
    name: &str,
    members: &[StructMember],
    span: u32,
    repr: HostRepr,
) -> Result<HostLayout, Error> {
    let fail = |detail: String| Error::HostLayout(name.into(), detail);
    let mut offset = 0u32;
    let mut align = repr.align;
    let mut sized = true;
    for member in members {
        let field = member.name.as_deref().unwrap_or_default();
        let layout = match &ctx.module.types[member.ty].inner {
            // `[T]` ends a struct, in Rust as on the GPU, and has no size.
            TypeInner::Array {
                base,
                size: naga::ArraySize::Dynamic,
                ..
            } => {
                sized = false;
                let element = host_layout(ctx, *base)
                    .map_err(|why| fail(format!("the elements of `{field}` are {why}")))?;
                HostLayout {
                    size: 0,
                    align: element.align,
                }
            }
            _ => host_layout(ctx, member.ty).map_err(|why| fail(format!("`{field}` is {why}")))?,
        };
        offset = offset.next_multiple_of(layout.align);
        if offset != member.offset {
            let pad = match member.offset.checked_sub(offset) {
                Some(bytes) => format!("; {bytes} bytes of padding before it line them up"),
                None => String::new(),
            };
            return Err(fail(format!(
                "`{field}` is at byte {offset} in Rust and {} on the GPU{pad}",
                member.offset
            )));
        }
        offset += layout.size;
        align = align.max(layout.align);
    }
    let size = offset.next_multiple_of(align);
    if sized && size != span {
        let pad = match span.checked_sub(size) {
            Some(bytes) => format!("; {bytes} bytes of padding at the end line them up"),
            None => String::new(),
        };
        return Err(fail(format!(
            "it is {size} bytes in Rust and {span} on the GPU{pad}"
        )));
    }
    Ok(HostLayout { size, align })
}

/// The layout `rustc` gives the `synaga-shader` type that stands for `ty`,
/// when it is the GPU's too, so that only where a field goes can differ.
/// `Err` says what about the type differs.
fn host_layout(ctx: &Context, ty: Handle<Type>) -> Result<HostLayout, String> {
    use naga::{ArraySize, ScalarKind, VectorSize};
    let named = || match &ctx.module.types[ty].name {
        Some(name) => format!("`{name}`"),
        None => "this type".into(),
    };
    let scalar = |scalar: naga::Scalar| match scalar.kind {
        ScalarKind::Bool => Err("a `bool`, which the GPU has no layout for".to_string()),
        _ => Ok(u32::from(scalar.width)),
    };
    Ok(match ctx.module.types[ty].inner {
        TypeInner::Scalar(s) | TypeInner::Atomic(s) => {
            let width = scalar(s)?;
            HostLayout {
                size: width,
                align: width,
            }
        }
        TypeInner::Vector { size, scalar: s } => {
            let width = scalar(s)?;
            HostLayout {
                size: size as u32 * width,
                align: width,
            }
        }
        // Column after column. A 3-lane column takes four lanes, on the GPU
        // and in `synaga-shader`'s matrices alike.
        TypeInner::Matrix {
            columns,
            rows,
            scalar: s,
        } => {
            let width = scalar(s)?;
            let lanes = match rows {
                VectorSize::Tri => 4,
                other => other as u32,
            };
            HostLayout {
                size: columns as u32 * lanes * width,
                align: width,
            }
        }
        TypeInner::Array {
            base,
            size: ArraySize::Constant(len),
            stride,
        } => {
            let element = host_layout(ctx, base)?;
            if element.size != stride {
                return Err(format!(
                    "an array whose elements are {} bytes apart in Rust and {stride} on the GPU",
                    element.size
                ));
            }
            HostLayout {
                size: element.size * len.get(),
                align: element.align,
            }
        }
        TypeInner::Struct { .. } => match ctx.host_layouts.get(&ty) {
            Some(&layout) => layout,
            None => return Err(format!("{}, which is not `#[repr(C)]`", named())),
        },
        _ => return Err(format!("{}, which the host cannot share", named())),
    })
}

pub(super) fn lower_struct_lit(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    lit: &syn::ExprStruct,
    env: &mut Env,
) -> Result<Typed, Error> {
    if lit.qself.is_some() {
        return Err(Error::UnsupportedExpr("struct literal".into()));
    }
    let path = super::path_segments(&lit.path);
    let name = super::last(&path);
    let ty = ctx
        .named_type(&path)?
        .ok_or_else(|| Error::UnknownStruct(name.clone()))?;
    let members = ctx
        .as_struct(ty)
        .ok_or_else(|| Error::UnknownStruct(name.clone()))?;
    let expected: Vec<(String, Handle<Type>)> = members
        .iter()
        .map(|m| (m.name.clone().unwrap_or_default(), m.ty))
        .collect();

    let mut provided: Vec<(String, Handle<Expression>, Handle<Type>)> = Vec::new();
    for field in &lit.fields {
        let fname = match &field.member {
            syn::Member::Named(ident) => ident.to_string(),
            syn::Member::Unnamed(_) => return Err(Error::UnsupportedExpr("tuple field".into())),
        };
        // The member's type is known, so an untyped integer literal can follow
        // it the way it follows a parameter type at a call.
        let hint = expected
            .iter()
            .find(|(n, _)| *n == fname)
            .and_then(|&(_, ty)| ctx.shape(ty).int_hint());
        let (expr, fty) = lower_expr_hinted(ctx, function, body, &field.expr, env, hint)?;
        if provided.iter().any(|(n, _, _)| n == &fname) {
            return Err(Error::DuplicateField(fname));
        }
        provided.push((fname, expr, fty));
    }

    // `..Default::default()` leaves the other fields zero, which is what a
    // shader struct's `Default` is; `..other` takes them from `other`.
    let rest = match lit.rest.as_deref() {
        None if provided.len() != expected.len() => {
            return Err(Error::StructFieldCount(name));
        }
        None => Rest::None,
        Some(rest) if is_default(rest, &name) => {
            // Only the fields left out come from the default, but `rustc`
            // calls the struct's own `default()` for them.
            let unseen = if ctx.derived_defaults.contains(&ty) {
                expected
                    .iter()
                    .filter(|(n, _)| !provided.iter().any(|(p, _, _)| p == n))
                    .find_map(|&(_, want_ty)| unseen_default(ctx, want_ty))
            } else {
                Some(name.clone())
            };
            if let Some(unseen) = unseen {
                return Err(Error::UnseenDefault(name, unseen));
            }
            Rest::Zero
        }
        Some(rest) => {
            let (base, base_ty) = super::expr::lower_expr(ctx, function, body, rest, env)?;
            if base_ty != ty {
                return Err(Error::TypeMismatch);
            }
            Rest::From(base)
        }
    };

    let mut components = Vec::with_capacity(expected.len());
    for (index, (want_name, want_ty)) in expected.iter().enumerate() {
        let component = match (provided.iter().find(|(n, _, _)| n == want_name), &rest) {
            (Some(&(_, value, ty)), _) if ty == *want_ty => value,
            (Some(_), _) => return Err(Error::TypeMismatch),
            (None, Rest::Zero) => function
                .expressions
                .append(Expression::ZeroValue(*want_ty), Span::UNDEFINED),
            (None, &Rest::From(base)) => {
                let index = index as u32;
                emit(function, body, Expression::AccessIndex { base, index })?
            }
            (None, Rest::None) => return Err(Error::MissingStructField(want_name.clone())),
        };
        components.push(component);
    }

    let handle = emit(function, body, Expression::Compose { ty, components })?;
    Ok((handle, ty))
}

/// Where the fields a struct literal leaves out come from.
enum Rest {
    None,
    Zero,
    From(Handle<Expression>),
}

/// `Default::default()` or `Name::default()`, for the struct `name`.
fn is_default(expr: &syn::Expr, name: &str) -> bool {
    let syn::Expr::Call(call) = expr else {
        return false;
    };
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return false;
    };
    let segments = super::path_segments(&path.path);
    call.args.is_empty()
        && matches!(&segments[..], [.., ty, f] if f == "default" && (ty == "Default" || ty == name))
}

pub(super) fn lower_struct_field(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    base: Handle<Expression>,
    base_ty: Handle<Type>,
    member: &str,
) -> Result<Typed, Error> {
    let members = ctx
        .as_struct(base_ty)
        .ok_or_else(|| Error::UnsupportedExpr("field".into()))?;
    let (index, field_ty) = members
        .iter()
        .enumerate()
        .find_map(|(i, m)| {
            m.name
                .as_deref()
                .filter(|n| *n == member)
                .map(|_| (i as u32, m.ty))
        })
        .ok_or_else(|| Error::UnknownField(member.into()))?;
    let handle = emit(function, body, Expression::AccessIndex { base, index })?;
    Ok((handle, field_ty))
}
