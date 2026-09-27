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

    Ok(ctx.module.types.insert(
        Type {
            name: Some(name),
            inner: TypeInner::Struct { members, span },
        },
        Span::UNDEFINED,
    ))
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
    if lit.rest.is_some() {
        return Err(Error::UnsupportedExpr("struct rest `..`".into()));
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

    if provided.len() != expected.len() {
        return Err(Error::StructFieldCount(name));
    }

    let mut components = Vec::with_capacity(expected.len());
    for (want_name, want_ty) in &expected {
        let found = provided
            .iter()
            .find(|(n, _, _)| n == want_name)
            .ok_or_else(|| Error::MissingStructField(want_name.clone()))?;
        if found.2 != *want_ty {
            return Err(Error::TypeMismatch);
        }
        components.push(found.1);
    }

    let handle = emit(function, body, Expression::Compose { ty, components })?;
    Ok((handle, ty))
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
