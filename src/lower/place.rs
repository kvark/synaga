//! Place expressions: the ones that name storage rather than produce a value.
//!
//! `s.a`, `v.x` and `a[i]` can be reached two ways. Loading the whole aggregate
//! and picking a component out of the value works for reading, but it reads an
//! entire uniform buffer to get one field, and it cannot be written through at
//! all. Walking the pointer instead — `AccessIndex` on `ptr<uniform, Camera>`
//! rather than on a loaded `Camera` — loads only what is asked for, and gives
//! `s.a = x` something to store to.
//!
//! Not everything is a place: a function argument is a value, and so is the
//! result of a call or a swizzle. Those fall back to `expr.rs`.

use naga::{Expression, Function, Handle, Scalar, Type};
use syn::Expr;

use super::emit::emit;
use super::env::{Env, Slot};
use super::expr::lower_expr;
use super::{Context, Shape};
use crate::Error;

pub(super) struct Place {
    pub pointer: Handle<Expression>,
    pub ty: Handle<Type>,
    /// False for a global in a read-only address space, or a `&T` parameter.
    pub writable: bool,
    /// Where the storage lives, which a pointer to it has to agree with.
    pub space: naga::AddressSpace,
    /// The binding the chain started from, for error messages.
    pub root: String,
}

/// Resolve `expr` to a pointer, or `None` if it does not name storage.
pub(super) fn lower_place(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut naga::Block,
    expr: &Expr,
    env: &mut Env,
) -> Result<Option<Place>, Error> {
    match expr {
        Expr::Paren(inner) => lower_place(ctx, function, body, &inner.expr, env),
        Expr::Group(inner) => lower_place(ctx, function, body, &inner.expr, env),
        // `*slot` is the slot. Rust writes it to dereference a resource wrapper.
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
            lower_place(ctx, function, body, &unary.expr, env)
        }
        Expr::Path(path) => {
            let Some(ident) = path.path.get_ident() else {
                return Ok(None);
            };
            let name = ident.to_string();
            let Some(binding) = env.lookup(&name) else {
                return Ok(None);
            };
            match binding.slot {
                // A function argument is a value; there is nothing to point at.
                Slot::Value(_) => Ok(None),
                Slot::Ptr(pointer) => Ok(Some(Place {
                    pointer,
                    ty: binding.ty,
                    writable: binding.writable,
                    space: binding.space,
                    root: name,
                })),
            }
        }
        Expr::Field(field) => {
            let Some(base) = lower_place(ctx, function, body, &field.base, env)? else {
                return Ok(None);
            };
            let syn::Member::Named(ident) = &field.member else {
                return Ok(None);
            };
            let Some((index, ty)) = component(ctx, base.ty, &ident.to_string()) else {
                return Ok(None);
            };
            let pointer = emit(
                function,
                body,
                Expression::AccessIndex {
                    base: base.pointer,
                    index,
                },
            )?;
            Ok(Some(Place {
                pointer,
                ty,
                ..base
            }))
        }
        Expr::Index(index) => {
            let Some(base) = lower_place(ctx, function, body, &index.expr, env)? else {
                return Ok(None);
            };
            let Some((bound, ty)) = element(ctx, base.ty) else {
                return Ok(None);
            };
            let pointer = match index_expr(ctx, function, body, &index.index, bound, env)? {
                IndexKind::Constant(index) => emit(
                    function,
                    body,
                    Expression::AccessIndex {
                        base: base.pointer,
                        index,
                    },
                )?,
                IndexKind::Dynamic(index) => emit(
                    function,
                    body,
                    Expression::Access {
                        base: base.pointer,
                        index,
                    },
                )?,
            };
            Ok(Some(Place {
                pointer,
                ty,
                ..base
            }))
        }
        _ => Ok(None),
    }
}

/// The index and type of `name` within `ty`: a struct field, or a single
/// vector component. Multi-letter swizzles are values, not places, and are not
/// reachable this way.
pub(super) fn component(
    ctx: &mut Context,
    ty: Handle<Type>,
    name: &str,
) -> Option<(u32, Handle<Type>)> {
    if let Some(members) = ctx.as_struct(ty) {
        return members
            .iter()
            .enumerate()
            .find(|(_, m)| m.name.as_deref() == Some(name))
            .map(|(i, m)| (i as u32, m.ty));
    }
    let Shape::Vector(size, scalar) = ctx.shape(ty) else {
        return None;
    };
    let index = vector_component(name)?;
    if index >= size as u32 {
        return None;
    }
    Some((index, ctx.intern_scalar(scalar)))
}

/// WGSL names vector components twice over: `xyzw` and `rgba`. Both are here,
/// though WGSL does not let a single swizzle mix the two sets.
pub(super) fn vector_component(name: &str) -> Option<u32> {
    match name {
        "x" | "r" => Some(0),
        "y" | "g" => Some(1),
        "z" | "b" => Some(2),
        "w" | "a" => Some(3),
        _ => None,
    }
}

/// Which of the two naming sets a component letter belongs to, so a swizzle
/// can be held to one of them.
fn component_set(letter: char) -> Option<bool> {
    match letter {
        'x' | 'y' | 'z' | 'w' => Some(false),
        'r' | 'g' | 'b' | 'a' => Some(true),
        _ => None,
    }
}

/// Component indices for a swizzle, rejecting a mix of `xyzw` and `rgba`.
pub(super) fn swizzle_components(member: &str) -> Option<Vec<u32>> {
    let mut sets = member.chars().map(component_set);
    let first = sets.next()??;
    if !sets.all(|set| set == Some(first)) {
        return None;
    }
    member
        .chars()
        .map(|c| vector_component(&c.to_string()))
        .collect()
}

/// The bound and element type of an indexable `ty`. A runtime-sized array has
/// no bound to check a literal index against.
pub(super) fn element(ctx: &mut Context, ty: Handle<Type>) -> Option<(Option<u32>, Handle<Type>)> {
    if let naga::TypeInner::BindingArray { base, size } = ctx.module.types[ty].inner {
        let bound = match size {
            naga::ArraySize::Constant(n) => Some(n.get()),
            _ => None,
        };
        return Some((bound, base));
    }
    if let Some((base, size)) = ctx.as_array(ty) {
        let bound = match size {
            naga::ArraySize::Constant(n) => Some(n.get()),
            _ => None,
        };
        return Some((bound, base));
    }
    match ctx.shape(ty) {
        Shape::Vector(size, scalar) => Some((Some(size as u32), ctx.intern_scalar(scalar))),
        Shape::Matrix(columns, rows, scalar) => {
            Some((Some(columns as u32), ctx.intern_vector(rows, scalar)))
        }
        _ => None,
    }
}

/// Read through a place, loading only the component asked for.
pub(super) fn load(
    function: &mut Function,
    body: &mut naga::Block,
    place: &Place,
) -> Result<Handle<Expression>, Error> {
    emit(
        function,
        body,
        Expression::Load {
            pointer: place.pointer,
        },
    )
}

/// Resolve an assignment target, turning "not a place" into the reason why.
pub(super) fn assign_place(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut naga::Block,
    expr: &Expr,
    env: &mut Env,
) -> Result<Place, Error> {
    if let Some(place) = lower_place(ctx, function, body, expr, env)? {
        if !place.writable {
            return Err(Error::AssignToReadonly(place.root));
        }
        return Ok(place);
    }
    // Name the specific problem where we can: assigning to an argument is a
    // different mistake from assigning to a call result.
    if let Some(name) = root_ident(expr) {
        if let Some(binding) = env.lookup(&name) {
            if matches!(binding.slot, Slot::Value(_)) {
                return Err(Error::AssignToArgument(name));
            }
        } else {
            return Err(Error::UnknownIdent(name));
        }
    }
    Err(Error::InvalidAssignTarget)
}

/// The identifier a place chain starts from, if it starts from one.
fn root_ident(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Paren(inner) => root_ident(&inner.expr),
        Expr::Group(inner) => root_ident(&inner.expr),
        Expr::Path(path) => path.path.get_ident().map(|i| i.to_string()),
        Expr::Field(field) => root_ident(&field.base),
        Expr::Index(index) => root_ident(&index.expr),
        _ => None,
    }
}

/// Shared by places and values: `base[index]`, with a literal index checked
/// against `bound` and a dynamic one required to be an integer.
pub(super) fn index_expr(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut naga::Block,
    index: &Expr,
    bound: Option<u32>,
    env: &mut Env,
) -> Result<IndexKind, Error> {
    if let Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(int),
        ..
    }) = strip(index)
    {
        let idx: u32 = int.base10_parse().map_err(Error::from)?;
        if matches!(bound, Some(bound) if idx >= bound) {
            return Err(Error::VecIndexRange);
        }
        return Ok(IndexKind::Constant(idx));
    }
    let (handle, ty) = lower_expr(ctx, function, body, index, env)?;
    match ctx.shape(ty) {
        Shape::Scalar(s) if s == Scalar::I32 || s == Scalar::U32 => {}
        _ => return Err(Error::TypeMismatch),
    }
    Ok(IndexKind::Dynamic(handle))
}

pub(super) enum IndexKind {
    Constant(u32),
    Dynamic(Handle<Expression>),
}

fn strip(expr: &Expr) -> &Expr {
    match expr {
        Expr::Paren(inner) => strip(&inner.expr),
        Expr::Group(inner) => strip(&inner.expr),
        other => other,
    }
}
