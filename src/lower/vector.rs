use naga::{Block, Expression, Function, Handle, Scalar, SwizzleComponent, Type, VectorSize};
use syn::Expr;

use super::emit::emit;
use super::env::Env;
use super::expr::{lower_expr, lower_expr_hinted};
use super::parse_vec_ident;
use super::place::{element, index_expr, swizzle_components, IndexKind};
use super::{Context, Shape, Typed};
use crate::Error;

pub(super) fn splat_mix(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    left: &mut Handle<Expression>,
    left_ty: &mut Handle<Type>,
    right: &mut Handle<Expression>,
    right_ty: &mut Handle<Type>,
) -> Result<(), Error> {
    match (ctx.shape(*left_ty), ctx.shape(*right_ty)) {
        (Shape::Vector(size, scalar), Shape::Scalar(s)) if s == scalar => {
            *right = emit(
                ctx,
                function,
                body,
                Expression::Splat {
                    size,
                    value: *right,
                },
            )?;
            *right_ty = ctx.intern_vector(size, scalar);
        }
        (Shape::Scalar(s), Shape::Vector(size, scalar)) if s == scalar => {
            *left = emit(
                ctx,
                function,
                body,
                Expression::Splat { size, value: *left },
            )?;
            *left_ty = ctx.intern_vector(size, scalar);
        }
        _ => {}
    }
    Ok(())
}

/// Shifting a vector by a single `u32` splats the amount across the lanes, the
/// way WGSL's `vec << u32` would.
pub(super) fn splat_shift(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    left_ty: Handle<Type>,
    right: &mut Handle<Expression>,
    right_ty: &mut Handle<Type>,
) -> Result<(), Error> {
    if let (Shape::Vector(size, _), Shape::Scalar(scalar)) =
        (ctx.shape(left_ty), ctx.shape(*right_ty))
    {
        if scalar == Scalar::U32 {
            *right = emit(
                ctx,
                function,
                body,
                Expression::Splat {
                    size,
                    value: *right,
                },
            )?;
            *right_ty = ctx.intern_vector(size, scalar);
        }
    }
    Ok(())
}

pub(super) fn lower_vec_ctor(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
    outer: Option<Scalar>,
) -> Result<Typed, Error> {
    let (name, turbofish) = match call.func.as_ref() {
        Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            let segment = &path.path.segments[0];
            (segment.ident.to_string(), super::turbofish_scalar(segment)?)
        }
        _ => return Err(Error::UnsupportedExpr("call".into())),
    };
    let (size, shorthand) =
        parse_vec_ident(&name).ok_or_else(|| Error::BadVecCtor(name.clone()))?;
    let shorthand = super::vec_scalar(&name, shorthand, turbofish)?;

    if call.args.is_empty() {
        return Err(Error::VecCtorArgs);
    }
    // `vec3::<u32>(1, 2, 3)`, or WGSL's `vec3u(1, 2, 3)`, fixes the component
    // type up front; plain `vec3(..)` takes it from where the vector goes, or
    // from the first argument, and the rest follow.
    let mut hint = shorthand
        .and_then(|s| Shape::Scalar(s).int_hint())
        .or(outer);
    let mut components = Vec::new();
    let mut component_tys = Vec::new();
    for arg in &call.args {
        let (handle, ty) = lower_expr_hinted(ctx, function, body, arg, env, hint)?;
        hint = hint.or_else(|| ctx.shape(ty).int_hint());
        components.push(handle);
        component_tys.push(ty);
    }

    let scalar = match (shorthand, ctx.shape(component_tys[0])) {
        (Some(s), _) => s,
        (None, Shape::Scalar(s) | Shape::Vector(_, s)) => s,
        (None, _) => return Err(Error::TypeMismatch),
    };

    let mut width = 0u32;
    for &ty in &component_tys {
        if let Some(s) = ctx.as_scalar(ty) {
            if s != scalar {
                return Err(Error::TypeMismatch);
            }
            width += 1;
        } else if let Some((comp_size, s)) = ctx.as_vector(ty) {
            if s != scalar {
                return Err(Error::TypeMismatch);
            }
            width += comp_size as u32;
        } else {
            return Err(Error::TypeMismatch);
        }
    }

    let ty = ctx.intern_vector(size, scalar);
    if components.len() == 1 && ctx.as_scalar(component_tys[0]).is_some() {
        let handle = emit(
            ctx,
            function,
            body,
            Expression::Splat {
                size,
                value: components[0],
            },
        )?;
        return Ok((handle, ty));
    }
    if components.len() == 1 && ctx.as_vector(component_tys[0]) == Some((size, scalar)) {
        return Ok((components[0], ty));
    }
    if width != size as u32 {
        return Err(Error::VecCtorArgs);
    }
    let handle = emit(ctx, function, body, Expression::Compose { ty, components })?;
    Ok((handle, ty))
}

pub(super) fn lower_field(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    field: &syn::ExprField,
    env: &mut Env,
) -> Result<Typed, Error> {
    let member = match &field.member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(_) => return Err(Error::UnsupportedExpr("tuple field".into())),
    };
    let (base, base_ty) = lower_expr(ctx, function, body, &field.base, env)?;
    if ctx.as_struct(base_ty).is_some() {
        return super::structure::lower_struct_field(ctx, function, body, base, base_ty, &member);
    }
    let letters =
        swizzle_components(&member).ok_or_else(|| Error::UnsupportedSwizzle(member.clone()))?;
    swizzle(ctx, function, body, base, base_ty, &letters, &member)
}

/// Build a swizzle: one component is an access, more than one is a `Swizzle`.
///
/// Shared with the method spelling, since `v.xyz` and `v.xyz()` mean the same
/// thing and should lower the same way.
pub(super) fn swizzle(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    base: Handle<Expression>,
    base_ty: Handle<Type>,
    letters: &[u32],
    member: &str,
) -> Result<Typed, Error> {
    let (vec_size, scalar) = ctx
        .as_vector(base_ty)
        .ok_or_else(|| Error::UnsupportedExpr("swizzle of a non-vector".into()))?;
    if letters.is_empty() || letters.len() > 4 || letters.iter().any(|&i| i >= vec_size as u32) {
        return Err(Error::UnsupportedSwizzle(member.into()));
    }
    let mut pattern = [SwizzleComponent::X; 4];
    for (slot, &index) in pattern.iter_mut().zip(letters.iter()) {
        *slot = match index {
            0 => SwizzleComponent::X,
            1 => SwizzleComponent::Y,
            2 => SwizzleComponent::Z,
            _ => SwizzleComponent::W,
        };
    }
    if letters.len() == 1 {
        let handle = emit(
            ctx,
            function,
            body,
            Expression::AccessIndex {
                base,
                index: letters[0],
            },
        )?;
        return Ok((handle, ctx.intern_scalar(scalar)));
    }
    let out_size = match letters.len() {
        2 => VectorSize::Bi,
        3 => VectorSize::Tri,
        _ => VectorSize::Quad,
    };
    let handle = emit(
        ctx,
        function,
        body,
        Expression::Swizzle {
            size: out_size,
            vector: base,
            pattern,
        },
    )?;
    Ok((handle, ctx.intern_vector(out_size, scalar)))
}

pub(super) fn lower_index(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    index: &syn::ExprIndex,
    env: &mut Env,
) -> Result<Typed, Error> {
    let (base, base_ty) = lower_expr(ctx, function, body, &index.expr, env)?;
    let (bound, result_ty) =
        element(ctx, base_ty).ok_or_else(|| Error::UnsupportedExpr("index".into()))?;
    let handle = match index_expr(ctx, function, body, &index.index, bound, env)? {
        IndexKind::Constant(index) => {
            emit(ctx, function, body, Expression::AccessIndex { base, index })?
        }
        IndexKind::Dynamic(index) => emit(ctx, function, body, Expression::Access { base, index })?,
    };
    Ok((handle, result_ty))
}
