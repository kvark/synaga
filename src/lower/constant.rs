//! Module-level `const` items.
//!
//! Naga keeps constant initializers in a separate arena from function bodies,
//! with its own rule: everything in it has to be evaluatable without running
//! the shader. So these are lowered by a small separate walk rather than by
//! `lower_expr`, which builds runtime expressions.

use naga::{Constant, Expression, Handle, Scalar, Span, Type};
use syn::{Expr, ItemConst};

use super::{parse_mat_ident, parse_vec_ident, Context, Shape};
use crate::Error;

pub(crate) struct ConstInfo {
    pub handle: Handle<Constant>,
    pub ty: Handle<Type>,
    /// The initializer, so a `const` can serve as an array length.
    pub init_expr: Handle<Expression>,
}

pub(super) fn lower_const_item(ctx: &mut Context, item: ItemConst) -> Result<usize, Error> {
    if !item.generics.params.is_empty() {
        return Err(Error::UnsupportedItem("generic const".into()));
    }
    let name = item.ident.to_string();
    let ty = ctx.lower_type(&item.ty)?;
    let hint = ctx.shape(ty).int_hint();
    let (init, init_ty) = lower_const_expr(ctx, &item.expr, hint)?;
    if init_ty != ty {
        return Err(Error::TypeMismatch);
    }
    let handle = ctx.module.constants.append(
        Constant {
            name: Some(name.clone()),
            ty,
            init,
        },
        Span::UNDEFINED,
    );
    ctx.consts.push(ConstInfo {
        handle,
        ty,
        init_expr: init,
    });
    Ok(ctx.consts.len() - 1)
}

/// Lower `expr` into `module.global_expressions`.
///
/// Deliberately narrow: literals, negation, and vector/matrix constructors are
/// what constants are actually written with, and everything else gets a clear
/// refusal instead of a module Naga rejects later.
fn lower_const_expr(
    ctx: &mut Context,
    expr: &Expr,
    hint: Option<Scalar>,
) -> Result<(Handle<Expression>, Handle<Type>), Error> {
    match expr {
        Expr::Paren(inner) => lower_const_expr(ctx, &inner.expr, hint),
        Expr::Group(inner) => lower_const_expr(ctx, &inner.expr, hint),
        Expr::Lit(lit) => {
            let (literal, ty) = super::expr::const_literal(ctx, lit, hint)?;
            let handle = ctx
                .module
                .global_expressions
                .append(Expression::Literal(literal), Span::UNDEFINED);
            Ok((handle, ty))
        }
        // Naga wants constants already folded, so `-1.0` negates the literal
        // rather than becoming a `Unary` expression the validator would reject.
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            let syn::Expr::Lit(lit) = strip_parens(&unary.expr) else {
                return Err(Error::UnsupportedConstExpr("negation".into()));
            };
            let (literal, ty) = super::expr::const_literal(ctx, lit, hint)?;
            let negated = match literal {
                naga::Literal::F32(v) => naga::Literal::F32(-v),
                naga::Literal::I32(v) => naga::Literal::I32(-v),
                _ => return Err(Error::BadOperandTypes("-".into())),
            };
            let handle = ctx
                .module
                .global_expressions
                .append(Expression::Literal(negated), Span::UNDEFINED);
            Ok((handle, ty))
        }
        Expr::Path(path) => {
            let segments = super::path_segments(&path.path);
            if let [enumeration, variant] = segments.as_slice() {
                if let Some(value) = ctx.scope.enum_variant(enumeration, variant) {
                    let ty = ctx.intern_scalar(Scalar::U32);
                    let handle = ctx.module.global_expressions.append(
                        Expression::Literal(naga::Literal::U32(value)),
                        Span::UNDEFINED,
                    );
                    return Ok((handle, ty));
                }
            }
            let index = ctx
                .constant(&segments)?
                .ok_or_else(|| Error::UnknownIdent(super::last(&segments)))?;
            let info = &ctx.consts[index];
            let (handle, ty) = (info.handle, info.ty);
            let expr = ctx
                .module
                .global_expressions
                .append(Expression::Constant(handle), Span::UNDEFINED);
            Ok((expr, ty))
        }
        Expr::Call(call) => lower_const_ctor(ctx, call, hint),
        // `cfg!(debug_assertions)`, settled by what the build was told.
        Expr::Macro(mac) if mac.mac.path.is_ident("cfg") => {
            let value = super::expr::eval_cfg(ctx, &mac.mac)?;
            let handle = ctx.module.global_expressions.append(
                Expression::Literal(naga::Literal::Bool(value)),
                Span::UNDEFINED,
            );
            Ok((handle, ctx.intern_scalar(Scalar::BOOL)))
        }
        other => Err(Error::UnsupportedConstExpr(super::emit::expr_kind(other))),
    }
}

pub(super) fn strip_parens(expr: &Expr) -> &Expr {
    match expr {
        Expr::Paren(inner) => strip_parens(&inner.expr),
        Expr::Group(inner) => strip_parens(&inner.expr),
        other => other,
    }
}

/// `vec3(1.0, 2.0, 3.0)` / `vec3(0.5)` / `mat2(..)` in constant position.
fn lower_const_ctor(
    ctx: &mut Context,
    call: &syn::ExprCall,
    hint: Option<Scalar>,
) -> Result<(Handle<Expression>, Handle<Type>), Error> {
    let name = match call.func.as_ref() {
        Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            path.path.segments[0].ident.to_string()
        }
        _ => return Err(Error::UnsupportedConstExpr("call".into())),
    };
    let vec = parse_vec_ident(&name);
    let mat = parse_mat_ident(&name);
    let shorthand = match (vec, mat) {
        (Some((_, s)), _) | (_, Some((_, _, s))) => s,
        _ => return Err(Error::UnsupportedConstExpr(name)),
    };
    if call.args.is_empty() {
        return Err(Error::VecCtorArgs);
    }

    let mut hint = shorthand.and_then(|s| Shape::Scalar(s).int_hint()).or(hint);
    let mut components = Vec::new();
    let mut component_tys = Vec::new();
    for arg in &call.args {
        let (handle, ty) = lower_const_expr(ctx, arg, hint)?;
        hint = hint.or_else(|| ctx.shape(ty).int_hint());
        components.push(handle);
        component_tys.push(ty);
    }
    let scalar = match (shorthand, ctx.shape(component_tys[0])) {
        (Some(s), _) => s,
        (None, Shape::Scalar(s) | Shape::Vector(_, s) | Shape::Matrix(_, _, s)) => s,
        (None, Shape::Other) => return Err(Error::TypeMismatch),
    };

    if let Some((size, _)) = vec {
        let ty = ctx.intern_vector(size, scalar);
        // A single scalar splats, as it does at runtime.
        if components.len() == 1 && matches!(ctx.shape(component_tys[0]), Shape::Scalar(_)) {
            let handle = ctx.module.global_expressions.append(
                Expression::Splat {
                    size,
                    value: components[0],
                },
                Span::UNDEFINED,
            );
            return Ok((handle, ty));
        }
        if components.len() != size as usize {
            return Err(Error::VecCtorArgs);
        }
        let handle = ctx
            .module
            .global_expressions
            .append(Expression::Compose { ty, components }, Span::UNDEFINED);
        return Ok((handle, ty));
    }

    let (columns, rows, _) = mat.expect("vector or matrix");
    let ty = ctx.intern_matrix(columns, rows, scalar);
    if components.len() != columns as usize
        || !component_tys
            .iter()
            .all(|&t| ctx.shape(t) == Shape::Vector(rows, scalar))
    {
        return Err(Error::MatCtorArgs);
    }
    let handle = ctx
        .module
        .global_expressions
        .append(Expression::Compose { ty, components }, Span::UNDEFINED);
    Ok((handle, ty))
}
