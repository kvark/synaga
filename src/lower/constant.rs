//! Module-level `const` items.
//!
//! Naga keeps constant initializers in a separate arena from function bodies,
//! with its own rule: everything in it has to be evaluatable without running
//! the shader. So these are lowered by a small separate walk rather than by
//! `lower_expr`, which builds runtime expressions, and the arithmetic in them is
//! folded first, by [`super::fold`], into the literal `rustc` would get.

use naga::{Constant, Expression, Handle, Literal, Scalar, Type};
use syn::{Expr, ItemConst};

use super::fold::{self, Ty, Value};
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
    let want = ctx.shape(ty).scalar();
    let (init, init_ty) = lower_const_expr(ctx, &item.expr, want)?;
    if init_ty != ty {
        return Err(Error::TypeMismatch);
    }
    let handle = ctx.module.constants.append(
        Constant {
            name: Some(name.clone()),
            ty,
            init,
        },
        ctx.span,
    );
    ctx.consts.push(ConstInfo {
        handle,
        ty,
        init_expr: init,
    });
    Ok(ctx.consts.len() - 1)
}

/// Lower `expr` into `module.global_expressions`, with `want` the scalar its
/// context gives it, as the `const`'s declared type does.
///
/// A literal, a constant or a type's constant is kept as written, and a
/// vector or matrix constructor is composed from its arguments. Anything
/// with arithmetic in it is folded to the literal it comes to, so what Naga
/// gets is already evaluated, the way it wants a constant.
fn lower_const_expr(
    ctx: &mut Context,
    expr: &Expr,
    want: Option<Scalar>,
) -> Result<(Handle<Expression>, Handle<Type>), Error> {
    match expr {
        Expr::Paren(inner) => lower_const_expr(ctx, &inner.expr, want),
        Expr::Group(inner) => lower_const_expr(ctx, &inner.expr, want),
        Expr::Lit(lit) => {
            let int_hint = want.and_then(|s| Shape::Scalar(s).int_hint());
            let (literal, ty) = super::expr::const_literal(ctx, lit, int_hint)?;
            let handle = ctx
                .module
                .global_expressions
                .append(Expression::Literal(literal), ctx.span);
            Ok((handle, ty))
        }
        Expr::Path(path) => {
            let segments = super::path_segments(&path.path);
            if let [ty_name, item] = segments.as_slice() {
                if let Some((literal, ty)) = type_const(ctx, ty_name, item) {
                    let handle = ctx
                        .module
                        .global_expressions
                        .append(Expression::Literal(literal), ctx.span);
                    return Ok((handle, ty));
                }
            }
            let Some(index) = ctx.constant(&segments)? else {
                // `PI`, or `f32::INFINITY`, which folding says WGSL cannot
                // write.
                return fold_literal(ctx, expr, want);
            };
            let info = &ctx.consts[index];
            let (handle, ty) = (info.handle, info.ty);
            let expr = ctx
                .module
                .global_expressions
                .append(Expression::Constant(handle), ctx.span);
            Ok((expr, ty))
        }
        Expr::Call(call) if is_constructor(call) => lower_const_ctor(ctx, call, want),
        // `cfg!(debug_assertions)`, settled by what the build was told.
        Expr::Macro(mac) if mac.mac.path.is_ident("cfg") => {
            let value = super::expr::eval_cfg(ctx, &mac.mac)?;
            let handle = ctx
                .module
                .global_expressions
                .append(Expression::Literal(naga::Literal::Bool(value)), ctx.span);
            Ok((handle, ctx.intern_scalar(Scalar::BOOL)))
        }
        _ => fold_literal(ctx, expr, want),
    }
}

/// `expr` folded to the literal it comes to.
fn fold_literal(
    ctx: &mut Context,
    expr: &Expr,
    want: Option<Scalar>,
) -> Result<(Handle<Expression>, Handle<Type>), Error> {
    let want = want.and_then(Ty::of);
    let value = match fold::fold(ctx, expr, want)? {
        // A float nothing gave a type is an `f64` to `rustc`, which a shader
        // has not; its context here is a vector's, unread.
        Value::F64(v) if want.is_none() => Value::F32(v as f32),
        value => value,
    };
    let literal = value.literal().map_err(|err| err.at(super::pos(expr)))?;
    let ty = ctx.intern_scalar(literal.scalar());
    let handle = ctx
        .module
        .global_expressions
        .append(Expression::Literal(literal), ctx.span);
    Ok((handle, ty))
}

/// `Ty::ITEM` where that is a literal: a set's flag or an enum's variant,
/// which is its own type when that type is a `u32` on the GPU, or a
/// primitive's own constant, as `u32::MAX` is.
pub(super) fn type_const(
    ctx: &mut Context,
    ty_name: &str,
    item: &str,
) -> Option<(Literal, Handle<Type>)> {
    let nominal = match ctx.scope.flags.get(ty_name) {
        Some(info) => info.flag(item).map(|value| (value, true)),
        None => ctx.scope.enum_variant(ty_name, item).map(|value| {
            let repr_u32 = ctx.scope.enums.get(ty_name).is_some_and(|e| e.repr_u32);
            (value, repr_u32)
        }),
    };
    match nominal {
        Some((value, true)) => Some((Literal::U32(value), ctx.intern_named_u32(ty_name))),
        Some((value, false)) => Some((Literal::U32(value), ctx.intern_scalar(Scalar::U32))),
        None => scalar_const(ty_name, item)
            .map(|literal| (literal, ctx.intern_scalar(literal.scalar()))),
    }
}

/// `vec3(..)` or `mat2(..)`: a call a constant composes rather than folds.
fn is_constructor(call: &syn::ExprCall) -> bool {
    match strip_parens(&call.func) {
        Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            let name = path.path.segments[0].ident.to_string();
            parse_vec_ident(&name).is_some() || parse_mat_ident(&name).is_some()
        }
        _ => false,
    }
}

/// `u32::MAX`, `f32::EPSILON`: a primitive's own constants, as literals.
/// There are none for an infinity or a NaN, which WGSL cannot write.
pub(super) fn scalar_const(ty: &str, name: &str) -> Option<naga::Literal> {
    use naga::Literal as L;
    Some(match (ty, name) {
        ("u32", "MAX") => L::U32(u32::MAX),
        ("u32", "MIN") => L::U32(u32::MIN),
        ("i32", "MAX") => L::I32(i32::MAX),
        ("i32", "MIN") => L::I32(i32::MIN),
        ("u32" | "i32", "BITS") => L::U32(32),
        ("f32", "MAX") => L::F32(f32::MAX),
        ("f32", "MIN") => L::F32(f32::MIN),
        ("f32", "MIN_POSITIVE") => L::F32(f32::MIN_POSITIVE),
        ("f32", "EPSILON") => L::F32(f32::EPSILON),
        ("f16", _) => L::F16(half_const(name)?),
        _ => return None,
    })
}

/// `f16::ONE`, `f16::PI`: the constants `half` gives an `f16`, but for its
/// infinities and NaN.
fn half_const(name: &str) -> Option<half::f16> {
    macro_rules! named {
        ($($name:ident)*) => {
            match name {
                $(stringify!($name) => Some(half::f16::$name),)*
                _ => None,
            }
        };
    }
    named!(
        ZERO NEG_ZERO ONE NEG_ONE MAX MIN MIN_POSITIVE EPSILON MIN_POSITIVE_SUBNORMAL MAX_SUBNORMAL
        E PI FRAC_1_PI FRAC_1_SQRT_2 FRAC_2_PI FRAC_2_SQRT_PI FRAC_PI_2 FRAC_PI_3 FRAC_PI_4
        FRAC_PI_6 FRAC_PI_8 LN_10 LN_2 LOG10_E LOG10_2 LOG2_E LOG2_10 SQRT_2
    )
}

/// `core::f32::consts::PI`, `std::f32::consts::TAU`: the constants `rustc`
/// has for `f32`, which the shader takes as literals of the same value.
pub(super) fn std_float(path: &[String]) -> Option<f32> {
    use core::f32::consts as c;
    let [.., ty, consts, name] = path else {
        return None;
    };
    if ty != "f32" || consts != "consts" {
        return None;
    }
    Some(match name.as_str() {
        "PI" => c::PI,
        "TAU" => c::TAU,
        "E" => c::E,
        "FRAC_PI_2" => c::FRAC_PI_2,
        "FRAC_PI_3" => c::FRAC_PI_3,
        "FRAC_PI_4" => c::FRAC_PI_4,
        "FRAC_PI_6" => c::FRAC_PI_6,
        "FRAC_PI_8" => c::FRAC_PI_8,
        "FRAC_1_PI" => c::FRAC_1_PI,
        "FRAC_2_PI" => c::FRAC_2_PI,
        "FRAC_2_SQRT_PI" => c::FRAC_2_SQRT_PI,
        "SQRT_2" => c::SQRT_2,
        "FRAC_1_SQRT_2" => c::FRAC_1_SQRT_2,
        "LN_2" => c::LN_2,
        "LN_10" => c::LN_10,
        "LOG2_E" => c::LOG2_E,
        "LOG2_10" => c::LOG2_10,
        "LOG10_E" => c::LOG10_E,
        "LOG10_2" => c::LOG10_2,
        _ => return None,
    })
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
    want: Option<Scalar>,
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

    let mut want = shorthand.or(want);
    let mut components = Vec::new();
    let mut component_tys = Vec::new();
    for arg in &call.args {
        let (handle, ty) = lower_const_expr(ctx, arg, want)?;
        want = want.or_else(|| ctx.shape(ty).scalar());
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
                ctx.span,
            );
            return Ok((handle, ty));
        }
        if components.len() != size as usize {
            return Err(Error::VecCtorArgs);
        }
        let handle = ctx
            .module
            .global_expressions
            .append(Expression::Compose { ty, components }, ctx.span);
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
        .append(Expression::Compose { ty, components }, ctx.span);
    Ok((handle, ty))
}
