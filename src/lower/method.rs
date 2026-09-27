//! The spellings a shader needs when it is also being type-checked by Rust.
//!
//! Rust cannot give one piece of memory a hundred overlapping names, cannot
//! overload a function on arity, and cannot make `a < b` yield one `bool` per
//! lane. So a checkable shader writes `v.xyz()`, `vec3::splat(x)`,
//! `v.extend(w)` and `a.cmple(b)` where WGSL writes `v.xyz`, `vec3(x)`,
//! `vec3(v, w)` and `a <= b`.
//!
//! This module reads those, and produces exactly what the WGSL spellings do.

use naga::{BinaryOperator, Block, Expression, Function, Scalar, VectorSize};
use syn::Expr;

use super::emit::emit;
use super::env::Env;
use super::expr::{bin_result_ty, lower_expr, lower_expr_hinted};
use super::place::swizzle_components;
use super::{parse_vec_ident, Context, Shape, Typed};
use crate::Error;

/// A method call. What the receiver is decides what the method means: an
/// atomic's `load` is not a texture's.
///
/// Some receivers are storage rather than a value. An atomic changes in place
/// and a ray query advances, so for those the receiver is found as a place
/// first, and its type says which it is. Anything else is loaded, as it would
/// be in any other expression.
pub(super) fn lower_method_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprMethodCall,
    env: &mut Env,
) -> Result<Typed, Error> {
    lower_method_any(ctx, function, body, call, env)?
        .ok_or_else(|| Error::ValueFromStatement(call.method.to_string()))
}

/// A method call anywhere: what it produces, or `None` for one that only
/// acts, like a texture's `store`, which is fine in statement or tail
/// position.
pub(super) fn lower_method_any(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprMethodCall,
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    let name = call.method.to_string();
    let args: Vec<&Expr> = call.args.iter().collect();

    let receiver = match super::place::lower_place(ctx, function, body, &call.receiver, env)? {
        Some(place) => {
            match ctx.module.types[place.ty].inner {
                naga::TypeInner::Atomic(scalar) => {
                    return super::atomic::lower_atomic_method(
                        ctx, function, body, place, scalar, &name, &args, env,
                    );
                }
                naga::TypeInner::RayQuery { .. } => {
                    return super::ray::lower_ray_method(
                        ctx, function, body, place, &name, &args, env,
                    );
                }
                // A runtime-sized array has a length only where it lives.
                naga::TypeInner::Array {
                    size: naga::ArraySize::Dynamic,
                    ..
                } if name == "len" && args.is_empty() => {
                    let handle = emit(function, body, Expression::ArrayLength(place.pointer))?;
                    return Ok(Some((handle, ctx.intern_scalar(Scalar::U32))));
                }
                _ => {}
            }
            let ty = place.ty;
            (super::place::load(function, body, &place)?, ty)
        }
        None => lower_expr(ctx, function, body, &call.receiver, env)?,
    };
    if super::texture::is_image(ctx, receiver.1) {
        return super::texture::lower_texture_method(
            ctx, function, body, receiver, &name, &args, env,
        );
    }
    lower_value_method(ctx, function, body, receiver, &name, &args, env).map(Some)
}

/// `v.xyz()`, `v.extend(w)`, `a.cmple(b)` and the rest, on a value.
fn lower_value_method(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    (base, base_ty): Typed,
    name: &str,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Typed, Error> {
    let name = name.to_string();

    // A swizzle takes no arguments and is named only by its components.
    if args.is_empty() {
        if let Some(components) = swizzle_components(&name) {
            return super::vector::swizzle(ctx, function, body, base, base_ty, &components, &name);
        }
    }

    match (name.as_str(), args) {
        // A fixed array's length is part of its type; `usize` is `u32` here.
        ("len", []) if ctx.as_array(base_ty).is_some() => {
            let Some((_, naga::ArraySize::Constant(len))) = ctx.as_array(base_ty) else {
                return Err(Error::UnsupportedMethod(name));
            };
            let handle = function.expressions.append(
                Expression::Literal(naga::Literal::U32(len.get())),
                naga::Span::UNDEFINED,
            );
            Ok((handle, ctx.intern_scalar(Scalar::U32)))
        }
        ("extend", [value]) => {
            let Shape::Vector(size, scalar) = ctx.shape(base_ty) else {
                return Err(Error::UnsupportedMethod(name));
            };
            let wider = match size {
                VectorSize::Bi => VectorSize::Tri,
                VectorSize::Tri => VectorSize::Quad,
                VectorSize::Quad => return Err(Error::UnsupportedMethod(name)),
            };
            let hint = Shape::Scalar(scalar).int_hint();
            let (value, value_ty) = lower_expr_hinted(ctx, function, body, value, env, hint)?;
            if ctx.shape(value_ty) != Shape::Scalar(scalar) {
                return Err(Error::TypeMismatch);
            }
            let ty = ctx.intern_vector(wider, scalar);
            let handle = emit(
                function,
                body,
                Expression::Compose {
                    ty,
                    components: vec![base, value],
                },
            )?;
            Ok((handle, ty))
        }
        ("truncate", []) => {
            let Shape::Vector(size, _) = ctx.shape(base_ty) else {
                return Err(Error::UnsupportedMethod(name));
            };
            let keep = match size {
                VectorSize::Bi => return Err(Error::UnsupportedMethod(name)),
                VectorSize::Tri => vec![0, 1],
                VectorSize::Quad => vec![0, 1, 2],
            };
            super::vector::swizzle(ctx, function, body, base, base_ty, &keep, &name)
        }
        (cmp, [rhs]) if compare_op(cmp).is_some() => {
            let op = compare_op(cmp).expect("checked above");
            let (left, left_ty) = (base, base_ty);
            let hint = ctx.shape(left_ty).int_hint();
            let (right, right_ty) = lower_expr_hinted(ctx, function, body, rhs, env, hint)?;
            let ty = bin_result_ty(ctx, op, left_ty, right_ty)?;
            let handle = emit(function, body, Expression::Binary { op, left, right })?;
            Ok((handle, ty))
        }
        _ => Err(Error::UnsupportedMethod(name)),
    }
}

/// The lane-wise comparisons, which Rust's operators cannot express.
fn compare_op(name: &str) -> Option<BinaryOperator> {
    Some(match name {
        "cmpeq" => BinaryOperator::Equal,
        "cmpne" => BinaryOperator::NotEqual,
        "cmplt" => BinaryOperator::Less,
        "cmple" => BinaryOperator::LessEqual,
        "cmpgt" => BinaryOperator::Greater,
        "cmpge" => BinaryOperator::GreaterEqual,
        _ => return None,
    })
}

/// `vec3::splat(x)`, `vec4::from(v)`, `vec4::ZERO`: a call or a constant
/// qualified by the type it belongs to.
pub(super) fn lower_qualified_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    ty_path: &[String],
    method: &str,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Typed, Error> {
    let ty_name = &super::last(ty_path);
    // `T::default()` is how Rust spells a zero value, and WGSL's `T()` is the
    // same thing. Any type may have one, so this comes before the vector names.
    if method == "default" && args.is_empty() {
        if let Some(ty) = super::call::zero_value_type(ctx, ty_path)? {
            let handle = function
                .expressions
                .append(Expression::ZeroValue(ty), naga::Span::UNDEFINED);
            return Ok((handle, ty));
        }
    }

    let Some((size, shorthand)) = parse_vec_ident(ty_name) else {
        return Err(Error::UnsupportedMethod(format!("{ty_name}::{method}")));
    };

    match (method, args) {
        ("splat", [value]) => {
            let hint = shorthand.and_then(|s| Shape::Scalar(s).int_hint());
            let (value, value_ty) = lower_expr_hinted(ctx, function, body, value, env, hint)?;
            let Shape::Scalar(scalar) = ctx.shape(value_ty) else {
                return Err(Error::TypeMismatch);
            };
            if matches!(shorthand, Some(want) if want != scalar) {
                return Err(Error::TypeMismatch);
            }
            let ty = ctx.intern_vector(size, scalar);
            let handle = emit(function, body, Expression::Splat { size, value })?;
            Ok((handle, ty))
        }
        // `vec4::from(v)` converts a vector's components, which a shader spells
        // as a cast. Rust's `as` only works on primitives, so vectors take this.
        ("from", [value]) => {
            let (value, value_ty) = lower_expr(ctx, function, body, value, env)?;
            let Shape::Vector(from_size, _) = ctx.shape(value_ty) else {
                return Err(Error::TypeMismatch);
            };
            if from_size != size {
                return Err(Error::TypeMismatch);
            }
            let scalar = shorthand.unwrap_or(Scalar::F32);
            let ty = ctx.intern_vector(size, scalar);
            let handle = emit(
                function,
                body,
                Expression::As {
                    expr: value,
                    kind: scalar.kind,
                    convert: Some(scalar.width),
                },
            )?;
            Ok((handle, ty))
        }
        _ => Err(Error::UnsupportedMethod(format!("{ty_name}::{method}"))),
    }
}

/// `vec4::ZERO` and `vec4::ONE`, which name a value rather than call anything.
pub(super) fn lower_qualified_const(
    ctx: &mut Context,
    function: &mut Function,
    ty_name: &str,
    constant: &str,
) -> Result<Typed, Error> {
    let Some((size, shorthand)) = parse_vec_ident(ty_name) else {
        return Err(Error::UnknownIdent(format!("{ty_name}::{constant}")));
    };
    let scalar = shorthand.unwrap_or(Scalar::F32);
    let ty = ctx.intern_vector(size, scalar);
    match constant {
        "ZERO" => {
            let handle = function
                .expressions
                .append(Expression::ZeroValue(ty), naga::Span::UNDEFINED);
            Ok((handle, ty))
        }
        _ => Err(Error::UnknownIdent(format!("{ty_name}::{constant}"))),
    }
}
