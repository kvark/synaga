//! Atomics, spelled as `synaga_shader::AtomicU32` spells them: the standard
//! methods, without an `Ordering`, since WGSL's atomics are relaxed and
//! nothing else.
//!
//! An atomic is storage the operation changes, so the receiver is always a
//! place: a field of a buffer, an element of one, or workgroup memory.

use naga::{AtomicFunction, Block, Expression, Function, Scalar, Span, Statement};
use syn::Expr;

use super::emit::emit;
use super::env::Env;
use super::expr::lower_expr_hinted;
use super::place::Place;
use super::{Context, Typed};
use crate::Error;

/// `counter.fetch_add(1)` and the rest, on an atomic holding a `scalar`.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_atomic_method(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    place: Place,
    scalar: Scalar,
    name: &str,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Typed, Error> {
    let ty = ctx.intern_scalar(scalar);

    let fun = match (name, args) {
        ("load", []) => {
            let handle = emit(
                function,
                body,
                Expression::Load {
                    pointer: place.pointer,
                },
            )?;
            return Ok((handle, ty));
        }
        // Naga has no atomic store of its own: a store through a pointer to
        // an atomic is one.
        ("store", [value]) => {
            let value = operand(ctx, function, body, value, env, scalar)?;
            body.push(
                Statement::Store {
                    pointer: place.pointer,
                    value,
                },
                Span::UNDEFINED,
            );
            // Nothing to hand back; `()` in Rust, and never used as a value.
            return Ok((value, ty));
        }
        ("compare_exchange_weak", [current, new]) => {
            let compare = operand(ctx, function, body, current, env, scalar)?;
            let value = operand(ctx, function, body, new, env, scalar)?;
            let result_ty = ctx.module.generate_predeclared_type(
                naga::PredeclaredType::AtomicCompareExchangeWeakResult(scalar),
            );
            let result = function.expressions.append(
                Expression::AtomicResult {
                    ty: result_ty,
                    comparison: true,
                },
                Span::UNDEFINED,
            );
            body.push(
                Statement::Atomic {
                    pointer: place.pointer,
                    fun: AtomicFunction::Exchange {
                        compare: Some(compare),
                    },
                    value,
                    result: Some(result),
                },
                Span::UNDEFINED,
            );
            return Ok((result, result_ty));
        }
        ("swap", [_]) => AtomicFunction::Exchange { compare: None },
        ("fetch_add", [_]) => AtomicFunction::Add,
        ("fetch_sub", [_]) => AtomicFunction::Subtract,
        ("fetch_and", [_]) => AtomicFunction::And,
        ("fetch_or", [_]) => AtomicFunction::InclusiveOr,
        ("fetch_xor", [_]) => AtomicFunction::ExclusiveOr,
        ("fetch_min", [_]) => AtomicFunction::Min,
        ("fetch_max", [_]) => AtomicFunction::Max,
        (
            "load"
            | "store"
            | "swap"
            | "fetch_add"
            | "fetch_sub"
            | "fetch_and"
            | "fetch_or"
            | "fetch_xor"
            | "fetch_min"
            | "fetch_max"
            | "compare_exchange_weak",
            _,
        ) => return Err(Error::WrongArgCount(name.into())),
        _ => return Err(Error::UnsupportedMethod(format!("{name} on an atomic"))),
    };

    let value = operand(ctx, function, body, args[0], env, scalar)?;
    // The old value is produced whether or not anybody wants it.
    let result = function.expressions.append(
        Expression::AtomicResult {
            ty,
            comparison: false,
        },
        Span::UNDEFINED,
    );
    body.push(
        Statement::Atomic {
            pointer: place.pointer,
            fun,
            value,
            result: Some(result),
        },
        Span::UNDEFINED,
    );
    Ok((result, ty))
}

/// A value for the atomic to combine with: the atomic's own scalar, which an
/// untyped literal takes on.
fn operand(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    arg: &Expr,
    env: &mut Env,
    scalar: Scalar,
) -> Result<naga::Handle<Expression>, Error> {
    let (value, value_ty) = lower_expr_hinted(ctx, function, body, arg, env, Some(scalar))?;
    if ctx.as_scalar(value_ty) != Some(scalar) {
        return Err(Error::TypeMismatch);
    }
    Ok(value)
}
