//! Constant arithmetic, folded the way `rustc` folds it.
//!
//! `rustc` evaluates a `const` before the program runs, and the GPU has to get
//! the value `rustc` got. Naga has an evaluator, but it folds by WGSL's rules,
//! which have no wrapping arithmetic and none of the `const fn`s a primitive has
//! in Rust: `u32::MAX.wrapping_add(1)` is 0 to `rustc` and an overflow to WGSL.
//! So the arithmetic happens here, with Rust's own operators on the values, and
//! the module gets the literal it comes to.
//!
//! What a `const` holds is small. An operator on a vector is a trait method,
//! which a `const` cannot call, so everything folded here is a scalar:
//! literals, other constants, the operators, `as`, `if`, and the `const fn`s of
//! the primitive types. An `f16` is `half`'s, whose arithmetic is trait
//! methods, so a constant one only comes from a conversion, a type's constant
//! or another constant, and goes only through `half`'s own `const fn`s.

use std::cmp::Ordering;

use naga::{Expression, Literal, Scalar};
use syn::{BinOp, Expr, UnOp};

use super::constant::{strip_parens, type_const};
use super::{last, path_segments, Context};
use crate::Error;

/// A type constant arithmetic happens in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ty {
    Bool,
    I32,
    U32,
    F32,
    F16,
    /// What `rustc` makes an unsuffixed float when nothing says otherwise, as
    /// in `(1.0 / 3.0) as f32`. A shader has no `f64`, so one only passes
    /// through.
    F64,
}

impl Ty {
    /// The type a constant of `scalar` folds in, if it is one this folds.
    pub(super) fn of(scalar: Scalar) -> Option<Ty> {
        Some(match scalar {
            Scalar::BOOL => Ty::Bool,
            Scalar::I32 => Ty::I32,
            Scalar::U32 => Ty::U32,
            Scalar::F32 => Ty::F32,
            Scalar::F16 => Ty::F16,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Ty::Bool => "bool",
            Ty::I32 => "i32",
            Ty::U32 => "u32",
            Ty::F32 => "f32",
            Ty::F16 => "f16",
            Ty::F64 => "f64",
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Ty::F32 | Ty::F16 | Ty::F64)
    }
}

/// A scalar, as `rustc` computes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Value {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(f32),
    F16(half::f16),
    F64(f64),
}

impl Value {
    fn ty(self) -> Ty {
        match self {
            Value::Bool(_) => Ty::Bool,
            Value::I32(_) => Ty::I32,
            Value::U32(_) => Ty::U32,
            Value::F32(_) => Ty::F32,
            Value::F16(_) => Ty::F16,
            Value::F64(_) => Ty::F64,
        }
    }

    /// The literal a module holds for this value.
    pub(super) fn literal(self) -> Result<Literal, Error> {
        Ok(match self {
            Value::Bool(v) => Literal::Bool(v),
            Value::I32(v) => Literal::I32(v),
            Value::U32(v) => Literal::U32(v),
            Value::F32(v) if v.is_finite() => Literal::F32(v),
            Value::F16(v) if v.is_finite() => Literal::F16(v),
            Value::F32(_) | Value::F16(_) => {
                return Err(Error::ConstArithmetic(
                    "comes to an infinity or a NaN, which WGSL cannot write".into(),
                ))
            }
            Value::F64(_) => return Err(Error::UnsupportedType("f64".into())),
        })
    }

    fn of_literal(literal: Literal) -> Result<Value, Error> {
        Ok(match literal {
            Literal::Bool(v) => Value::Bool(v),
            Literal::I32(v) => Value::I32(v),
            Literal::U32(v) => Value::U32(v),
            Literal::F32(v) => Value::F32(v),
            Literal::F16(v) => Value::F16(v),
            other => return Err(Error::UnsupportedType(format!("{:?}", other.scalar()))),
        })
    }
}

/// The value `expr` comes to, with `want` the type its context gives an
/// unsuffixed literal, as a `const`'s declared type does.
pub(super) fn fold(ctx: &mut Context, expr: &Expr, want: Option<Ty>) -> Result<Value, Error> {
    fold_inner(ctx, expr, want).map_err(|err| err.at(super::pos(expr)))
}

/// What `const_index` came to: its literal, through any chain of constants
/// defined as one another.
pub(super) fn const_value(ctx: &Context, const_index: usize) -> Result<Value, Error> {
    let mut init = ctx.consts[const_index].init_expr;
    loop {
        match ctx.module.global_expressions[init] {
            Expression::Literal(literal) => return Value::of_literal(literal),
            Expression::Constant(other) => init = ctx.module.constants[other].init,
            _ => {
                return Err(Error::UnsupportedConstExpr(
                    "a vector or matrix in arithmetic".into(),
                ))
            }
        }
    }
}

fn fold_inner(ctx: &mut Context, expr: &Expr, want: Option<Ty>) -> Result<Value, Error> {
    match expr {
        Expr::Paren(inner) => fold(ctx, &inner.expr, want),
        Expr::Group(inner) => fold(ctx, &inner.expr, want),
        Expr::Lit(lit) => literal(&lit.lit, want, false),
        Expr::Unary(unary) => match unary.op {
            // `-2147483648` is an `i32`, though `2147483648` is not.
            UnOp::Neg(_) => match strip_parens(&unary.expr) {
                Expr::Lit(lit) => literal(&lit.lit, want, true),
                inner => neg(fold(ctx, inner, want)?),
            },
            UnOp::Not(_) => not(fold(ctx, &unary.expr, want)?),
            _ => Err(Error::UnsupportedConstExpr("`*`".into())),
        },
        Expr::Binary(binary) => fold_binary(ctx, binary, want),
        // `as` gives its operand no type, so `1.5 as u32` is an `f64` cast,
        // as it is to `rustc`.
        Expr::Cast(cast) => {
            let to = cast_target(&cast.ty)?;
            let from = natural(ctx, &cast.expr)?;
            cast_to(fold(ctx, &cast.expr, from)?, to)
        }
        Expr::Path(path) => path_value(ctx, &path.path),
        Expr::MethodCall(call) => fold_method(ctx, call, want),
        Expr::Call(call) => fold_call(ctx, call),
        Expr::If(if_expr) => {
            let Value::Bool(condition) = fold(ctx, &if_expr.cond, Some(Ty::Bool))? else {
                return Err(Error::TypeMismatch);
            };
            // Only the branch taken is evaluated, as in `rustc`.
            if condition {
                block_value(ctx, &if_expr.then_branch, want)
            } else {
                match &if_expr.else_branch {
                    Some((_, otherwise)) => fold(ctx, otherwise, want),
                    None => Err(Error::IfExprMissingElse),
                }
            }
        }
        Expr::Block(block) => block_value(ctx, &block.block, want),
        Expr::Macro(mac) if mac.mac.path.is_ident("cfg") => {
            Ok(Value::Bool(super::expr::eval_cfg(ctx, &mac.mac)?))
        }
        other => Err(Error::UnsupportedConstExpr(super::emit::expr_kind(other))),
    }
}

/// The tail of a block that is only a tail: `{ 1 }`, as an `if` holds one.
fn block_value(ctx: &mut Context, block: &syn::Block, want: Option<Ty>) -> Result<Value, Error> {
    match block.stmts.as_slice() {
        [syn::Stmt::Expr(expr, None)] => fold(ctx, expr, want),
        _ => Err(Error::UnsupportedConstExpr(
            "a block with statements".into(),
        )),
    }
}

/// The type `expr` has whatever its context: what a suffix, a constant's type
/// or an `as` says. `None` for an unsuffixed literal, and for anything made of
/// only those, which take their type from around them.
fn natural(ctx: &mut Context, expr: &Expr) -> Result<Option<Ty>, Error> {
    Ok(match expr {
        Expr::Paren(inner) => natural(ctx, &inner.expr)?,
        Expr::Group(inner) => natural(ctx, &inner.expr)?,
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Bool(_) => Some(Ty::Bool),
            syn::Lit::Int(int) if !int.suffix().is_empty() => Some(suffix_ty(int.suffix())?),
            syn::Lit::Float(float) if !float.suffix().is_empty() => {
                Some(suffix_ty(float.suffix())?)
            }
            _ => None,
        },
        Expr::Unary(unary) => natural(ctx, &unary.expr)?,
        Expr::Cast(cast) => Some(cast_target(&cast.ty)?),
        Expr::Binary(binary) => match binary.op {
            BinOp::And(_)
            | BinOp::Or(_)
            | BinOp::Eq(_)
            | BinOp::Ne(_)
            | BinOp::Lt(_)
            | BinOp::Le(_)
            | BinOp::Gt(_)
            | BinOp::Ge(_) => Some(Ty::Bool),
            BinOp::Shl(_) | BinOp::Shr(_) => natural(ctx, &binary.left)?,
            _ => match natural(ctx, &binary.left)? {
                Some(ty) => Some(ty),
                None => natural(ctx, &binary.right)?,
            },
        },
        // A path names a value of its own type, which folding it finds.
        Expr::Path(path) => Some(path_value(ctx, &path.path)?.ty()),
        Expr::MethodCall(call) => match method_result(&call.method.to_string()) {
            Some(ty) => Some(ty),
            None => natural(ctx, &call.receiver)?,
        },
        Expr::Call(call) => match call_name(call).as_deref() {
            Some("f32::from_bits") => Some(Ty::F32),
            Some("f16::from_f32_const" | "f16::from_f64_const" | "f16::from_bits") => Some(Ty::F16),
            _ => None,
        },
        Expr::If(if_expr) => match if_expr.then_branch.stmts.as_slice() {
            [syn::Stmt::Expr(tail, None)] => natural(ctx, tail)?,
            _ => None,
        },
        Expr::Block(block) => match block.block.stmts.as_slice() {
            [syn::Stmt::Expr(tail, None)] => natural(ctx, tail)?,
            _ => None,
        },
        Expr::Macro(mac) if mac.mac.path.is_ident("cfg") => Some(Ty::Bool),
        _ => None,
    })
}

/// What `u32` or `1.5f32` says a literal is.
fn suffix_ty(suffix: &str) -> Result<Ty, Error> {
    Ok(match suffix {
        "u32" | "usize" => Ty::U32,
        "i32" | "isize" => Ty::I32,
        "f32" => Ty::F32,
        "f64" => Ty::F64,
        other => return Err(Error::UnsupportedType(other.into())),
    })
}

/// A literal, typed by its suffix, else by `want`, else as `rustc` defaults
/// one: `i32` for an integer, `f64` for a float. `negated` takes a leading `-`
/// in, as `rustc` does for `-2147483648`.
fn literal(lit: &syn::Lit, want: Option<Ty>, negated: bool) -> Result<Value, Error> {
    match lit {
        syn::Lit::Bool(b) if !negated => Ok(Value::Bool(b.value)),
        syn::Lit::Int(int) => {
            let ty = match (int.suffix(), want) {
                ("", Some(Ty::I32 | Ty::U32)) => want.expect("matched"),
                // An integer literal is never a float to `rustc`, which says
                // so before this ever sees it, and nothing is an `f16`.
                ("", Some(Ty::F32 | Ty::F16 | Ty::F64 | Ty::Bool)) => {
                    return Err(Error::TypeMismatch)
                }
                ("", None) => Ty::I32,
                (suffix, _) => suffix_ty(suffix)?,
            };
            let magnitude: u64 = int.base10_parse()?;
            let out_of_range = || {
                Error::ConstArithmetic(format!("has a literal out of range for `{}`", ty.name()))
            };
            let value = match ty {
                Ty::I32 => {
                    // `-2147483648` reaches one further than `2147483647`.
                    let limit = (1u64 << 31) - u64::from(!negated);
                    if magnitude > limit {
                        return Err(out_of_range());
                    }
                    let signed = magnitude as i64;
                    Value::I32((if negated { -signed } else { signed }) as i32)
                }
                Ty::U32 if negated => return Err(Error::BadOperandTypes("-".into())),
                Ty::U32 => Value::U32(u32::try_from(magnitude).map_err(|_| out_of_range())?),
                // `1f32` is an integer token with a float suffix.
                Ty::F32 => Value::F32(signed_float(magnitude as f32, negated)),
                Ty::F64 => Value::F64(signed_float(magnitude as f64, negated)),
                Ty::F16 | Ty::Bool => return Err(Error::TypeMismatch),
            };
            Ok(value)
        }
        syn::Lit::Float(float) => {
            let ty = match (float.suffix(), want) {
                ("", Some(Ty::F32)) => Ty::F32,
                ("", Some(Ty::I32 | Ty::U32 | Ty::F16 | Ty::Bool)) => {
                    return Err(Error::TypeMismatch)
                }
                ("", _) => Ty::F64,
                (suffix, _) => suffix_ty(suffix)?,
            };
            // Parsed as the type it is, rather than through an `f64`, so the
            // digits round once, as they do for `rustc`.
            Ok(match ty {
                Ty::F32 => Value::F32(signed_float(float.base10_parse::<f32>()?, negated)),
                _ => Value::F64(signed_float(float.base10_parse::<f64>()?, negated)),
            })
        }
        _ => Err(Error::UnsupportedConstExpr("literal".into())),
    }
}

fn signed_float<F: std::ops::Neg<Output = F>>(value: F, negated: bool) -> F {
    if negated {
        -value
    } else {
        value
    }
}

fn overflows(ty: Ty) -> Error {
    Error::ConstArithmetic(format!("overflows `{}`", ty.name()))
}

/// `half` gives an `f16` its operators by traits, whose methods a `const`
/// cannot call.
fn half_operator(op: &str) -> Error {
    Error::ConstArithmetic(format!(
        "uses `{op}` on an `f16`, which `half` implements by a trait a `const` cannot call"
    ))
}

fn neg(value: Value) -> Result<Value, Error> {
    Ok(match value {
        Value::I32(v) => Value::I32(v.checked_neg().ok_or_else(|| overflows(Ty::I32))?),
        Value::F32(v) => Value::F32(-v),
        Value::F64(v) => Value::F64(-v),
        Value::F16(_) => return Err(half_operator("-")),
        Value::U32(_) | Value::Bool(_) => return Err(Error::BadOperandTypes("-".into())),
    })
}

fn not(value: Value) -> Result<Value, Error> {
    Ok(match value {
        Value::Bool(v) => Value::Bool(!v),
        Value::I32(v) => Value::I32(!v),
        Value::U32(v) => Value::U32(!v),
        Value::F32(_) | Value::F16(_) | Value::F64(_) => {
            return Err(Error::BadOperandTypes("!".into()))
        }
    })
}

fn fold_binary(
    ctx: &mut Context,
    binary: &syn::ExprBinary,
    want: Option<Ty>,
) -> Result<Value, Error> {
    let (left, right) = (&*binary.left, &*binary.right);
    match binary.op {
        BinOp::And(_) | BinOp::Or(_) => {
            let and = matches!(binary.op, BinOp::And(_));
            let Value::Bool(first) = fold(ctx, left, Some(Ty::Bool))? else {
                return Err(Error::BadOperandTypes(op_name(&binary.op).into()));
            };
            // It short-circuits, so the right side may be what an `&&` guards.
            if first != and {
                return Ok(Value::Bool(first));
            }
            match fold(ctx, right, Some(Ty::Bool))? {
                Value::Bool(second) => Ok(Value::Bool(second)),
                _ => Err(Error::BadOperandTypes(op_name(&binary.op).into())),
            }
        }
        // The amount is any integer, whatever the value shifted is.
        BinOp::Shl(_) | BinOp::Shr(_) => {
            let value_ty = natural(ctx, left)?.or(want);
            let value = fold(ctx, left, value_ty)?;
            let amount_ty = natural(ctx, right)?;
            let amount = fold(ctx, right, amount_ty)?;
            shift(&binary.op, value, amount)
        }
        BinOp::Eq(_) | BinOp::Ne(_) | BinOp::Lt(_) | BinOp::Le(_) | BinOp::Gt(_) | BinOp::Ge(_) => {
            let ty = match natural(ctx, left)? {
                Some(ty) => Some(ty),
                None => natural(ctx, right)?,
            };
            let (a, b) = (fold(ctx, left, ty)?, fold(ctx, right, ty)?);
            compare(&binary.op, a, b)
        }
        BinOp::Add(_)
        | BinOp::Sub(_)
        | BinOp::Mul(_)
        | BinOp::Div(_)
        | BinOp::Rem(_)
        | BinOp::BitAnd(_)
        | BinOp::BitOr(_)
        | BinOp::BitXor(_) => {
            let ty = match natural(ctx, left)? {
                Some(ty) => Some(ty),
                None => natural(ctx, right)?,
            }
            .or(want);
            let (a, b) = (fold(ctx, left, ty)?, fold(ctx, right, ty)?);
            arithmetic(&binary.op, a, b)
        }
        _ => Err(Error::UnsupportedConstExpr("an assignment".into())),
    }
}

fn op_name(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add(_) => "+",
        BinOp::Sub(_) => "-",
        BinOp::Mul(_) => "*",
        BinOp::Div(_) => "/",
        BinOp::Rem(_) => "%",
        BinOp::And(_) => "&&",
        BinOp::Or(_) => "||",
        BinOp::BitXor(_) => "^",
        BinOp::BitAnd(_) => "&",
        BinOp::BitOr(_) => "|",
        BinOp::Shl(_) => "<<",
        BinOp::Shr(_) => ">>",
        BinOp::Eq(_) => "==",
        BinOp::Ne(_) => "!=",
        BinOp::Lt(_) => "<",
        BinOp::Le(_) => "<=",
        BinOp::Gt(_) => ">",
        BinOp::Ge(_) => ">=",
        _ => "operator",
    }
}

/// `a op b` on two integers of one type, refused where `rustc` refuses it.
macro_rules! int_arithmetic {
    ($op:expr, $a:expr, $b:expr, $ty:expr) => {{
        let (a, b) = ($a, $b);
        let result = match $op {
            BinOp::Div(_) | BinOp::Rem(_) if b == 0 => {
                return Err(Error::ConstArithmetic("divides by zero".into()))
            }
            BinOp::Add(_) => a.checked_add(b),
            BinOp::Sub(_) => a.checked_sub(b),
            BinOp::Mul(_) => a.checked_mul(b),
            BinOp::Div(_) => a.checked_div(b),
            BinOp::Rem(_) => a.checked_rem(b),
            BinOp::BitAnd(_) => Some(a & b),
            BinOp::BitOr(_) => Some(a | b),
            BinOp::BitXor(_) => Some(a ^ b),
            other => return Err(Error::BadOperandTypes(op_name(other).into())),
        };
        result.ok_or_else(|| overflows($ty))?
    }};
}

/// `a op b` on two floats of one type: IEEE arithmetic, which is what `rustc`
/// does too, `%` included.
macro_rules! float_arithmetic {
    ($op:expr, $a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        match $op {
            BinOp::Add(_) => a + b,
            BinOp::Sub(_) => a - b,
            BinOp::Mul(_) => a * b,
            BinOp::Div(_) => a / b,
            BinOp::Rem(_) => a % b,
            other => return Err(Error::BadOperandTypes(op_name(other).into())),
        }
    }};
}

fn arithmetic(op: &BinOp, a: Value, b: Value) -> Result<Value, Error> {
    Ok(match (a, b) {
        (Value::I32(a), Value::I32(b)) => Value::I32(int_arithmetic!(op, a, b, Ty::I32)),
        (Value::U32(a), Value::U32(b)) => Value::U32(int_arithmetic!(op, a, b, Ty::U32)),
        (Value::F32(a), Value::F32(b)) => Value::F32(float_arithmetic!(op, a, b)),
        (Value::F64(a), Value::F64(b)) => Value::F64(float_arithmetic!(op, a, b)),
        (Value::F16(_), Value::F16(_)) => return Err(half_operator(op_name(op))),
        (Value::Bool(a), Value::Bool(b)) => Value::Bool(match op {
            BinOp::BitAnd(_) => a & b,
            BinOp::BitOr(_) => a | b,
            BinOp::BitXor(_) => a ^ b,
            other => return Err(Error::BadOperandTypes(op_name(other).into())),
        }),
        _ => return Err(Error::TypeMismatch),
    })
}

fn shift(op: &BinOp, value: Value, amount: Value) -> Result<Value, Error> {
    let amount = match amount {
        Value::U32(n) => n,
        Value::I32(n) => u32::try_from(n)
            .map_err(|_| Error::ConstArithmetic("shifts by a negative amount".into()))?,
        _ => return Err(Error::BadShiftType),
    };
    let too_far = |ty: Ty| {
        Error::ConstArithmetic(format!(
            "shifts a `{}` by {amount}, its width or more",
            ty.name()
        ))
    };
    let left = matches!(op, BinOp::Shl(_));
    Ok(match value {
        Value::I32(v) => Value::I32(
            if left {
                v.checked_shl(amount)
            } else {
                v.checked_shr(amount)
            }
            .ok_or_else(|| too_far(Ty::I32))?,
        ),
        Value::U32(v) => Value::U32(
            if left {
                v.checked_shl(amount)
            } else {
                v.checked_shr(amount)
            }
            .ok_or_else(|| too_far(Ty::U32))?,
        ),
        _ => return Err(Error::BadOperandTypes(op_name(op).into())),
    })
}

fn compare(op: &BinOp, a: Value, b: Value) -> Result<Value, Error> {
    let ordering = match (a, b) {
        (Value::I32(a), Value::I32(b)) => a.partial_cmp(&b),
        (Value::U32(a), Value::U32(b)) => a.partial_cmp(&b),
        (Value::F32(a), Value::F32(b)) => a.partial_cmp(&b),
        (Value::F64(a), Value::F64(b)) => a.partial_cmp(&b),
        (Value::F16(_), Value::F16(_)) => return Err(half_operator(op_name(op))),
        (Value::Bool(a), Value::Bool(b)) => a.partial_cmp(&b),
        _ => return Err(Error::TypeMismatch),
    };
    // A NaN is unordered: unequal to everything, itself included.
    Ok(Value::Bool(match op {
        BinOp::Eq(_) => ordering == Some(Ordering::Equal),
        BinOp::Ne(_) => ordering != Some(Ordering::Equal),
        BinOp::Lt(_) => ordering == Some(Ordering::Less),
        BinOp::Le(_) => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
        BinOp::Gt(_) => ordering == Some(Ordering::Greater),
        _ => matches!(ordering, Some(Ordering::Greater | Ordering::Equal)),
    }))
}

/// The type `as` converts to.
fn cast_target(ty: &syn::Type) -> Result<Ty, Error> {
    let name = match ty {
        syn::Type::Path(path) if path.qself.is_none() => path
            .path
            .get_ident()
            .map(|ident| ident.to_string())
            .unwrap_or_default(),
        _ => String::new(),
    };
    match name.as_str() {
        "u32" | "usize" => Ok(Ty::U32),
        "i32" | "isize" => Ok(Ty::I32),
        "f32" => Ok(Ty::F32),
        "f64" => Ok(Ty::F64),
        _ => Err(Error::UnsupportedCast(name)),
    }
}

/// `value as to`, as Rust's `as` does it: an integer wraps into the other's
/// bits, and a float saturates into an integer, a NaN becoming 0.
fn cast_to(value: Value, to: Ty) -> Result<Value, Error> {
    macro_rules! to {
        ($v:expr) => {
            match to {
                Ty::I32 => Value::I32($v as i32),
                Ty::U32 => Value::U32($v as u32),
                Ty::F32 => Value::F32($v as f32),
                Ty::F64 => Value::F64($v as f64),
                Ty::F16 | Ty::Bool => return Err(Error::UnsupportedCast(to.name().into())),
            }
        };
    }
    Ok(match value {
        Value::I32(v) => to!(v),
        Value::U32(v) => to!(v),
        Value::F32(v) => to!(v),
        Value::F64(v) => to!(v),
        // `as` is for primitives, which `half`'s `f16` is not.
        Value::F16(_) => return Err(Error::UnsupportedCast(to.name().into())),
        // `true as f32` is no cast `rustc` takes.
        Value::Bool(_) if to.is_float() => return Err(Error::UnsupportedCast(to.name().into())),
        Value::Bool(v) => to!(u32::from(v)),
    })
}

/// What a path names in a constant: a type's own constant, as `u32::MAX` or
/// `Mode::Depth` is, another constant, or one of `core::f32::consts`.
fn path_value(ctx: &mut Context, path: &syn::Path) -> Result<Value, Error> {
    let segments = path_segments(path);
    if let [ty, item] = segments.as_slice() {
        if let Some((literal, _)) = type_const(ctx, ty, item) {
            return Value::of_literal(literal);
        }
        // No literal can be one of these, but arithmetic can pass through
        // one, as `f32::NAN != f32::NAN` does.
        match (ty.as_str(), item.as_str()) {
            ("f32", "NAN") => return Ok(Value::F32(f32::NAN)),
            ("f32", "INFINITY") => return Ok(Value::F32(f32::INFINITY)),
            ("f32", "NEG_INFINITY") => return Ok(Value::F32(f32::NEG_INFINITY)),
            ("f16", "NAN") => return Ok(Value::F16(half::f16::NAN)),
            ("f16", "INFINITY") => return Ok(Value::F16(half::f16::INFINITY)),
            ("f16", "NEG_INFINITY") => return Ok(Value::F16(half::f16::NEG_INFINITY)),
            _ => {}
        }
    }
    if let Some(index) = ctx.constant(&segments)? {
        return const_value(ctx, index);
    }
    match ctx.float_const(&segments) {
        Some(value) => Ok(Value::F32(value)),
        None => Err(Error::UnknownIdent(last(&segments))),
    }
}

/// The path a call names, as `f32::from_bits`.
fn call_name(call: &syn::ExprCall) -> Option<String> {
    match strip_parens(&call.func) {
        Expr::Path(path) if path.qself.is_none() => Some(path_segments(&path.path).join("::")),
        _ => None,
    }
}

/// The functions a constant calls: `f32::from_bits`, and `half`'s conversions
/// into an `f16`, which round to the nearest as `half` does.
fn fold_call(ctx: &mut Context, call: &syn::ExprCall) -> Result<Value, Error> {
    match (call_name(call).as_deref(), call.args.len()) {
        (Some("f32::from_bits"), 1) => match fold(ctx, &call.args[0], Some(Ty::U32))? {
            Value::U32(bits) => Ok(Value::F32(f32::from_bits(bits))),
            _ => Err(Error::TypeMismatch),
        },
        (Some("f16::from_f32_const"), 1) => match fold(ctx, &call.args[0], Some(Ty::F32))? {
            Value::F32(v) => Ok(Value::F16(half::f16::from_f32(v))),
            _ => Err(Error::TypeMismatch),
        },
        (Some("f16::from_f64_const"), 1) => match fold(ctx, &call.args[0], Some(Ty::F64))? {
            Value::F64(v) => Ok(Value::F16(half::f16::from_f64(v))),
            _ => Err(Error::TypeMismatch),
        },
        // The bits are a `u16`, which a shader has not, so they fold as a
        // `u32` that has to fit.
        (Some("f16::from_bits"), 1) => match fold(ctx, &call.args[0], Some(Ty::U32))? {
            Value::U32(bits) => match u16::try_from(bits) {
                Ok(bits) => Ok(Value::F16(half::f16::from_bits(bits))),
                Err(_) => Err(Error::ConstArithmetic(
                    "has a literal out of range for `u16`".into(),
                )),
            },
            _ => Err(Error::TypeMismatch),
        },
        (Some(name), _) => Err(Error::UnsupportedConstExpr(format!("{name}()"))),
        (None, _) => Err(Error::UnsupportedConstExpr("call".into())),
    }
}

/// The type a primitive's method returns when that is not its receiver's.
fn method_result(name: &str) -> Option<Ty> {
    match name {
        "count_ones" | "count_zeros" | "leading_zeros" | "trailing_zeros" | "ilog2"
        | "unsigned_abs" | "to_bits" => Some(Ty::U32),
        "to_f32_const" => Some(Ty::F32),
        "to_f64_const" => Some(Ty::F64),
        "is_power_of_two" | "is_positive" | "is_negative" | "is_nan" | "is_finite"
        | "is_infinite" | "is_sign_positive" | "is_sign_negative" => Some(Ty::Bool),
        _ => None,
    }
}

/// The argument count and argument type of the primitive methods a constant
/// can call: `None` for the receiver's own type.
fn method_args(name: &str) -> Option<(usize, Option<Ty>)> {
    Some(match name {
        "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "saturating_sub"
        | "saturating_mul" | "div_ceil" | "div_euclid" | "rem_euclid" | "min" | "max"
        | "copysign" => (1, None),
        "clamp" => (2, None),
        "wrapping_shl" | "wrapping_shr" | "rotate_left" | "rotate_right" | "pow"
        | "wrapping_pow" | "saturating_pow" => (1, Some(Ty::U32)),
        "wrapping_neg" | "count_ones" | "count_zeros" | "leading_zeros" | "trailing_zeros"
        | "reverse_bits" | "swap_bytes" | "abs" | "signum" | "unsigned_abs" | "ilog2"
        | "is_power_of_two" | "next_power_of_two" | "is_positive" | "is_negative" | "to_bits"
        | "recip" | "to_degrees" | "to_radians" | "is_nan" | "is_finite" | "is_infinite"
        | "is_sign_positive" | "is_sign_negative" | "to_f32_const" | "to_f64_const" => (0, None),
        _ => return None,
    })
}

/// An integer receiver's method, by name, computed by that very method.
macro_rules! int_method {
    ($prim:ident, $variant:ident, $v:expr, $name:expr, $args:expr, $unsupported:expr) => {{
        let v: $prim = $v;
        let same = |args: &[Value]| match args.first() {
            Some(Value::$variant(other)) => Ok(*other),
            _ => Err(Error::TypeMismatch),
        };
        let checked = |result: Option<$prim>| result.ok_or_else(|| overflows(Ty::$variant));
        Ok(match $name {
            "wrapping_add" => Value::$variant(v.wrapping_add(same($args)?)),
            "wrapping_sub" => Value::$variant(v.wrapping_sub(same($args)?)),
            "wrapping_mul" => Value::$variant(v.wrapping_mul(same($args)?)),
            "wrapping_neg" => Value::$variant(v.wrapping_neg()),
            "wrapping_shl" => Value::$variant(v.wrapping_shl(u32_arg($args)?)),
            "wrapping_shr" => Value::$variant(v.wrapping_shr(u32_arg($args)?)),
            "wrapping_pow" => Value::$variant(v.wrapping_pow(u32_arg($args)?)),
            "saturating_add" => Value::$variant(v.saturating_add(same($args)?)),
            "saturating_sub" => Value::$variant(v.saturating_sub(same($args)?)),
            "saturating_mul" => Value::$variant(v.saturating_mul(same($args)?)),
            "saturating_pow" => Value::$variant(v.saturating_pow(u32_arg($args)?)),
            "pow" => Value::$variant(checked(v.checked_pow(u32_arg($args)?))?),
            "rotate_left" => Value::$variant(v.rotate_left(u32_arg($args)?)),
            "rotate_right" => Value::$variant(v.rotate_right(u32_arg($args)?)),
            "div_euclid" | "rem_euclid" => {
                let other = same($args)?;
                if other == 0 {
                    return Err(Error::ConstArithmetic("divides by zero".into()));
                }
                Value::$variant(checked(match $name {
                    "div_euclid" => v.checked_div_euclid(other),
                    _ => v.checked_rem_euclid(other),
                })?)
            }
            "count_ones" => Value::U32(v.count_ones()),
            "count_zeros" => Value::U32(v.count_zeros()),
            "leading_zeros" => Value::U32(v.leading_zeros()),
            "trailing_zeros" => Value::U32(v.trailing_zeros()),
            "reverse_bits" => Value::$variant(v.reverse_bits()),
            "swap_bytes" => Value::$variant(v.swap_bytes()),
            "ilog2" => Value::U32(v.checked_ilog2().ok_or_else(|| {
                Error::ConstArithmetic("takes the logarithm of a number below 1".into())
            })?),
            _ => {
                return int_specific!(
                    $prim,
                    $variant,
                    v,
                    $name,
                    $args,
                    $unsupported,
                    same,
                    checked
                )
            }
        })
    }};
}

/// The methods only a signed or only an unsigned integer has.
macro_rules! int_specific {
    (i32, $variant:ident, $v:expr, $name:expr, $args:expr, $unsupported:expr, $same:expr, $checked:expr) => {{
        Ok(match $name {
            "abs" => Value::I32($checked($v.checked_abs())?),
            "signum" => Value::I32($v.signum()),
            "unsigned_abs" => Value::U32($v.unsigned_abs()),
            "is_positive" => Value::Bool($v.is_positive()),
            "is_negative" => Value::Bool($v.is_negative()),
            _ => return Err($unsupported()),
        })
    }};
    (u32, $variant:ident, $v:expr, $name:expr, $args:expr, $unsupported:expr, $same:expr, $checked:expr) => {{
        Ok(match $name {
            "div_ceil" => {
                let other = $same($args)?;
                if other == 0 {
                    return Err(Error::ConstArithmetic("divides by zero".into()));
                }
                Value::U32($v.div_ceil(other))
            }
            "is_power_of_two" => Value::Bool($v.is_power_of_two()),
            "next_power_of_two" => Value::U32($checked($v.checked_next_power_of_two())?),
            _ => return Err($unsupported()),
        })
    }};
}

/// A float receiver's method, by name, computed by that very method.
macro_rules! float_method {
    ($variant:ident, $v:expr, $name:expr, $args:expr, $unsupported:expr) => {{
        let v = $v;
        let arg = |index: usize| match $args.get(index) {
            Some(Value::$variant(other)) => Ok(*other),
            _ => Err(Error::TypeMismatch),
        };
        Ok(match $name {
            "abs" => Value::$variant(v.abs()),
            "signum" => Value::$variant(v.signum()),
            "copysign" => Value::$variant(v.copysign(arg(0)?)),
            "recip" => Value::$variant(v.recip()),
            "to_degrees" => Value::$variant(v.to_degrees()),
            "to_radians" => Value::$variant(v.to_radians()),
            "min" => Value::$variant(v.min(arg(0)?)),
            "max" => Value::$variant(v.max(arg(0)?)),
            "clamp" => {
                let (low, high) = (arg(0)?, arg(1)?);
                // `clamp` panics on these, which in a constant is an error.
                if !matches!(
                    low.partial_cmp(&high),
                    Some(Ordering::Less | Ordering::Equal)
                ) {
                    return Err(Error::ConstArithmetic(
                        "clamps between bounds that are out of order or NaN".into(),
                    ));
                }
                Value::$variant(v.clamp(low, high))
            }
            "is_nan" => Value::Bool(v.is_nan()),
            "is_finite" => Value::Bool(v.is_finite()),
            "is_infinite" => Value::Bool(v.is_infinite()),
            "is_sign_positive" => Value::Bool(v.is_sign_positive()),
            "is_sign_negative" => Value::Bool(v.is_sign_negative()),
            "to_bits" => float_bits!($variant, v, $unsupported),
            _ => return Err($unsupported()),
        })
    }};
}

/// `to_bits` is a `u32` for an `f32` and a `u64`, which a shader has not, for
/// an `f64`.
macro_rules! float_bits {
    (F32, $v:expr, $unsupported:expr) => {
        Value::U32($v.to_bits())
    };
    (F64, $v:expr, $unsupported:expr) => {
        return Err($unsupported())
    };
}

/// A primitive's `const fn` method: `u32::MAX.wrapping_add(1)`,
/// `2u32.pow(10)`, `1.5f32.to_bits()`.
fn fold_method(
    ctx: &mut Context,
    call: &syn::ExprMethodCall,
    want: Option<Ty>,
) -> Result<Value, Error> {
    let name = call.method.to_string();
    let Some((count, arg_ty)) = method_args(&name) else {
        return Err(Error::UnsupportedConstExpr(format!("{name}()")));
    };
    if call.args.len() != count {
        return Err(Error::WrongArgCount(name));
    }
    // `rustc` finds a method by its receiver's type, so a receiver has one of
    // its own, as `2u32` does; the context's is only a fallback.
    let receiver_ty = match natural(ctx, &call.receiver)? {
        Some(ty) => Some(ty),
        None if method_result(&name).is_none() => want,
        None => None,
    };
    let receiver = fold(ctx, &call.receiver, receiver_ty)?;
    let mut args = Vec::with_capacity(count);
    for arg in &call.args {
        args.push(fold(ctx, arg, Some(arg_ty.unwrap_or(receiver.ty())))?);
    }
    let unsupported = || Error::UnsupportedMethod(format!("{}::{name}", receiver.ty().name()));
    let (method, args) = (name.as_str(), &args[..]);
    match receiver {
        Value::I32(v) => int_method!(i32, I32, v, method, args, unsupported),
        Value::U32(v) => int_method!(u32, U32, v, method, args, unsupported),
        Value::F32(v) => float_method!(F32, v, method, args, unsupported),
        Value::F64(v) => float_method!(F64, v, method, args, unsupported),
        Value::F16(v) => half_method(v, method, args, unsupported),
        Value::Bool(_) => Err(unsupported()),
    }
}

/// The `const fn`s `half` gives an `f16`, computed by `half`. `to_bits` is one
/// too, but its `u16` is no type a shader has.
fn half_method(
    v: half::f16,
    name: &str,
    args: &[Value],
    unsupported: impl FnOnce() -> Error,
) -> Result<Value, Error> {
    Ok(match name {
        "to_f32_const" => Value::F32(v.to_f32()),
        "to_f64_const" => Value::F64(v.to_f64()),
        "signum" => Value::F16(v.signum()),
        "copysign" => match args.first() {
            Some(Value::F16(sign)) => Value::F16(v.copysign(*sign)),
            _ => return Err(Error::TypeMismatch),
        },
        "is_nan" => Value::Bool(v.is_nan()),
        "is_finite" => Value::Bool(v.is_finite()),
        "is_infinite" => Value::Bool(v.is_infinite()),
        "is_sign_positive" => Value::Bool(v.is_sign_positive()),
        "is_sign_negative" => Value::Bool(v.is_sign_negative()),
        _ => return Err(unsupported()),
    })
}

/// The `u32` argument a shift, rotation or power takes.
fn u32_arg(args: &[Value]) -> Result<u32, Error> {
    match args.first() {
        Some(Value::U32(n)) => Ok(*n),
        _ => Err(Error::TypeMismatch),
    }
}
