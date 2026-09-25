use naga::{
    BinaryOperator, Block, Expression, Function, Handle, Literal, Scalar, ScalarKind, Span,
    Statement, Type, UnaryOperator,
};
use syn::{BinOp, Expr};

use super::call::lower_call;
use super::emit::{emit, expr_kind};
use super::env::{Env, Slot};
use super::place::{self, lower_place};
use super::stmt::{lower_block, lower_if_expr};
use super::vector::{lower_field, lower_index, splat_mix, splat_shift};
use super::{Context, Shape, Typed};
use crate::Error;

pub(super) fn lower_expr(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    expr: &Expr,
    env: &mut Env,
) -> Result<Typed, Error> {
    lower_expr_hinted(ctx, function, body, expr, env, None)
}

/// Lower `expr`, letting untyped integer literals take the scalar type `hint`.
///
/// Rust infers `1` from its context (`x << 1`, `f(1)`, `let n: u32 = 1`); this
/// is how far that inference goes here. `hint` is only ever an integer scalar,
/// so a float context leaves `1` alone, exactly as Rust would.
pub(super) fn lower_expr_hinted(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    expr: &Expr,
    env: &mut Env,
    hint: Option<Scalar>,
) -> Result<Typed, Error> {
    match expr {
        Expr::Paren(inner) => lower_expr_hinted(ctx, function, body, &inner.expr, env, hint),
        Expr::Group(inner) => lower_expr_hinted(ctx, function, body, &inner.expr, env, hint),
        Expr::Path(path) => {
            // `vec4::ZERO` names a value on a type rather than a binding.
            if path.path.segments.len() == 2 {
                let ty = path.path.segments[0].ident.to_string();
                let item = path.path.segments[1].ident.to_string();
                return super::method::lower_qualified_const(ctx, function, &ty, &item);
            }
            let ident = path
                .path
                .get_ident()
                .ok_or_else(|| Error::UnsupportedExpr("path".into()))?;
            let name = ident.to_string();
            let Some(binding) = env.lookup(&name) else {
                return lower_const_ref(ctx, function, &name);
            };
            let ty = binding.ty;
            let expr = match binding.slot {
                Slot::Value(handle) => handle,
                Slot::Ptr(pointer) => emit(function, body, Expression::Load { pointer })?,
            };
            Ok((expr, ty))
        }
        Expr::Lit(lit) => lower_lit(ctx, function, lit, hint),
        Expr::Binary(bin) => {
            if let Some(op) = map_compound_op(&bin.op) {
                return lower_compound_assign(ctx, function, body, &bin.left, &bin.right, op, env);
            }
            let op = map_bin_op(&bin.op)?;
            lower_binary(ctx, function, body, op, &bin.left, &bin.right, env)
        }
        Expr::Unary(unary) => lower_unary(ctx, function, body, unary, env),
        Expr::Cast(cast) => lower_cast(ctx, function, body, cast, env),
        Expr::Assign(assign) => lower_assign(ctx, function, body, &assign.left, &assign.right, env),
        Expr::If(if_expr) => lower_if_expr(ctx, function, body, if_expr, env),
        Expr::Block(b) => {
            env.push_scope();
            let tail = lower_block(ctx, function, body, &b.block, env)?;
            env.pop_scope();
            tail.ok_or(Error::MissingBlockValue)
        }
        Expr::Call(call) => lower_call(ctx, function, body, call, env),
        Expr::Field(_) | Expr::Index(_) => {
            // A place loads just the component; anything else (a swizzle, a
            // field of a function argument) falls back to the value walk.
            if let Some(place) = lower_place(ctx, function, body, expr, env)? {
                return Ok((place::load(function, body, &place)?, place.ty));
            }
            match expr {
                Expr::Field(field) => lower_field(ctx, function, body, field, env),
                Expr::Index(index) => lower_index(ctx, function, body, index, env),
                _ => unreachable!("matched above"),
            }
        }
        Expr::Struct(lit) => super::structure::lower_struct_lit(ctx, function, body, lit, env),
        Expr::Array(array) => lower_array_lit(ctx, function, body, array, env),
        Expr::Reference(reference) => lower_reference(ctx, function, body, reference, env),
        Expr::MethodCall(call) => super::method::lower_method_call(ctx, function, body, call, env),
        _ => Err(Error::UnsupportedExpr(expr_kind(expr))),
    }
}

/// A module-level `const` referenced from a function body.
fn lower_const_ref(ctx: &mut Context, function: &mut Function, name: &str) -> Result<Typed, Error> {
    // WGSL predeclares the ray flags and intersection kinds as bare names.
    if let Some(value) = super::ray::predeclared_const(name) {
        let handle = function.expressions.append(
            Expression::Literal(naga::Literal::U32(value)),
            Span::UNDEFINED,
        );
        return Ok((handle, ctx.intern_scalar(Scalar::U32)));
    }
    let info = ctx
        .consts
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| Error::UnknownIdent(name.into()))?;
    // `Constant` is already a constant expression; emitting it would be wrong.
    let handle = function
        .expressions
        .append(Expression::Constant(info.handle), Span::UNDEFINED);
    Ok((handle, info.ty))
}

fn lower_unary(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    unary: &syn::ExprUnary,
    env: &mut Env,
) -> Result<Typed, Error> {
    // `*place` loads the place. Rust resource wrappers (`Workgroup<T>` and the
    // rest) need the star to see the inner value; in the shader the name is
    // already that value, so the star is the load.
    if matches!(unary.op, syn::UnOp::Deref(_)) {
        if let Some(place) = super::place::lower_place(ctx, function, body, &unary.expr, env)? {
            return Ok((super::place::load(function, body, &place)?, place.ty));
        }
        let (inner, ty) = lower_expr(ctx, function, body, &unary.expr, env)?;
        let Some(base) = ctx.pointee(ty) else {
            return Err(Error::UnsupportedExpr("deref".into()));
        };
        let handle = emit(function, body, Expression::Load { pointer: inner })?;
        return Ok((handle, base));
    }
    let (inner, ty) = lower_expr(ctx, function, body, &unary.expr, env)?;
    let op = match unary.op {
        // Naga has no negation for matrices or unsigned integers.
        syn::UnOp::Neg(_) => match ctx.shape(ty).elem_kind() {
            Some(ScalarKind::Float | ScalarKind::Sint) => UnaryOperator::Negate,
            _ => return Err(Error::BadOperandTypes("-".into())),
        },
        // `!` is logical on `bool` and bitwise on integers, as in Rust.
        syn::UnOp::Not(_) => match ctx.shape(ty).elem_kind() {
            Some(ScalarKind::Bool) => UnaryOperator::LogicalNot,
            Some(ScalarKind::Sint | ScalarKind::Uint) => UnaryOperator::BitwiseNot,
            _ => return Err(Error::BadOperandTypes("!".into())),
        },
        _ => return Err(Error::UnsupportedExpr("unary".into())),
    };
    let handle = emit(function, body, Expression::Unary { op, expr: inner })?;
    Ok((handle, ty))
}

fn lower_binary(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    op: BinaryOperator,
    left_expr: &Expr,
    right_expr: &Expr,
    env: &mut Env,
) -> Result<Typed, Error> {
    let shift = is_shift(op);
    // Lower the side that pins down the type first, so an untyped integer
    // literal on the other side can follow it. Literals have no side effects,
    // so swapping the order is not observable.
    let (mut left, mut left_ty, mut right, mut right_ty) =
        if !shift && is_untyped_int(left_expr) && !is_untyped_int(right_expr) {
            let (right, right_ty) = lower_expr(ctx, function, body, right_expr, env)?;
            let hint = ctx.shape(right_ty).int_hint();
            let (left, left_ty) = lower_expr_hinted(ctx, function, body, left_expr, env, hint)?;
            (left, left_ty, right, right_ty)
        } else {
            let (left, left_ty) = lower_expr(ctx, function, body, left_expr, env)?;
            // Shift amounts are always `u32`, whatever the left operand is.
            let hint = if shift {
                Some(Scalar::U32)
            } else {
                ctx.shape(left_ty).int_hint()
            };
            let (right, right_ty) = lower_expr_hinted(ctx, function, body, right_expr, env, hint)?;
            (left, left_ty, right, right_ty)
        };

    if shift {
        splat_shift(ctx, function, body, left_ty, &mut right, &mut right_ty)?;
    } else if op != BinaryOperator::Multiply {
        // `vec * scalar` is a single Naga multiply. Splatting the scalar first
        // forces a component-wise product, which the SPIR-V writer cannot emit
        // as OpVectorTimesScalar.
        splat_mix(
            ctx,
            function,
            body,
            &mut left,
            &mut left_ty,
            &mut right,
            &mut right_ty,
        )?;
    }
    let ty = bin_result_ty(ctx, op, left_ty, right_ty)?;
    let handle = emit(function, body, Expression::Binary { op, left, right })?;
    Ok((handle, ty))
}

fn lower_assign(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    left: &Expr,
    right: &Expr,
    env: &mut Env,
) -> Result<Typed, Error> {
    let place = place::assign_place(ctx, function, body, left, env)?;
    let (pointer, ty) = (place.pointer, place.ty);
    let hint = ctx.shape(ty).int_hint();
    let (value, value_ty) = lower_expr_hinted(ctx, function, body, right, env, hint)?;
    if value_ty != ty {
        return Err(Error::TypeMismatch);
    }
    body.push(Statement::Store { pointer, value }, Span::UNDEFINED);
    Ok((value, ty))
}

/// `x += e` and friends: same operand rules as the matching binary operator,
/// with the result required to fit back into `x`.
fn lower_compound_assign(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    left: &Expr,
    right: &Expr,
    op: BinaryOperator,
    env: &mut Env,
) -> Result<Typed, Error> {
    let place = place::assign_place(ctx, function, body, left, env)?;
    let (pointer, ty) = (place.pointer, place.ty);
    let shift = is_shift(op);
    let hint = if shift {
        Some(Scalar::U32)
    } else {
        ctx.shape(ty).int_hint()
    };
    // Rust evaluates the right operand first.
    let (mut rhs, mut rhs_ty) = lower_expr_hinted(ctx, function, body, right, env, hint)?;
    let mut lhs = emit(function, body, Expression::Load { pointer })?;
    let mut lhs_ty = ty;
    if shift {
        splat_shift(ctx, function, body, lhs_ty, &mut rhs, &mut rhs_ty)?;
    } else {
        splat_mix(
            ctx,
            function,
            body,
            &mut lhs,
            &mut lhs_ty,
            &mut rhs,
            &mut rhs_ty,
        )?;
    }
    if bin_result_ty(ctx, op, lhs_ty, rhs_ty)? != ty {
        return Err(Error::TypeMismatch);
    }
    let value = emit(
        function,
        body,
        Expression::Binary {
            op,
            left: lhs,
            right: rhs,
        },
    )?;
    body.push(Statement::Store { pointer, value }, Span::UNDEFINED);
    Ok((value, ty))
}

/// `&mut x` / `&x`: a pointer to storage, for an out-parameter.
///
/// Naga wants the pointer's address space to match where the storage lives, so
/// the place's own space comes along rather than being assumed.
fn lower_reference(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    reference: &syn::ExprReference,
    env: &mut Env,
) -> Result<Typed, Error> {
    // A texture or sampler is a handle, so `&tex` is just `tex`: Naga wants the
    // global itself, and there is no memory to take the address of.
    if let Some(place) = lower_place(ctx, function, body, &reference.expr, env)? {
        if !super::texture::is_handle(ctx, place.ty) {
            return finish_reference(ctx, reference, place);
        }
    }
    let (handle, ty) = lower_expr(ctx, function, body, &reference.expr, env)?;
    if super::texture::is_handle(ctx, ty) {
        return Ok((handle, ty));
    }
    let place = lower_place(ctx, function, body, &reference.expr, env)?
        .ok_or(Error::InvalidAssignTarget)?;
    finish_reference(ctx, reference, place)
}

fn finish_reference(
    ctx: &mut Context,
    reference: &syn::ExprReference,
    place: super::place::Place,
) -> Result<Typed, Error> {
    if reference.mutability.is_some() && !place.writable {
        return Err(Error::AssignToReadonly(place.root));
    }
    let ty = ctx.intern_handle_type(naga::TypeInner::Pointer {
        base: place.ty,
        space: place.space,
    });
    Ok((place.pointer, ty))
}

/// `[a, b, c]`: a fixed-size array, typed from its first element.
fn lower_array_lit(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    array: &syn::ExprArray,
    env: &mut Env,
) -> Result<Typed, Error> {
    let Some(len) = core::num::NonZeroU32::new(array.elems.len() as u32) else {
        return Err(Error::UnsupportedExpr("empty array literal".into()));
    };
    let mut hint = None;
    let mut components = Vec::new();
    let mut base = None;
    for elem in &array.elems {
        let (handle, ty) = lower_expr_hinted(ctx, function, body, elem, env, hint)?;
        match base {
            None => {
                hint = ctx.shape(ty).int_hint();
                base = Some(ty);
            }
            Some(base) if base != ty => return Err(Error::TypeMismatch),
            Some(_) => {}
        }
        components.push(handle);
    }
    let base = base.expect("non-empty");
    let ty = ctx.intern_array(base, naga::ArraySize::Constant(len))?;
    let handle = emit(function, body, Expression::Compose { ty, components })?;
    Ok((handle, ty))
}

fn lower_cast(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    cast: &syn::ExprCast,
    env: &mut Env,
) -> Result<Typed, Error> {
    let target = ctx.lower_type(&cast.ty)?;
    let (value, value_ty) = lower_expr(ctx, function, body, &cast.expr, env)?;
    if value_ty == target {
        return Ok((value, target));
    }
    // Component-wise conversion: scalar to scalar, or vector to same-size vector.
    let scalar = match (ctx.shape(value_ty), ctx.shape(target)) {
        (Shape::Scalar(_), Shape::Scalar(to)) => to,
        (Shape::Vector(from, _), Shape::Vector(to_size, to)) if from == to_size => to,
        _ => return Err(Error::UnsupportedCast(type_name(&cast.ty))),
    };
    let handle = emit(
        function,
        body,
        Expression::As {
            expr: value,
            kind: scalar.kind,
            convert: Some(scalar.width),
        },
    )?;
    Ok((handle, target))
}

fn type_name(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(path) => match path.path.segments.last() {
            Some(seg) => seg.ident.to_string(),
            None => "type".into(),
        },
        _ => "type".into(),
    }
}

fn lower_lit(
    ctx: &mut Context,
    function: &mut Function,
    lit: &syn::ExprLit,
    hint: Option<Scalar>,
) -> Result<Typed, Error> {
    let (literal, ty) = const_literal(ctx, lit, hint)?;
    let handle = function
        .expressions
        .append(Expression::Literal(literal), Span::UNDEFINED);
    Ok((handle, ty))
}

/// The Naga literal and type a Rust literal denotes, with `hint` standing in
/// for Rust's integer inference. Shared with constant lowering, which builds
/// its expressions in a different arena.
pub(super) fn const_literal(
    ctx: &mut Context,
    lit: &syn::ExprLit,
    hint: Option<Scalar>,
) -> Result<(Literal, Handle<Type>), Error> {
    Ok(match &lit.lit {
        syn::Lit::Float(f) => {
            if f.suffix() == "f64" {
                return Err(Error::UnsupportedType("f64".into()));
            }
            (
                Literal::F32(f.base10_parse().map_err(Error::from)?),
                ctx.intern_scalar(Scalar::F32),
            )
        }
        syn::Lit::Int(i) => {
            let scalar = match (i.suffix(), hint) {
                ("", Some(hint)) => hint,
                ("" | "i32", _) => Scalar::I32,
                ("u32", _) => Scalar::U32,
                (other, _) => return Err(Error::UnsupportedType(other.into())),
            };
            let literal = match scalar.kind {
                ScalarKind::Uint => Literal::U32(i.base10_parse().map_err(Error::from)?),
                _ => Literal::I32(i.base10_parse().map_err(Error::from)?),
            };
            (literal, ctx.intern_scalar(scalar))
        }
        syn::Lit::Bool(b) => (Literal::Bool(b.value()), ctx.intern_scalar(Scalar::BOOL)),
        _ => return Err(Error::UnsupportedExpr("literal".into())),
    })
}

/// Is this an integer literal with no suffix, and so open to a type hint?
pub(super) fn is_untyped_int(expr: &Expr) -> bool {
    match expr {
        Expr::Paren(inner) => is_untyped_int(&inner.expr),
        Expr::Group(inner) => is_untyped_int(&inner.expr),
        Expr::Lit(lit) => matches!(&lit.lit, syn::Lit::Int(i) if i.suffix().is_empty()),
        _ => false,
    }
}

pub(super) fn is_shift(op: BinaryOperator) -> bool {
    matches!(op, BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight)
}

/// Result type of `left op right`, rejecting everything Naga's validator would.
///
/// Mirrors `naga::valid::ExpressionError::InvalidBinaryOperandTypes`: catching
/// these here turns a module that fails validation into a clear frontend error.
pub(super) fn bin_result_ty(
    ctx: &mut Context,
    op: BinaryOperator,
    left: Handle<Type>,
    right: Handle<Type>,
) -> Result<Handle<Type>, Error> {
    use BinaryOperator as Bo;
    use ScalarKind as Sk;

    if op == Bo::Multiply {
        return super::matrix::multiply_result_ty(ctx, left, right);
    }
    if is_shift(op) {
        return shift_result_ty(ctx, op, left, right);
    }
    if left != right {
        return Err(Error::TypeMismatch);
    }

    let bad = || Error::BadOperandTypes(op_name(op).into());
    let shape = ctx.shape(left);
    match op {
        // Addition and subtraction are the only component-wise operators Naga
        // also defines on matrices.
        Bo::Add | Bo::Subtract => match (shape.elem_kind(), shape) {
            (Some(Sk::Uint | Sk::Sint | Sk::Float), _) | (None, Shape::Matrix(..)) => Ok(left),
            _ => Err(bad()),
        },
        Bo::Divide | Bo::Modulo => match shape.elem_kind() {
            Some(Sk::Uint | Sk::Sint | Sk::Float) => Ok(left),
            _ => Err(bad()),
        },
        Bo::Equal | Bo::NotEqual => match shape.elem_kind() {
            Some(_) => Ok(ctx.bool_like(left)),
            None => Err(bad()),
        },
        Bo::Less | Bo::LessEqual | Bo::Greater | Bo::GreaterEqual => match shape.elem_kind() {
            Some(Sk::Uint | Sk::Sint | Sk::Float) => Ok(ctx.bool_like(left)),
            _ => Err(bad()),
        },
        Bo::LogicalAnd | Bo::LogicalOr => match shape.elem_kind() {
            Some(Sk::Bool) => Ok(left),
            _ => Err(bad()),
        },
        Bo::And | Bo::InclusiveOr => match shape.elem_kind() {
            Some(Sk::Bool | Sk::Sint | Sk::Uint) => Ok(left),
            _ => Err(bad()),
        },
        Bo::ExclusiveOr => match shape.elem_kind() {
            Some(Sk::Sint | Sk::Uint) => Ok(left),
            _ => Err(bad()),
        },
        Bo::Multiply | Bo::ShiftLeft | Bo::ShiftRight => unreachable!("handled above"),
    }
}

/// Naga wants the shift amount to be `u32`, with the same vector size as the
/// value being shifted.
fn shift_result_ty(
    ctx: &mut Context,
    op: BinaryOperator,
    left: Handle<Type>,
    right: Handle<Type>,
) -> Result<Handle<Type>, Error> {
    match ctx.shape(left).elem_kind() {
        Some(ScalarKind::Sint | ScalarKind::Uint) => {}
        _ => return Err(Error::BadOperandTypes(op_name(op).into())),
    }
    let sizes_match = match (ctx.shape(left), ctx.shape(right)) {
        (Shape::Scalar(_), Shape::Scalar(s)) => s.kind == ScalarKind::Uint,
        (Shape::Vector(a, _), Shape::Vector(b, s)) => a == b && s.kind == ScalarKind::Uint,
        _ => false,
    };
    if sizes_match {
        Ok(left)
    } else {
        Err(Error::BadShiftType)
    }
}

fn op_name(op: BinaryOperator) -> &'static str {
    use BinaryOperator as Bo;
    match op {
        Bo::Add => "+",
        Bo::Subtract => "-",
        Bo::Multiply => "*",
        Bo::Divide => "/",
        Bo::Modulo => "%",
        Bo::Equal => "==",
        Bo::NotEqual => "!=",
        Bo::Less => "<",
        Bo::LessEqual => "<=",
        Bo::Greater => ">",
        Bo::GreaterEqual => ">=",
        Bo::And => "&",
        Bo::ExclusiveOr => "^",
        Bo::InclusiveOr => "|",
        Bo::LogicalAnd => "&&",
        Bo::LogicalOr => "||",
        Bo::ShiftLeft => "<<",
        Bo::ShiftRight => ">>",
    }
}

fn map_bin_op(op: &BinOp) -> Result<BinaryOperator, Error> {
    Ok(match op {
        BinOp::Add(_) => BinaryOperator::Add,
        BinOp::Sub(_) => BinaryOperator::Subtract,
        BinOp::Mul(_) => BinaryOperator::Multiply,
        BinOp::Div(_) => BinaryOperator::Divide,
        BinOp::Rem(_) => BinaryOperator::Modulo,
        BinOp::Eq(_) => BinaryOperator::Equal,
        BinOp::Ne(_) => BinaryOperator::NotEqual,
        BinOp::Lt(_) => BinaryOperator::Less,
        BinOp::Le(_) => BinaryOperator::LessEqual,
        BinOp::Gt(_) => BinaryOperator::Greater,
        BinOp::Ge(_) => BinaryOperator::GreaterEqual,
        BinOp::And(_) => BinaryOperator::LogicalAnd,
        BinOp::Or(_) => BinaryOperator::LogicalOr,
        BinOp::BitAnd(_) => BinaryOperator::And,
        BinOp::BitOr(_) => BinaryOperator::InclusiveOr,
        BinOp::BitXor(_) => BinaryOperator::ExclusiveOr,
        BinOp::Shl(_) => BinaryOperator::ShiftLeft,
        BinOp::Shr(_) => BinaryOperator::ShiftRight,
        _ => return Err(Error::UnsupportedBinOp("compound assignment".into())),
    })
}

fn map_compound_op(op: &BinOp) -> Option<BinaryOperator> {
    Some(match op {
        BinOp::AddAssign(_) => BinaryOperator::Add,
        BinOp::SubAssign(_) => BinaryOperator::Subtract,
        BinOp::MulAssign(_) => BinaryOperator::Multiply,
        BinOp::DivAssign(_) => BinaryOperator::Divide,
        BinOp::RemAssign(_) => BinaryOperator::Modulo,
        BinOp::BitAndAssign(_) => BinaryOperator::And,
        BinOp::BitOrAssign(_) => BinaryOperator::InclusiveOr,
        BinOp::BitXorAssign(_) => BinaryOperator::ExclusiveOr,
        BinOp::ShlAssign(_) => BinaryOperator::ShiftLeft,
        BinOp::ShrAssign(_) => BinaryOperator::ShiftRight,
        _ => return None,
    })
}
