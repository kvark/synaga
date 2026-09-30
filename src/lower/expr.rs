use naga::{
    BinaryOperator, Block, Expression, Function, Handle, Literal, Scalar, ScalarKind, Span,
    Statement, Type, UnaryOperator,
};
use syn::{BinOp, Expr};

use super::call::lower_call;
use super::constant;
use super::emit::{emit, expr_kind};
use super::env::{Env, Slot};
use super::place::{self, lower_place};
use super::stmt::{lower_block_hinted, lower_if_expr};
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
            let segments = super::path_segments(&path.path);
            let module_path = ctx.is_module_path(&segments);
            // `core::f32::consts::PI` is a literal to the shader.
            if let Some(value) = constant::std_float(&segments) {
                return Ok(float_literal(ctx, function, value));
            }
            // `Vec4::ZERO` names a value on a type rather than a binding, and
            // `Vec4::<u32>::ZERO` says the scalar as well.
            if let [ty @ .., item] = &segments[..] {
                if !ty.is_empty() && !module_path {
                    let ty_segment = &path.path.segments[path.path.segments.len() - 2];
                    let on = super::method::OnType {
                        path: ty,
                        turbofish: super::turbofish_scalar(ty_segment)?,
                        hint,
                    };
                    return super::method::lower_qualified_const(ctx, function, body, on, item);
                }
            }
            // A local shadows a module item. A path through a module cannot
            // name a local, so it goes straight to the items.
            let name = super::last(&segments);
            let binding = match module_path {
                false => env.lookup(&name),
                true => None,
            };
            let Some(binding) = binding else {
                return lower_item_ref(ctx, function, body, &segments, env);
            };
            let ty = binding.ty;
            let expr = match binding.slot {
                Slot::Value(handle) => handle,
                Slot::Ptr(pointer) => emit(function, body, Expression::Load { pointer })?,
            };
            Ok((expr, ty))
        }
        // `cfg!(debug_assertions)`, settled by what the build was told.
        Expr::Macro(mac) if mac.mac.path.is_ident("cfg") => {
            let value = eval_cfg(ctx, &mac.mac)?;
            let handle = function
                .expressions
                .append(Expression::Literal(Literal::Bool(value)), Span::UNDEFINED);
            Ok((handle, ctx.intern_scalar(Scalar::BOOL)))
        }
        Expr::Lit(lit) => lower_lit(ctx, function, lit, hint),
        Expr::Binary(bin) => {
            if let Some(op) = map_compound_op(&bin.op) {
                return lower_compound_assign(ctx, function, body, &bin.left, &bin.right, op, env);
            }
            let op = map_bin_op(&bin.op)?;
            lower_binary(ctx, function, body, op, (&bin.left, &bin.right), env, hint)
        }
        Expr::Unary(unary) => lower_unary(ctx, function, body, unary, env, hint),
        Expr::Cast(cast) => lower_cast(ctx, function, body, cast, env),
        Expr::Assign(assign) => lower_assign(ctx, function, body, &assign.left, &assign.right, env),
        Expr::If(if_expr) => lower_if_expr(ctx, function, body, if_expr, env, hint),
        Expr::Match(matched) => {
            let position = super::switch::Position::Value(hint);
            super::switch::lower_match(ctx, function, body, matched, env, position)?
                .ok_or(Error::MissingBlockValue)
        }
        Expr::Block(syn::ExprBlock { block, .. }) | Expr::Unsafe(syn::ExprUnsafe { block, .. }) => {
            env.push_scope();
            let tail = lower_block_hinted(ctx, function, body, block, env, hint)?;
            env.pop_scope();
            tail.ok_or(Error::MissingBlockValue)
        }
        Expr::Call(call) => lower_call(ctx, function, body, call, env, hint),
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
        Expr::MethodCall(call) => {
            super::method::lower_method_call(ctx, function, body, call, env, hint)
        }
        _ => Err(Error::UnsupportedExpr(expr_kind(expr))),
    }
}

fn float_literal(ctx: &mut Context, function: &mut Function, value: f32) -> Typed {
    let handle = function
        .expressions
        .append(Expression::Literal(Literal::F32(value)), Span::UNDEFINED);
    (handle, ctx.intern_scalar(Scalar::F32))
}

/// A module-level `const` or `static` referenced from a function body.
fn lower_item_ref(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    path: &[String],
    env: &mut Env,
) -> Result<Typed, Error> {
    let name = super::last(path);
    if let Some(index) = ctx.constant(path)? {
        let info = &ctx.consts[index];
        // `Constant` is already a constant expression; emitting it would be wrong.
        let handle = function
            .expressions
            .append(Expression::Constant(info.handle), Span::UNDEFINED);
        return Ok((handle, info.ty));
    }
    // `PI`, after `use core::f32::consts::PI`.
    if let Some(value) = ctx.float_const(path) {
        return Ok(float_literal(ctx, function, value));
    }
    // WGSL predeclares the ray flags and intersection kinds as bare names.
    if let Some(value) = super::ray::predeclared_const(&name) {
        let handle = function.expressions.append(
            Expression::Literal(naga::Literal::U32(value)),
            Span::UNDEFINED,
        );
        return Ok((handle, ctx.intern_scalar(Scalar::U32)));
    }
    // A global reached through its module, `lighting::sun`, is bound by name
    // like any other.
    if path.len() > 1 {
        if let Some(binding) = env.lookup(&name) {
            let ty = binding.ty;
            let expr = match binding.slot {
                Slot::Value(handle) => handle,
                Slot::Ptr(pointer) => emit(function, body, Expression::Load { pointer })?,
            };
            return Ok((expr, ty));
        }
    }
    Err(Error::UnknownIdent(name))
}

/// Does the predicate in `cfg!(...)` hold for this build?
pub(super) fn eval_cfg(ctx: &Context, mac: &syn::Macro) -> Result<bool, Error> {
    let meta: syn::Meta = mac
        .parse_body()
        .map_err(|_| Error::UnsupportedCfg(mac.tokens.to_string()))?;
    ctx.cfg.eval(&meta)
}

fn lower_unary(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    unary: &syn::ExprUnary,
    env: &mut Env,
    hint: Option<Scalar>,
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
    // `-` and `!` keep their operand's type, so where the result goes is
    // where the operand goes.
    let (inner, ty) = lower_expr_hinted(ctx, function, body, &unary.expr, env, hint)?;
    // A set's `!` is its complement, which stays within the declared flags.
    if matches!(unary.op, syn::UnOp::Not(_)) {
        if let Some(all) = ctx.flags_of(ty).map(|info| info.all) {
            return Ok((super::nominal::complement(function, body, inner, all)?, ty));
        }
    }
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
    operands: (&Expr, &Expr),
    env: &mut Env,
    outer: Option<Scalar>,
) -> Result<Typed, Error> {
    lower_binary_as(
        ctx,
        function,
        body,
        op,
        operands,
        env,
        outer,
        Comparison::Rust,
    )
}

/// What comparing two vectors produces.
#[derive(Clone, Copy, PartialEq)]
enum Comparison {
    /// One `bool`, as Rust's operators give.
    Rust,
    /// One per lane, as WGSL's operators give.
    Lanes,
}

/// `all(a < b)` and `any(a != b)`: WGSL's spelling of `a.cmplt(b).all()` and
/// `a.cmpne(b).any()`, whose comparison is lane by lane. Rust cannot write
/// it, since `all` takes lanes and Rust's `a < b` is one `bool`, so a source
/// that says it is one `rustc` never saw, and it means what it means in WGSL.
pub(super) fn lower_lanewise(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    expr: &Expr,
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    let Expr::Binary(bin) = super::constant::strip_parens(expr) else {
        return Ok(None);
    };
    let op = match map_compound_op(&bin.op) {
        Some(_) => return Ok(None),
        None => map_bin_op(&bin.op)?,
    };
    if whole_vector_comparison(op).is_none() {
        return Ok(None);
    }
    let operands = (&*bin.left, &*bin.right);
    lower_binary_as(
        ctx,
        function,
        body,
        op,
        operands,
        env,
        None,
        Comparison::Lanes,
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
fn lower_binary_as(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    op: BinaryOperator,
    (left_expr, right_expr): (&Expr, &Expr),
    env: &mut Env,
    outer: Option<Scalar>,
    comparison: Comparison,
) -> Result<Typed, Error> {
    if matches!(op, BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr) {
        return lower_lazy(ctx, function, body, op, (left_expr, right_expr), env);
    }
    let shift = is_shift(op);
    // Arithmetic, bitwise operators and shifts produce their operands' type,
    // so where the result goes says what the operands are too. A comparison's
    // `bool` says nothing about them.
    let outer = match op {
        BinaryOperator::Add
        | BinaryOperator::Subtract
        | BinaryOperator::Multiply
        | BinaryOperator::Divide
        | BinaryOperator::Modulo
        | BinaryOperator::And
        | BinaryOperator::ExclusiveOr
        | BinaryOperator::InclusiveOr
        | BinaryOperator::ShiftLeft
        | BinaryOperator::ShiftRight => outer,
        _ => None,
    };
    // Lower the side that pins down the type first, so an untyped integer
    // literal on the other side can follow it. Literals have no side effects,
    // so swapping the order is not observable.
    let (mut left, mut left_ty, mut right, mut right_ty) =
        if !shift && is_untyped_int(left_expr) && !is_untyped_int(right_expr) {
            let (right, right_ty) = lower_expr_hinted(ctx, function, body, right_expr, env, outer)?;
            let hint = ctx.shape(right_ty).int_hint();
            let (left, left_ty) = lower_expr_hinted(ctx, function, body, left_expr, env, hint)?;
            (left, left_ty, right, right_ty)
        } else {
            let (left, left_ty) = lower_expr_hinted(ctx, function, body, left_expr, env, outer)?;
            // Shift amounts are always `u32`, whatever the left operand is.
            let hint = if shift {
                Some(Scalar::U32)
            } else {
                ctx.shape(left_ty).int_hint()
            };
            let (right, right_ty) = lower_expr_hinted(ctx, function, body, right_expr, env, hint)?;
            (left, left_ty, right, right_ty)
        };

    // A set's `-` is the flags of the left that the right does not have.
    if op == BinaryOperator::Subtract && ctx.flags_of(left_ty).is_some() {
        if right_ty != left_ty {
            return Err(Error::TypeMismatch);
        }
        let handle = super::nominal::difference(function, body, left, right)?;
        return Ok((handle, left_ty));
    }
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
    // Rust compares two vectors to one `bool`: `a == b` when every lane is
    // equal, `a < b` when every lane is less. The lane-wise forms are
    // `a.cmpeq(b)` and the rest.
    if let Some(fold) = whole_vector_comparison(op) {
        if comparison == Comparison::Rust && matches!(ctx.shape(ty), Shape::Vector(..)) {
            let handle = emit(
                function,
                body,
                Expression::Relational {
                    fun: fold,
                    argument: handle,
                },
            )?;
            return Ok((handle, ctx.intern_scalar(Scalar::BOOL)));
        }
    }
    Ok((handle, ty))
}

/// How a comparison of two vectors folds its lanes into Rust's one `bool`:
/// `!=` holds if any lane differs, every other comparison if every lane holds.
fn whole_vector_comparison(op: BinaryOperator) -> Option<naga::RelationalFunction> {
    use BinaryOperator as Bo;
    match op {
        Bo::NotEqual => Some(naga::RelationalFunction::Any),
        Bo::Equal | Bo::Less | Bo::LessEqual | Bo::Greater | Bo::GreaterEqual => {
            Some(naga::RelationalFunction::All)
        }
        _ => None,
    }
}

/// `a && b` and `a || b`, which evaluate `b` only when `a` does not already
/// decide, as Rust's lazy boolean operators do.
///
/// A right side that only computes is evaluated either way, as a plain Naga
/// `&&`: nothing can tell, and the guarded form costs a local. Anything that
/// acts, like a call or an atomic, is a statement, and anything that indexes
/// may be out of bounds; either goes under an `if` on the left side.
fn lower_lazy(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    op: BinaryOperator,
    (left_expr, right_expr): (&Expr, &Expr),
    env: &mut Env,
) -> Result<Typed, Error> {
    let (left, left_ty) = lower_expr(ctx, function, body, left_expr, env)?;
    let mut right_block = Block::new();
    let (right, right_ty) = lower_expr(ctx, function, &mut right_block, right_expr, env)?;
    let ty = bin_result_ty(ctx, op, left_ty, right_ty)?;
    let pure = right_block
        .iter()
        .all(|statement| matches!(statement, Statement::Emit(_)));
    if pure && !indexes(right_expr) {
        body.append(&mut right_block);
        let handle = emit(function, body, Expression::Binary { op, left, right })?;
        return Ok((handle, ty));
    }
    let local = function.local_variables.append(
        naga::LocalVariable {
            name: None,
            ty,
            init: None,
        },
        Span::UNDEFINED,
    );
    let pointer = function
        .expressions
        .append(Expression::LocalVariable(local), Span::UNDEFINED);
    body.push(
        Statement::Store {
            pointer,
            value: left,
        },
        Span::UNDEFINED,
    );
    right_block.push(
        Statement::Store {
            pointer,
            value: right,
        },
        Span::UNDEFINED,
    );
    let (accept, reject) = match op {
        BinaryOperator::LogicalAnd => (right_block, Block::new()),
        _ => (Block::new(), right_block),
    };
    body.push(
        Statement::If {
            condition: left,
            accept,
            reject,
        },
        Span::UNDEFINED,
    );
    let value = emit(function, body, Expression::Load { pointer })?;
    Ok((value, ty))
}

/// Does `expr` index anything, which could be out of bounds?
fn indexes(expr: &Expr) -> bool {
    struct Finder(bool);
    impl<'ast> syn::visit::Visit<'ast> for Finder {
        fn visit_expr_index(&mut self, _: &'ast syn::ExprIndex) {
            self.0 = true;
        }
    }
    let mut finder = Finder(false);
    syn::visit::Visit::visit_expr(&mut finder, expr);
    finder.0
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
    if op == BinaryOperator::Subtract && ctx.flags_of(ty).is_some() {
        if rhs_ty != ty {
            return Err(Error::TypeMismatch);
        }
        let value = super::nominal::difference(function, body, lhs, rhs)?;
        body.push(Statement::Store { pointer, value }, Span::UNDEFINED);
        return Ok((value, ty));
    }
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
    // `Mode::Depth as u32` changes only which Rust type the `u32` is.
    if ctx.module.types[value_ty].inner == ctx.module.types[target].inner {
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
