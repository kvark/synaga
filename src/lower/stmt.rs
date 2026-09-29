use std::collections::HashSet;

use naga::{Block, Expression, Function, Handle, LocalVariable, Span, Statement, Type};
use syn::visit::{self, Visit};
use syn::{BinOp, Block as SynBlock, Expr, Local, Pat, Stmt, Type as SynType};

use super::emit::emit;
use super::env::{Env, Slot};
use super::expr::{is_untyped_int, lower_expr, lower_expr_hinted};
use super::{Context, Typed};
use crate::Error;

/// Does this expression produce a value in tail position?
///
/// `if cond { return a; } else { return b; }` is a perfectly good function body
/// even though the `if` itself yields nothing, so the shape of the branches —
/// not just the keyword — decides whether to lower it as a value or a statement.
fn yields_value(expr: &Expr) -> bool {
    match expr {
        Expr::Break(_)
        | Expr::Continue(_)
        | Expr::ForLoop(_)
        | Expr::Loop(_)
        | Expr::Return(_)
        | Expr::While(_) => false,
        Expr::Paren(inner) => yields_value(&inner.expr),
        Expr::Group(inner) => yields_value(&inner.expr),
        Expr::Block(b) => block_yields_value(&b.block),
        Expr::Unsafe(u) => block_yields_value(&u.block),
        Expr::If(if_expr) => match &if_expr.else_branch {
            Some((_, else_expr)) => {
                block_yields_value(&if_expr.then_branch) && yields_value(else_expr)
            }
            None => false,
        },
        _ => true,
    }
}

fn block_yields_value(block: &SynBlock) -> bool {
    matches!(block.stmts.last(), Some(Stmt::Expr(expr, None)) if yields_value(expr))
}

pub(super) fn lower_block(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    block: &SynBlock,
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    lower_block_hinted(ctx, function, body, block, env, None)
}

/// [`lower_block`], for a block whose value goes somewhere that says its
/// type: `hint` reaches the tail, as Rust's inference would carry it there.
pub(super) fn lower_block_hinted(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    block: &SynBlock,
    env: &mut Env,
    hint: Option<naga::Scalar>,
) -> Result<Option<Typed>, Error> {
    lower_stmts(ctx, function, body, &block.stmts, env, hint)
}

/// A function's body. What it ends in is returned from where it is, so an
/// `if` in tail position returns from each branch, as a `return` in each
/// would, rather than through a local both branches store to: idiomatic Rust
/// leaves the `return`s out, and the module should not be the worse for it.
pub(super) fn lower_body(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    block: &SynBlock,
    env: &mut Env,
) -> Result<(), Error> {
    if function.result.is_none() {
        // `rustc` has checked that whatever the body ends in is `()`.
        let _ = lower_block(ctx, function, body, block, env)?;
        return Ok(());
    }
    lower_returning_block(ctx, function, body, block, env)?;
    if !always_jumps(body) {
        return Err(Error::MissingReturn(
            function.name.clone().unwrap_or_default(),
        ));
    }
    Ok(())
}

/// A block the function returns the value of.
fn lower_returning_block(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    block: &SynBlock,
    env: &mut Env,
) -> Result<(), Error> {
    // An `if` or a block is taken apart even when a branch ends in its own
    // `return`, as in `if c { return a; } else { b }`.
    let returns = |tail: &Expr| match tail {
        Expr::If(if_expr) => if_expr.else_branch.is_some(),
        Expr::Block(_) | Expr::Unsafe(_) => true,
        other => yields_value(other),
    };
    match block.stmts.split_last() {
        Some((Stmt::Expr(tail, None), rest)) if returns(tail) => {
            let _ = lower_stmts(ctx, function, body, rest, env, None)?;
            lower_returning_expr(ctx, function, body, tail, env)
        }
        // No value at the end: the block leaves by its own `return`s, which
        // the caller checks for.
        _ => {
            let hint = return_hint(ctx, function);
            let _ = lower_stmts(ctx, function, body, &block.stmts, env, hint)?;
            Ok(())
        }
    }
}

/// An expression in tail position, returned: each branch of an `if` returns
/// its own value.
fn lower_returning_expr(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    expr: &Expr,
    env: &mut Env,
) -> Result<(), Error> {
    match expr {
        Expr::Paren(inner) => lower_returning_expr(ctx, function, body, &inner.expr, env),
        Expr::Group(inner) => lower_returning_expr(ctx, function, body, &inner.expr, env),
        Expr::Block(syn::ExprBlock { block, .. }) | Expr::Unsafe(syn::ExprUnsafe { block, .. }) => {
            env.push_scope();
            lower_returning_block(ctx, function, body, block, env)?;
            env.pop_scope();
            Ok(())
        }
        Expr::If(syn::ExprIf {
            cond,
            then_branch,
            else_branch: Some((_, else_expr)),
            ..
        }) => {
            let (condition, _) = lower_expr(ctx, function, body, cond, env)?;
            let mut accept = Block::new();
            env.push_scope();
            lower_returning_block(ctx, function, &mut accept, then_branch, env)?;
            env.pop_scope();
            let mut reject = Block::new();
            env.push_scope();
            lower_returning_expr(ctx, function, &mut reject, else_expr, env)?;
            env.pop_scope();
            body.push(
                Statement::If {
                    condition,
                    accept,
                    reject,
                },
                Span::UNDEFINED,
            );
            Ok(())
        }
        other => {
            let hint = return_hint(ctx, function);
            if let Some((value, ty)) = lower_tail(ctx, function, body, other, env, hint)? {
                check_return(function, ty)?;
                body.push(Statement::Return { value: Some(value) }, Span::UNDEFINED);
            }
            Ok(())
        }
    }
}

/// Statements, in order, and the value of the last if it has one.
fn lower_stmts(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    stmts: &[Stmt],
    env: &mut Env,
    hint: Option<naga::Scalar>,
) -> Result<Option<Typed>, Error> {
    let mut tail = None;
    for (i, stmt) in stmts.iter().enumerate() {
        let last = i + 1 == stmts.len();
        match stmt {
            Stmt::Local(local) => {
                lower_local(ctx, function, body, local, env)?;
                tail = None;
            }
            Stmt::Expr(Expr::Return(ret), _) => {
                let value = match ret.expr.as_deref() {
                    Some(expr) => {
                        let hint = return_hint(ctx, function);
                        let (value, ty) = lower_expr_hinted(ctx, function, body, expr, env, hint)?;
                        check_return(function, ty)?;
                        Some(value)
                    }
                    None => None,
                };
                body.push(Statement::Return { value }, Span::UNDEFINED);
                tail = None;
            }
            Stmt::Expr(expr, semi) => {
                if last && semi.is_none() && yields_value(expr) {
                    tail = lower_tail(ctx, function, body, expr, env, hint)?;
                } else {
                    lower_stmt_expr(ctx, function, body, expr, env)?;
                    tail = None;
                }
            }
            Stmt::Item(_) => return Err(Error::UnsupportedStmt("item in block".into())),
            Stmt::Macro(_) => return Err(Error::UnsupportedStmt("macro".into())),
        }
    }
    Ok(tail)
}

/// Does control flow always leave `block` through a jump, rather than running
/// off the end?
///
/// A function with a result has to return on every path. Naga's validator does
/// not check this, so a body like `if c { return a; }` would otherwise reach a
/// backend as a shader that falls off the end.
pub(super) fn always_jumps(block: &Block) -> bool {
    match block.last() {
        Some(Statement::Return { .. } | Statement::Kill) => true,
        Some(Statement::If { accept, reject, .. }) => always_jumps(accept) && always_jumps(reject),
        // A loop nobody breaks out of never falls through.
        Some(Statement::Loop { body, break_if, .. }) => break_if.is_none() && !has_break(body),
        _ => false,
    }
}

/// Is there a `break` targeting *this* loop? Nested loops capture their own.
fn has_break(block: &Block) -> bool {
    block.iter().any(|stmt| match stmt {
        Statement::Break => true,
        Statement::If { accept, reject, .. } => has_break(accept) || has_break(reject),
        Statement::Block(inner) => has_break(inner),
        _ => false,
    })
}

/// The last expression of a block, which is its value if it has one.
///
/// Whether it has one is not always visible in the syntax: `f()` is `()` when
/// `f` returns nothing, and so is `t.store(c, v)`. So a call, and an `if` or a
/// block ending in one, may turn out to have no value, which is fine here.
pub(super) fn lower_tail(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    expr: &Expr,
    env: &mut Env,
    hint: Option<naga::Scalar>,
) -> Result<Option<Typed>, Error> {
    match expr {
        Expr::Call(call) => super::call::lower_call_any(ctx, function, body, call, env, hint),
        Expr::MethodCall(call) => {
            super::method::lower_method_any(ctx, function, body, call, env, hint)
        }
        Expr::If(if_expr) => lower_if_any(ctx, function, body, if_expr, env, hint),
        Expr::Block(syn::ExprBlock { block, .. }) | Expr::Unsafe(syn::ExprUnsafe { block, .. }) => {
            env.push_scope();
            let tail = lower_block_hinted(ctx, function, body, block, env, hint)?;
            env.pop_scope();
            Ok(tail)
        }
        Expr::Paren(inner) => lower_tail(ctx, function, body, &inner.expr, env, hint),
        Expr::Group(inner) => lower_tail(ctx, function, body, &inner.expr, env, hint),
        other => lower_expr_hinted(ctx, function, body, other, env, hint).map(Some),
    }
}

/// The scalar a function's result says, for the literals in what it returns:
/// `fn f() -> u32 { 1 }`.
pub(super) fn return_hint(ctx: &Context, function: &Function) -> Option<naga::Scalar> {
    let result = function.result.as_ref()?;
    ctx.shape(result.ty).int_hint()
}

/// A returned value has to be what the function says it returns. Naga checks
/// too, but names neither the function nor the problem.
pub(super) fn check_return(function: &Function, ty: Handle<Type>) -> Result<(), Error> {
    match &function.result {
        Some(result) if result.ty != ty => Err(Error::ReturnMismatch(
            function.name.clone().unwrap_or_default(),
        )),
        _ => Ok(()),
    }
}

fn lower_stmt_expr(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    expr: &Expr,
    env: &mut Env,
) -> Result<(), Error> {
    match expr {
        Expr::If(if_expr) => lower_if_stmt(ctx, function, body, if_expr, env),
        Expr::While(while_expr) => lower_while(ctx, function, body, while_expr, env),
        Expr::Loop(loop_expr) => lower_loop(ctx, function, body, loop_expr, env),
        Expr::Break(brk) => lower_break(body, brk),
        Expr::ForLoop(for_expr) => lower_for(ctx, function, body, for_expr, env),
        // Some builtins write rather than produce, so they only make sense here.
        Expr::Call(call) => super::call::lower_call_stmt(ctx, function, body, call, env),
        Expr::MethodCall(call) => {
            super::method::lower_method_any(ctx, function, body, call, env, None).map(|_| ())
        }
        Expr::Continue(cont) => lower_continue(body, cont),
        // `unsafe` is for `rustc`, and an older `get_mut` wanted it. The shader
        // has nothing to say about it.
        Expr::Block(syn::ExprBlock { block, .. }) | Expr::Unsafe(syn::ExprUnsafe { block, .. }) => {
            env.push_scope();
            let _ = lower_block(ctx, function, body, block, env)?;
            env.pop_scope();
            Ok(())
        }
        Expr::Paren(inner) => lower_stmt_expr(ctx, function, body, &inner.expr, env),
        Expr::Group(inner) => lower_stmt_expr(ctx, function, body, &inner.expr, env),
        other => {
            let _ = lower_expr(ctx, function, body, other, env)?;
            Ok(())
        }
    }
}

fn lower_while(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    while_expr: &syn::ExprWhile,
    env: &mut Env,
) -> Result<(), Error> {
    if while_expr.label.is_some() {
        return Err(Error::LoopLabel);
    }
    let mut loop_body = Block::new();
    env.push_scope();
    let (condition, _) = lower_expr(ctx, function, &mut loop_body, &while_expr.cond, env)?;
    let mut accept = Block::new();
    let _ = lower_block(ctx, function, &mut accept, &while_expr.body, env)?;
    let mut reject = Block::new();
    reject.push(Statement::Break, Span::UNDEFINED);
    loop_body.push(
        Statement::If {
            condition,
            accept,
            reject,
        },
        Span::UNDEFINED,
    );
    env.pop_scope();
    body.push(
        Statement::Loop {
            body: loop_body,
            continuing: Block::new(),
            break_if: None,
        },
        Span::UNDEFINED,
    );
    Ok(())
}

fn lower_loop(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    loop_expr: &syn::ExprLoop,
    env: &mut Env,
) -> Result<(), Error> {
    if loop_expr.label.is_some() {
        return Err(Error::LoopLabel);
    }
    let mut loop_body = Block::new();
    env.push_scope();
    let _ = lower_block(ctx, function, &mut loop_body, &loop_expr.body, env)?;
    env.pop_scope();
    body.push(
        Statement::Loop {
            body: loop_body,
            continuing: Block::new(),
            break_if: None,
        },
        Span::UNDEFINED,
    );
    Ok(())
}

/// `for i in a..b` / `a..=b`.
///
/// Rust evaluates the range once, before the loop, so both ends are lowered
/// into the enclosing block; Naga keeps those expressions valid inside the
/// loop. The counter is a local, the bound check opens the body, and the step
/// goes in the loop's `continuing` block so `continue` still reaches it.
fn lower_for(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    for_expr: &syn::ExprForLoop,
    env: &mut Env,
) -> Result<(), Error> {
    if for_expr.label.is_some() {
        return Err(Error::LoopLabel);
    }
    let Expr::Range(range) = strip_parens(&for_expr.expr) else {
        return Err(Error::UnsupportedStmt("`for` over a non-range".into()));
    };
    let (Some(start), Some(end)) = (range.start.as_deref(), range.end.as_deref()) else {
        return Err(Error::UnsupportedStmt("unbounded range".into()));
    };
    let name = match &*for_expr.pat {
        Pat::Ident(ident) if ident.by_ref.is_none() => ident.ident.to_string(),
        Pat::Wild(_) => "_".to_string(),
        _ => return Err(Error::PatternParam),
    };

    // Whichever end is not an untyped literal fixes the counter's type, the
    // same way `0..n` infers in Rust.
    let (init, ty, bound) = if is_untyped_int(start) && !is_untyped_int(end) {
        let (bound, ty) = lower_expr(ctx, function, body, end, env)?;
        let hint = ctx.shape(ty).int_hint();
        let (init, init_ty) = lower_expr_hinted(ctx, function, body, start, env, hint)?;
        (init, init_ty, bound)
    } else {
        let (init, ty) = lower_expr(ctx, function, body, start, env)?;
        let hint = ctx.shape(ty).int_hint();
        let (bound, _) = lower_expr_hinted(ctx, function, body, end, env, hint)?;
        (init, ty, bound)
    };
    match ctx.shape(ty).elem_kind() {
        Some(naga::ScalarKind::Sint | naga::ScalarKind::Uint) => {}
        _ => {
            return Err(Error::UnsupportedStmt(
                "`for` over a non-integer range".into(),
            ))
        }
    }

    let counter = function.local_variables.append(
        LocalVariable {
            name: Some(name.clone()),
            ty,
            init: None,
        },
        Span::UNDEFINED,
    );
    let pointer = function
        .expressions
        .append(Expression::LocalVariable(counter), Span::UNDEFINED);
    body.push(
        Statement::Store {
            pointer,
            value: init,
        },
        Span::UNDEFINED,
    );

    env.push_scope();
    env.push(name, Slot::Ptr(pointer), ty);

    // `if !(i < end) { break; }`
    let mut loop_body = Block::new();
    let current = emit(function, &mut loop_body, Expression::Load { pointer })?;
    let condition = emit(
        function,
        &mut loop_body,
        Expression::Binary {
            op: match range.limits {
                syn::RangeLimits::Closed(_) => naga::BinaryOperator::LessEqual,
                syn::RangeLimits::HalfOpen(_) => naga::BinaryOperator::Less,
            },
            left: current,
            right: bound,
        },
    )?;
    let mut reject = Block::new();
    reject.push(Statement::Break, Span::UNDEFINED);
    loop_body.push(
        Statement::If {
            condition,
            accept: Block::new(),
            reject,
        },
        Span::UNDEFINED,
    );
    let _ = lower_block(ctx, function, &mut loop_body, &for_expr.body, env)?;

    // `continuing { i += 1; }`
    let mut continuing = Block::new();
    let step = emit(function, &mut continuing, Expression::Load { pointer })?;
    let one = function
        .expressions
        .append(Expression::Literal(int_one(ctx, ty)), Span::UNDEFINED);
    let next = emit(
        function,
        &mut continuing,
        Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: step,
            right: one,
        },
    )?;
    continuing.push(
        Statement::Store {
            pointer,
            value: next,
        },
        Span::UNDEFINED,
    );
    env.pop_scope();

    body.push(
        Statement::Loop {
            body: loop_body,
            continuing,
            break_if: None,
        },
        Span::UNDEFINED,
    );
    Ok(())
}

fn int_one(ctx: &Context, ty: Handle<Type>) -> naga::Literal {
    match ctx.shape(ty).scalar().map(|s| s.kind) {
        Some(naga::ScalarKind::Uint) => naga::Literal::U32(1),
        _ => naga::Literal::I32(1),
    }
}

fn strip_parens(expr: &Expr) -> &Expr {
    match expr {
        Expr::Paren(inner) => strip_parens(&inner.expr),
        Expr::Group(inner) => strip_parens(&inner.expr),
        other => other,
    }
}

fn lower_break(body: &mut Block, brk: &syn::ExprBreak) -> Result<(), Error> {
    if brk.label.is_some() {
        return Err(Error::LoopLabel);
    }
    if brk.expr.is_some() {
        return Err(Error::BreakValue);
    }
    body.push(Statement::Break, Span::UNDEFINED);
    Ok(())
}

fn lower_continue(body: &mut Block, cont: &syn::ExprContinue) -> Result<(), Error> {
    if cont.label.is_some() {
        return Err(Error::LoopLabel);
    }
    body.push(Statement::Continue, Span::UNDEFINED);
    Ok(())
}

fn lower_local(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    local: &Local,
    env: &mut Env,
) -> Result<(), Error> {
    // `let _ = e;` evaluates for effect and binds nothing, as in Rust.
    if matches!(strip_pat(&local.pat), Pat::Wild(_)) {
        if let Some(init) = &local.init {
            let _ = lower_expr(ctx, function, body, &init.expr, env)?;
        }
        return Ok(());
    }

    let (name, annot) = bind_ident_pat(&local.pat)?;
    if let Some(init) = &local.init {
        if init.diverge.is_none() && bind_borrow(ctx, function, body, &name, &init.expr, env)? {
            return Ok(());
        }
    }
    let annot = annot.map(|ty| ctx.lower_type(ty)).transpose()?;

    // `let x: T;` declares the slot and leaves it to a later assignment, the
    // way WGSL's bare `var x: T;` does. Without an annotation there is nothing
    // to infer the type from.
    let init = match &local.init {
        Some(init) if init.diverge.is_some() => {
            return Err(Error::UnsupportedStmt("let else".into()))
        }
        Some(init) => Some(&*init.expr),
        None if annot.is_some() => None,
        None => return Err(Error::MissingLetInit),
    };

    let (value, ty) = match init {
        // A ray query is not a constructible value. `RayQuery::default()` is
        // how Rust names the local; the query itself starts at `initialize`.
        Some(expr) if is_ray_query_default(expr) => {
            let ty = super::ray::parse_ray_type(ctx, "ray_query").expect("ray_query");
            if matches!(annot, Some(want) if want != ty) {
                return Err(Error::TypeMismatch);
            }
            (None, ty)
        }
        Some(expr) => {
            let hint = annot.and_then(|ty| ctx.shape(ty).int_hint());
            let (value, value_ty) = lower_expr_hinted(ctx, function, body, expr, env, hint)?;
            if matches!(annot, Some(ty) if ty != value_ty) {
                return Err(Error::TypeMismatch);
            }
            (Some(value), value_ty)
        }
        None => (None, annot.expect("checked above")),
    };

    // A binding that is never assigned and never passed as storage is a value.
    // A local for it forces a store and a reload, and that extra function
    // memory is what Windows lavapipe crashes on or miscompiles.
    if !ctx.addressed.contains(&name) {
        if let Some(value) = value {
            env.push_in(
                name,
                Slot::Value(value),
                ty,
                false,
                naga::AddressSpace::Function,
            );
            return Ok(());
        }
    }

    let local_var = function.local_variables.append(
        LocalVariable {
            name: Some(name.clone()),
            ty,
            init: None,
        },
        Span::UNDEFINED,
    );
    let pointer = function
        .expressions
        .append(Expression::LocalVariable(local_var), Span::UNDEFINED);
    if let Some(value) = value {
        body.push(Statement::Store { pointer, value }, Span::UNDEFINED);
    }
    env.push(name, Slot::Ptr(pointer), ty);
    Ok(())
}

/// `let p = &mut place;` names the place: `p.x = 1.0` stores through it and
/// `p.x` loads from it, as `place.x` would. Nothing is copied, and the place's
/// index is worked out once, where the borrow is. Naga has no local that holds
/// a pointer, so `p` is the pointer itself rather than a variable. A handle
/// (`&tex`) and a borrow of a value that is not a place are left to `let`.
fn bind_borrow(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    name: &str,
    init: &Expr,
    env: &mut Env,
) -> Result<bool, Error> {
    let Expr::Reference(reference) = strip_parens(init) else {
        return Ok(false);
    };
    let Some(place) = super::place::lower_place(ctx, function, body, &reference.expr, env)? else {
        return Ok(false);
    };
    if super::texture::is_handle(ctx, place.ty) {
        return Ok(false);
    }
    let mutable = reference.mutability.is_some();
    if mutable && !place.writable {
        return Err(Error::AssignToReadonly(place.root));
    }
    env.push_in(
        name.to_string(),
        Slot::Ptr(place.pointer),
        place.ty,
        mutable,
        place.space,
    );
    Ok(true)
}

/// `ray_query::default()`, the checkable spelling of an uninitialized query local.
/// Names assigned, borrowed, or passed as storage anywhere in `block`.
pub(super) fn addressed_names(block: &SynBlock) -> HashSet<String> {
    let mut found = Addressed::default();
    found.visit_block(block);
    found.names
}

#[derive(Default)]
struct Addressed {
    names: HashSet<String>,
}

impl<'ast> Visit<'ast> for Addressed {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        match expr {
            Expr::Assign(assign) => note_root(&assign.left, &mut self.names),
            Expr::Binary(bin) if is_compound_assign(&bin.op) => {
                note_root(&bin.left, &mut self.names);
            }
            Expr::Reference(reference) => note_root(&reference.expr, &mut self.names),
            // A ray query's operations change it, so it has to be storage.
            Expr::MethodCall(call) if super::ray::is_ray_method(&call.method.to_string()) => {
                note_root(&call.receiver, &mut self.names);
            }
            _ => {}
        }
        visit::visit_expr(self, expr);
    }
}

fn is_compound_assign(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_)
    )
}

fn note_root(expr: &Expr, names: &mut HashSet<String>) {
    match expr {
        Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                names.insert(ident.to_string());
            }
        }
        Expr::Field(field) => note_root(&field.base, names),
        Expr::Index(index) => note_root(&index.expr, names),
        Expr::Paren(inner) => note_root(&inner.expr, names),
        Expr::Group(inner) => note_root(&inner.expr, names),
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
            note_root(&unary.expr, names);
        }
        Expr::MethodCall(call) if super::place::is_get_mut(call) => {
            note_root(&call.receiver, names);
        }
        Expr::Unsafe(block) => {
            if let Some(inner) = super::place::single_expr(&block.block) {
                note_root(inner, names);
            }
        }
        _ => {}
    }
}

fn is_ray_query_default(expr: &Expr) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    if !call.args.is_empty() {
        return false;
    }
    let Expr::Path(path) = call.func.as_ref() else {
        return false;
    };
    if path.qself.is_some() || path.path.segments.len() != 2 {
        return false;
    }
    let ty = &path.path.segments[0].ident;
    (ty == "RayQuery" || ty == "ray_query") && path.path.segments[1].ident == "default"
}

fn strip_pat(pat: &Pat) -> &Pat {
    match pat {
        Pat::Type(inner) => strip_pat(&inner.pat),
        Pat::Paren(inner) => strip_pat(&inner.pat),
        other => other,
    }
}

fn bind_ident_pat(pat: &Pat) -> Result<(String, Option<&SynType>), Error> {
    match pat {
        Pat::Ident(ident) if ident.by_ref.is_none() => Ok((ident.ident.to_string(), None)),
        Pat::Type(pat_ty) => {
            let (name, _) = bind_ident_pat(&pat_ty.pat)?;
            Ok((name, Some(&*pat_ty.ty)))
        }
        _ => Err(Error::PatternParam),
    }
}

fn lower_if_stmt(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    if_expr: &syn::ExprIf,
    env: &mut Env,
) -> Result<(), Error> {
    let (condition, _) = lower_expr(ctx, function, body, &if_expr.cond, env)?;
    let mut accept = Block::new();
    env.push_scope();
    let _ = lower_block(ctx, function, &mut accept, &if_expr.then_branch, env)?;
    env.pop_scope();
    let mut reject = Block::new();
    if let Some((_, else_expr)) = &if_expr.else_branch {
        env.push_scope();
        match else_expr.as_ref() {
            Expr::Block(b) => {
                let _ = lower_block(ctx, function, &mut reject, &b.block, env)?;
            }
            Expr::If(inner) => lower_if_stmt(ctx, function, &mut reject, inner, env)?,
            other => {
                let _ = lower_expr(ctx, function, &mut reject, other, env)?;
            }
        }
        env.pop_scope();
    }
    body.push(
        Statement::If {
            condition,
            accept,
            reject,
        },
        Span::UNDEFINED,
    );
    Ok(())
}

pub(super) fn lower_if_expr(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    if_expr: &syn::ExprIf,
    env: &mut Env,
    hint: Option<naga::Scalar>,
) -> Result<Typed, Error> {
    lower_if_any(ctx, function, body, if_expr, env, hint)?.ok_or(Error::MissingBlockValue)
}

/// An `if` with an `else`, whose value is its branches' if they have one: an
/// `if` whose branches both end in a call that returns nothing is a statement.
fn lower_if_any(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    if_expr: &syn::ExprIf,
    env: &mut Env,
    hint: Option<naga::Scalar>,
) -> Result<Option<Typed>, Error> {
    let else_expr = if_expr
        .else_branch
        .as_ref()
        .map(|(_, e)| e.as_ref())
        .ok_or(Error::IfExprMissingElse)?;
    let (condition, _) = lower_expr(ctx, function, body, &if_expr.cond, env)?;
    let mut accept = Block::new();
    env.push_scope();
    let then_tail =
        lower_block_hinted(ctx, function, &mut accept, &if_expr.then_branch, env, hint)?;
    env.pop_scope();
    let mut reject = Block::new();
    env.push_scope();
    let else_tail = lower_tail(ctx, function, &mut reject, else_expr, env, hint)?;
    env.pop_scope();
    let ((then_val, then_ty), (else_val, else_ty)) = match (then_tail, else_tail) {
        (Some(then), Some(other)) => (then, other),
        (None, None) => {
            body.push(
                Statement::If {
                    condition,
                    accept,
                    reject,
                },
                Span::UNDEFINED,
            );
            return Ok(None);
        }
        _ => return Err(Error::MissingBlockValue),
    };
    if then_ty != else_ty {
        return Err(Error::TypeMismatch);
    }
    let ty = then_ty;
    let local = function.local_variables.append(
        LocalVariable {
            name: None,
            ty,
            init: None,
        },
        Span::UNDEFINED,
    );
    let pointer = function
        .expressions
        .append(Expression::LocalVariable(local), Span::UNDEFINED);
    accept.push(
        Statement::Store {
            pointer,
            value: then_val,
        },
        Span::UNDEFINED,
    );
    reject.push(
        Statement::Store {
            pointer,
            value: else_val,
        },
        Span::UNDEFINED,
    );
    body.push(
        Statement::If {
            condition,
            accept,
            reject,
        },
        Span::UNDEFINED,
    );
    let loaded = emit(function, body, Expression::Load { pointer })?;
    Ok(Some((loaded, ty)))
}
