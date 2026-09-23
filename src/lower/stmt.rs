use naga::{Block, Expression, Function, Handle, LocalVariable, Span, Statement, Type};
use syn::{Block as SynBlock, Expr, Local, Pat, Stmt, Type as SynType};

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
    let mut tail = None;
    for (i, stmt) in block.stmts.iter().enumerate() {
        let last = i + 1 == block.stmts.len();
        match stmt {
            Stmt::Local(local) => {
                lower_local(ctx, function, body, local, env)?;
                tail = None;
            }
            Stmt::Expr(Expr::Return(ret), _) => {
                let value = match ret.expr.as_deref() {
                    Some(expr) => Some(lower_expr(ctx, function, body, expr, env)?.0),
                    None => None,
                };
                body.push(Statement::Return { value }, Span::UNDEFINED);
                tail = None;
            }
            Stmt::Expr(expr, semi) => {
                if last && semi.is_none() && yields_value(expr) {
                    tail = Some(lower_expr(ctx, function, body, expr, env)?);
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
        Expr::Continue(cont) => lower_continue(body, cont),
        Expr::Block(b) => {
            env.push_scope();
            let _ = lower_block(ctx, function, body, &b.block, env)?;
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
        // `ray_query` is not a constructible value. `ray_query::default()` is
        // how Rust names the local; the query itself starts at `rayQueryInitialize`.
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

/// `ray_query::default()`, the checkable spelling of an uninitialized query local.
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
    path.path.segments[0].ident == "ray_query" && path.path.segments[1].ident == "default"
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
) -> Result<Typed, Error> {
    let else_expr = if_expr
        .else_branch
        .as_ref()
        .map(|(_, e)| e.as_ref())
        .ok_or(Error::IfExprMissingElse)?;
    let (condition, _) = lower_expr(ctx, function, body, &if_expr.cond, env)?;
    let mut accept = Block::new();
    env.push_scope();
    let (then_val, then_ty) = lower_block(ctx, function, &mut accept, &if_expr.then_branch, env)?
        .ok_or(Error::MissingBlockValue)?;
    env.pop_scope();
    let mut reject = Block::new();
    env.push_scope();
    let (else_val, else_ty) = match else_expr {
        Expr::Block(b) => lower_block(ctx, function, &mut reject, &b.block, env)?
            .ok_or(Error::MissingBlockValue)?,
        Expr::If(inner) => lower_if_expr(ctx, function, &mut reject, inner, env)?,
        other => lower_expr(ctx, function, &mut reject, other, env)?,
    };
    env.pop_scope();
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
    Ok((loaded, ty))
}
