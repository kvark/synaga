//! `match`: on an integer, an enum or a flags set it is Naga's `Switch`, and
//! on a `bool` an `if`.
//!
//! A pattern is a value known before the shader runs, a literal, a `const`,
//! an enum's variant or a set's flag, or several of those with `|`, or `_`,
//! or a name, which binds the value. Rust takes the first arm that matches,
//! and a `switch` the one whose case is the value, so an arm or a value that
//! an earlier arm already takes is dropped, as `rustc` warns it is never
//! reached. A `match` with no `_` covers every value `rustc` allows, and its
//! last arm is also the `switch`'s default, which Naga requires.
//!
//! A `break` in a `switch` leaves the `switch`, where Rust's leaves the loop
//! around the `match`. A `match` with an arm that breaks out of a loop is an
//! `if` chain instead.

use naga::{Block, Expression, Function, Handle, LocalVariable, Scalar, ScalarKind};
use naga::{Statement, SwitchCase, SwitchValue};
use syn::visit::{self, Visit};
use syn::{Expr, ExprMatch, Pat};

use super::emit::emit;
use super::env::{Env, Slot};
use super::expr::lower_expr;
use super::{Context, Typed};
use crate::Error;

/// Where a `match` is, which is what its arms do with their values.
pub(super) enum Position {
    /// A statement: the values go nowhere.
    Statement,
    /// A value, which each arm stores for the `match` to produce. `hint` is
    /// where it goes, for its literals.
    Value(Option<Scalar>),
    /// The end of a function that returns: each arm returns its own.
    Return,
}

/// What one arm matches.
struct Arm<'a> {
    values: Vec<SwitchValue>,
    /// `_` or a name: everything no earlier arm takes.
    default: bool,
    /// The name a binding pattern gives the value.
    binding: Option<String>,
    body: &'a Expr,
}

/// Lower a `match`. What it produces is a value only in value position, and
/// only if an arm has one: arms that end in a call returning nothing make a
/// statement of it, as for an `if`.
pub(super) fn lower_match(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    matched: &ExprMatch,
    env: &mut Env,
    position: Position,
) -> Result<Option<Typed>, Error> {
    let (selector, ty) = lower_expr(ctx, function, body, &matched.expr, env)?;
    let kind = match ctx.module.types[ty].inner {
        naga::TypeInner::Scalar(scalar) if scalar.width == 4 || scalar.kind == ScalarKind::Bool => {
            scalar.kind
        }
        _ => return Err(Error::MatchOn),
    };
    if kind == ScalarKind::Float {
        return Err(Error::MatchOn);
    }
    let arms = arms(ctx, matched, ty, kind)?;
    let chain = kind == ScalarKind::Bool || matched.arms.iter().any(|a| breaks_out(&a.body));

    // Each arm's block, and its value if it has one.
    let mut blocks = Vec::with_capacity(arms.len());
    let mut value_ty = None;
    for arm in &arms {
        let mut block = Block::new();
        env.push_scope();
        if let Some(name) = &arm.binding {
            env.push(name.clone(), Slot::Value(selector), ty);
        }
        let value = match position {
            Position::Return => {
                super::stmt::lower_returning_expr(ctx, function, &mut block, arm.body, env)?;
                None
            }
            Position::Statement => {
                super::stmt::lower_arm(ctx, function, &mut block, arm.body, env, None)?;
                None
            }
            Position::Value(hint) => {
                let hint = hint.or_else(|| value_ty.and_then(|t| ctx.shape(t).int_hint()));
                let value = super::stmt::lower_arm(ctx, function, &mut block, arm.body, env, hint)?;
                if let Some((_, arm_ty)) = value {
                    if value_ty.is_some_and(|t| t != arm_ty) {
                        return Err(Error::TypeMismatch);
                    }
                    value_ty = Some(arm_ty);
                }
                value
            }
        };
        env.pop_scope();
        blocks.push((block, value));
    }

    // In value position the arms store what they produce, unless none does.
    let local = match value_ty {
        Some(value_ty) => {
            let local = function.local_variables.append(
                LocalVariable {
                    name: None,
                    ty: value_ty,
                    init: None,
                },
                ctx.span,
            );
            let pointer = function
                .expressions
                .append(Expression::LocalVariable(local), ctx.span);
            for (block, value) in &mut blocks {
                match value {
                    Some((value, _)) => block.push(
                        Statement::Store {
                            pointer,
                            value: *value,
                        },
                        ctx.span,
                    ),
                    // An arm without a value has to leave by a jump.
                    None if super::stmt::always_jumps(block) => {}
                    None => return Err(Error::MissingBlockValue),
                }
            }
            Some((pointer, value_ty))
        }
        None => None,
    };

    let bodies = arms.iter().zip(blocks.into_iter().map(|(block, _)| block));
    let statement = match chain {
        true => if_chain(ctx, function, selector, ty, bodies)?,
        false => switch(selector, bodies),
    };
    body.push(statement, ctx.span);
    match local {
        Some((pointer, ty)) => {
            let value = emit(ctx, function, body, Expression::Load { pointer })?;
            Ok(Some((value, ty)))
        }
        None => Ok(None),
    }
}

/// The arms, each with the values no earlier arm takes, and those an arm is
/// left with. One that is left with none is never reached, and is dropped.
fn arms<'a>(
    ctx: &mut Context,
    matched: &'a ExprMatch,
    ty: Handle<naga::Type>,
    kind: ScalarKind,
) -> Result<Vec<Arm<'a>>, Error> {
    let mut arms: Vec<Arm> = Vec::new();
    let mut taken = Vec::new();
    for arm in &matched.arms {
        if arm.guard.is_some() {
            return Err(Error::MatchGuard);
        }
        // Everything after a catch-all is unreachable.
        if arms.iter().any(|a| a.default) {
            break;
        }
        let mut found = Arm {
            values: Vec::new(),
            default: false,
            binding: None,
            body: &arm.body,
        };
        pattern(ctx, &arm.pat, ty, kind, &mut found)?;
        found.values.retain(|value| !taken.contains(value));
        taken.extend(found.values.iter().copied());
        if found.values.is_empty() && !found.default {
            continue;
        }
        arms.push(found);
    }
    // `rustc` has checked that the arms cover every value, so the last one
    // takes whatever is left, which is nothing a checked shader can reach.
    match arms.last_mut() {
        Some(last) => last.default = true,
        None => return Err(Error::MatchOn),
    }
    Ok(arms)
}

/// What `pat` matches, added to `arm`.
fn pattern(
    ctx: &mut Context,
    pat: &Pat,
    ty: Handle<naga::Type>,
    kind: ScalarKind,
    arm: &mut Arm,
) -> Result<(), Error> {
    match pat {
        Pat::Or(or) => {
            for case in &or.cases {
                pattern(ctx, case, ty, kind, arm)?;
            }
            Ok(())
        }
        Pat::Paren(inner) => pattern(ctx, &inner.pat, ty, kind, arm),
        Pat::Wild(_) => {
            arm.default = true;
            Ok(())
        }
        Pat::Lit(lit) => {
            let value = match &lit.lit {
                syn::Lit::Int(int) => int.base10_parse::<i64>().map_err(Error::from)?,
                syn::Lit::Bool(b) if kind == ScalarKind::Bool => i64::from(b.value),
                _ => return Err(Error::MatchPattern),
            };
            arm.values.push(switch_value(value, kind)?);
            Ok(())
        }
        // A name is a variant or a `const` if one is in scope, and a binding
        // otherwise, as in Rust.
        Pat::Ident(ident) if ident.by_ref.is_none() && ident.subpat.is_none() => {
            let name = ident.ident.to_string();
            let variant = ctx.module.types[ty]
                .name
                .as_ref()
                .and_then(|enumeration| ctx.scope.enum_variant(enumeration, &name));
            if let Some(value) = variant {
                arm.values.push(switch_value(value.into(), kind)?);
            } else if let Some(value) = const_value(ctx, std::slice::from_ref(&name))? {
                arm.values.push(switch_value(value, kind)?);
            } else if ident.mutability.is_some() {
                return Err(Error::MatchPattern);
            } else {
                arm.default = true;
                arm.binding = Some(name);
            }
            Ok(())
        }
        Pat::Path(path) if path.qself.is_none() => {
            let segments = super::path_segments(&path.path);
            let value = named_value(ctx, &segments, ty)?.ok_or(Error::MatchPattern)?;
            arm.values.push(switch_value(value, kind)?);
            Ok(())
        }
        _ => Err(Error::MatchPattern),
    }
}

/// `Mode::Depth`, `Flags::SPACE`, `RayQueryIntersection::None` or a `const`
/// reached through a module.
fn named_value(
    ctx: &mut Context,
    path: &[String],
    ty: Handle<naga::Type>,
) -> Result<Option<i64>, Error> {
    if let [.., owner, item] = path {
        if let Some(value) = ctx.scope.enum_variant(owner, item) {
            return Ok(Some(value.into()));
        }
        if let Some(value) = ctx.scope.flags.get(owner).and_then(|f| f.flag(item)) {
            return Ok(Some(value.into()));
        }
        if let Some(value) = super::ray::typed_const(owner, item) {
            return Ok(Some(value.into()));
        }
    }
    let _ = ty;
    const_value(ctx, path)
}

/// The value of the `const` `path` names, if it names one that is an integer.
fn const_value(ctx: &mut Context, path: &[String]) -> Result<Option<i64>, Error> {
    let Some(index) = ctx.constant(path)? else {
        return Ok(None);
    };
    let mut init = ctx.consts[index].init_expr;
    loop {
        match ctx.module.global_expressions[init] {
            Expression::Literal(naga::Literal::U32(v)) => return Ok(Some(v.into())),
            Expression::Literal(naga::Literal::I32(v)) => return Ok(Some(v.into())),
            Expression::Constant(other) => init = ctx.module.constants[other].init,
            _ => return Err(Error::MatchPattern),
        }
    }
}

/// A case value of the selector's kind, if `value` is one.
fn switch_value(value: i64, kind: ScalarKind) -> Result<SwitchValue, Error> {
    match kind {
        ScalarKind::Uint => u32::try_from(value)
            .map(SwitchValue::U32)
            .map_err(|_| Error::MatchPattern),
        ScalarKind::Sint => i32::try_from(value)
            .map(SwitchValue::I32)
            .map_err(|_| Error::MatchPattern),
        // A `bool` becomes an `if`, which only needs to know which is which.
        _ => Ok(SwitchValue::U32(value as u32)),
    }
}

/// Naga's `switch`: each arm's values fall through to the last, which holds
/// the arm, and the arm that takes the rest holds the default too.
fn switch<'a, 'b: 'a>(
    selector: Handle<Expression>,
    arms: impl Iterator<Item = (&'a Arm<'b>, Block)>,
) -> Statement {
    let mut cases = Vec::new();
    for (arm, block) in arms {
        let mut values = arm.values.clone();
        if arm.default {
            values.push(SwitchValue::Default);
        }
        let last = values.len() - 1;
        let mut block = Some(block);
        for (i, value) in values.into_iter().enumerate() {
            let body = match i == last {
                true => block.take().expect("the last value holds the arm"),
                false => Block::new(),
            };
            cases.push(SwitchCase {
                value,
                body,
                fall_through: i != last,
            });
        }
    }
    Statement::Switch { selector, cases }
}

/// `if selector == a || selector == b { .. } else if ..`, for a `bool`, or a
/// `match` whose arms leave the loop around it.
fn if_chain<'a, 'b: 'a>(
    ctx: &mut Context,
    function: &mut Function,
    selector: Handle<Expression>,
    ty: Handle<naga::Type>,
    arms: impl Iterator<Item = (&'a Arm<'b>, Block)>,
) -> Result<Statement, Error> {
    let arms: Vec<_> = arms.collect();
    // Built from the last arm out, which is the `else` of the one before it.
    let mut rest: Option<Block> = None;
    for (arm, block) in arms.into_iter().rev() {
        let Some(mut reject) = rest.take() else {
            rest = Some(block);
            continue;
        };
        let mut condition_block = Block::new();
        let condition = match_condition(ctx, function, &mut condition_block, selector, ty, arm)?;
        condition_block.push(
            Statement::If {
                condition,
                accept: block,
                reject: std::mem::take(&mut reject),
            },
            ctx.span,
        );
        rest = Some(condition_block);
    }
    Ok(Statement::Block(rest.unwrap_or_default()))
}

/// Whether `selector` is one of `arm`'s values.
fn match_condition(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    selector: Handle<Expression>,
    ty: Handle<naga::Type>,
    arm: &Arm,
) -> Result<Handle<Expression>, Error> {
    let bool_ty = ctx.module.types[ty].inner == naga::TypeInner::Scalar(Scalar::BOOL);
    let mut condition: Option<Handle<Expression>> = None;
    for value in &arm.values {
        let test = match (bool_ty, *value) {
            (true, SwitchValue::U32(1)) => selector,
            (true, _) => emit(
                ctx,
                function,
                body,
                Expression::Unary {
                    op: naga::UnaryOperator::LogicalNot,
                    expr: selector,
                },
            )?,
            (false, value) => {
                let literal = match value {
                    SwitchValue::I32(v) => naga::Literal::I32(v),
                    SwitchValue::U32(v) => naga::Literal::U32(v),
                    SwitchValue::Default => unreachable!("a default is not a value"),
                };
                let literal = function
                    .expressions
                    .append(Expression::Literal(literal), ctx.span);
                emit(
                    ctx,
                    function,
                    body,
                    Expression::Binary {
                        op: naga::BinaryOperator::Equal,
                        left: selector,
                        right: literal,
                    },
                )?
            }
        };
        condition = Some(match condition {
            None => test,
            Some(left) => emit(
                ctx,
                function,
                body,
                Expression::Binary {
                    op: naga::BinaryOperator::LogicalOr,
                    left,
                    right: test,
                },
            )?,
        });
    }
    condition.ok_or(Error::MatchPattern)
}

/// Does `expr` break out of a loop around it, rather than one of its own?
fn breaks_out(expr: &Expr) -> bool {
    struct Finder(bool);
    impl<'ast> Visit<'ast> for Finder {
        fn visit_expr(&mut self, expr: &'ast Expr) {
            match expr {
                Expr::Break(_) => self.0 = true,
                // A loop inside takes the breaks inside it.
                Expr::Loop(_) | Expr::While(_) | Expr::ForLoop(_) | Expr::Closure(_) => {}
                _ => visit::visit_expr(self, expr),
            }
        }
    }
    let mut finder = Finder(false);
    finder.visit_expr(expr);
    finder.0
}
