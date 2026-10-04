//! Subgroup operations: `subgroup_add(x)` and the rest, WGSL's builtins in
//! snake case.
//!
//! Each is a statement that writes a result, as an atomic is: the call becomes
//! the statement, and its value is the result expression the statement names.

use naga::{
    Block, CollectiveOperation, Direction, Expression, Function, GatherMode, Handle, Literal,
    Scalar, ScalarKind, Statement, SubgroupOperation, Type, VectorSize,
};
use syn::{Expr, ExprCall};

use super::env::Env;
use super::expr::lower_expr_hinted;
use super::{Context, Shape, Typed};
use crate::Error;

enum Op {
    Collective(SubgroupOperation, CollectiveOperation),
    Ballot,
    BroadcastFirst,
    QuadSwap(Direction),
    /// A gather from the lane its second argument picks.
    Indexed(Indexed),
}

#[derive(Clone, Copy)]
enum Indexed {
    Broadcast,
    Shuffle,
    ShuffleXor,
    ShuffleUp,
    ShuffleDown,
    QuadBroadcast,
}

impl Indexed {
    fn mode(self, index: Handle<Expression>) -> GatherMode {
        match self {
            Indexed::Broadcast => GatherMode::Broadcast(index),
            Indexed::Shuffle => GatherMode::Shuffle(index),
            Indexed::ShuffleXor => GatherMode::ShuffleXor(index),
            Indexed::ShuffleUp => GatherMode::ShuffleUp(index),
            Indexed::ShuffleDown => GatherMode::ShuffleDown(index),
            Indexed::QuadBroadcast => GatherMode::QuadBroadcast(index),
        }
    }

    /// A broadcast reads one lane for the whole subgroup, so WGSL wants the
    /// lane known before the shader runs. A shuffle's may differ per lane.
    fn constant(self) -> bool {
        matches!(self, Indexed::Broadcast | Indexed::QuadBroadcast)
    }
}

/// What the operation takes, which WGSL's signatures say and Naga holds a
/// module to.
#[derive(Clone, Copy, PartialEq)]
enum Operand {
    Bool,
    Number,
    Integer,
}

fn classify(name: &str) -> Option<Op> {
    use CollectiveOperation as Co;
    use SubgroupOperation as Sg;
    Some(match name {
        "subgroup_all" | "subgroupAll" => Op::Collective(Sg::All, Co::Reduce),
        "subgroup_any" | "subgroupAny" => Op::Collective(Sg::Any, Co::Reduce),
        "subgroup_add" | "subgroupAdd" => Op::Collective(Sg::Add, Co::Reduce),
        "subgroup_mul" | "subgroupMul" => Op::Collective(Sg::Mul, Co::Reduce),
        "subgroup_min" | "subgroupMin" => Op::Collective(Sg::Min, Co::Reduce),
        "subgroup_max" | "subgroupMax" => Op::Collective(Sg::Max, Co::Reduce),
        "subgroup_and" | "subgroupAnd" => Op::Collective(Sg::And, Co::Reduce),
        "subgroup_or" | "subgroupOr" => Op::Collective(Sg::Or, Co::Reduce),
        "subgroup_xor" | "subgroupXor" => Op::Collective(Sg::Xor, Co::Reduce),
        "subgroup_exclusive_add" | "subgroupExclusiveAdd" => {
            Op::Collective(Sg::Add, Co::ExclusiveScan)
        }
        "subgroup_exclusive_mul" | "subgroupExclusiveMul" => {
            Op::Collective(Sg::Mul, Co::ExclusiveScan)
        }
        "subgroup_inclusive_add" | "subgroupInclusiveAdd" => {
            Op::Collective(Sg::Add, Co::InclusiveScan)
        }
        "subgroup_inclusive_mul" | "subgroupInclusiveMul" => {
            Op::Collective(Sg::Mul, Co::InclusiveScan)
        }
        "subgroup_ballot" | "subgroupBallot" => Op::Ballot,
        "subgroup_broadcast_first" | "subgroupBroadcastFirst" => Op::BroadcastFirst,
        "subgroup_broadcast" | "subgroupBroadcast" => Op::Indexed(Indexed::Broadcast),
        "subgroup_shuffle" | "subgroupShuffle" => Op::Indexed(Indexed::Shuffle),
        "subgroup_shuffle_xor" | "subgroupShuffleXor" => Op::Indexed(Indexed::ShuffleXor),
        "subgroup_shuffle_up" | "subgroupShuffleUp" => Op::Indexed(Indexed::ShuffleUp),
        "subgroup_shuffle_down" | "subgroupShuffleDown" => Op::Indexed(Indexed::ShuffleDown),
        "quad_broadcast" | "quadBroadcast" => Op::Indexed(Indexed::QuadBroadcast),
        "quad_swap_x" | "quadSwapX" => Op::QuadSwap(Direction::X),
        "quad_swap_y" | "quadSwapY" => Op::QuadSwap(Direction::Y),
        "quad_swap_diagonal" | "quadSwapDiagonal" => Op::QuadSwap(Direction::Diagonal),
        _ => return None,
    })
}

fn operand(op: SubgroupOperation) -> Operand {
    match op {
        SubgroupOperation::All | SubgroupOperation::Any => Operand::Bool,
        SubgroupOperation::And | SubgroupOperation::Or | SubgroupOperation::Xor => Operand::Integer,
        _ => Operand::Number,
    }
}

/// `name(args..)` if `name` is a subgroup operation, `Ok(None)` if it is not
/// one. `hint` is the integer scalar where the value goes, if that says one.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_subgroup_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    name: &str,
    call: &ExprCall,
    env: &mut Env,
    hint: Option<Scalar>,
) -> Result<Option<Typed>, Error> {
    let Some(op) = classify(name) else {
        return Ok(None);
    };
    let args: Vec<&Expr> = call.args.iter().collect();
    let arity = match op {
        Op::Indexed(_) => 2,
        _ => 1,
    };
    if args.len() != arity {
        return Err(Error::WrongArgCount(name.into()));
    }
    let wanted = match op {
        Op::Collective(op, _) => operand(op),
        Op::Ballot => Operand::Bool,
        _ => Operand::Number,
    };
    let hint = if wanted == Operand::Bool { None } else { hint };
    let (argument, ty) = lower_expr_hinted(ctx, function, body, args[0], env, hint)?;
    check_operand(ctx, name, ty, wanted)?;

    if let Op::Ballot = op {
        let result = function
            .expressions
            .append(Expression::SubgroupBallotResult, ctx.span);
        body.push(
            Statement::SubgroupBallot {
                result,
                predicate: Some(argument),
            },
            ctx.span,
        );
        let ty = ctx.intern_vector(VectorSize::Quad, Scalar::U32);
        return Ok(Some((result, ty)));
    }

    let mode = match op {
        Op::Collective(op, collective_op) => {
            let result = operation_result(function, ctx, ty);
            body.push(
                Statement::SubgroupCollectiveOperation {
                    op,
                    collective_op,
                    argument,
                    result,
                },
                ctx.span,
            );
            return Ok(Some((result, ty)));
        }
        Op::BroadcastFirst => GatherMode::BroadcastFirst,
        Op::QuadSwap(direction) => GatherMode::QuadSwap(direction),
        Op::Indexed(indexed) => {
            indexed.mode(lane(ctx, function, body, name, args[1], env, indexed)?)
        }
        Op::Ballot => unreachable!("handled above"),
    };
    let result = operation_result(function, ctx, ty);
    body.push(
        Statement::SubgroupGather {
            mode,
            argument,
            result,
        },
        ctx.span,
    );
    Ok(Some((result, ty)))
}

fn operation_result(
    function: &mut Function,
    ctx: &Context,
    ty: Handle<Type>,
) -> Handle<Expression> {
    function
        .expressions
        .append(Expression::SubgroupOperationResult { ty }, ctx.span)
}

/// Whether `ty` is what `name` takes: a `bool` for a vote or a ballot, an
/// integer scalar or vector for a bitwise reduction, and any numeric scalar or
/// vector otherwise.
fn check_operand(
    ctx: &Context,
    name: &str,
    ty: Handle<Type>,
    wanted: Operand,
) -> Result<(), Error> {
    let fits = match (ctx.shape(ty), wanted) {
        (Shape::Scalar(scalar), Operand::Bool) => scalar == Scalar::BOOL,
        (Shape::Scalar(scalar) | Shape::Vector(_, scalar), Operand::Integer) => {
            matches!(scalar.kind, ScalarKind::Sint | ScalarKind::Uint)
        }
        (Shape::Scalar(scalar) | Shape::Vector(_, scalar), Operand::Number) => matches!(
            scalar.kind,
            ScalarKind::Sint | ScalarKind::Uint | ScalarKind::Float
        ),
        _ => false,
    };
    match fits {
        true => Ok(()),
        false => Err(Error::BadOperandTypes(name.into())),
    }
}

/// The lane a gather reads, a `u32`. A broadcast's is a constant, folded to
/// the literal it comes to, since the whole subgroup reads the one lane.
#[allow(clippy::too_many_arguments)]
fn lane(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    name: &str,
    expr: &Expr,
    env: &mut Env,
    indexed: Indexed,
) -> Result<Handle<Expression>, Error> {
    if indexed.constant() {
        // A local is a value the shader computes, which the subgroup cannot
        // agree on beforehand.
        let lane = match env.reads_local(expr) {
            true => None,
            false => ctx.const_u32(expr)?,
        };
        let lane = lane.ok_or_else(|| Error::NonConstantLane(name.into()))?;
        return Ok(function
            .expressions
            .append(Expression::Literal(Literal::U32(lane)), ctx.span));
    }
    let (index, ty) = lower_expr_hinted(ctx, function, body, expr, env, Some(Scalar::U32))?;
    match ctx.as_scalar(ty) {
        Some(Scalar::U32) => Ok(index),
        _ => Err(Error::BadOperandTypes(name.into())),
    }
}
