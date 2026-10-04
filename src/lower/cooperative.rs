//! Cooperative matrices: `CoopMat8x8<f32, A>` and `CoopMat16x16<T, R>`,
//! WGSL's `coop_mat8x8` and `coop_mat16x16`.
//!
//! A matrix is loaded from a slice of its scalars, `&data[offset..]`, which is
//! a pointer to the element at `offset` to Naga, and stored back the same
//! way. `a.mul_add(b, c)` is Naga's multiply-add, and `+`, `-` and a scalar's
//! `*` are its operators, which `expr` takes from here.

use naga::{
    Block, CooperativeData, CooperativeRole, CooperativeSize, Expression, Function, Handle, Scalar,
    Statement, Type, TypeInner,
};
use syn::Expr;

use super::constant::strip_parens;
use super::emit::emit;
use super::env::Env;
use super::expr::{lower_expr, lower_expr_hinted};
use super::place::{index_expr, lower_place, IndexKind};
use super::{Context, Typed};
use crate::Error;

/// What a cooperative matrix type is: its size, square, its scalar and its
/// role.
pub(super) type Matrix = (CooperativeSize, Scalar, CooperativeRole);

/// `ty` as a cooperative matrix, if it is one.
pub(super) fn matrix(ctx: &Context, ty: Handle<Type>) -> Option<Matrix> {
    match ctx.module.types[ty].inner {
        TypeInner::CooperativeMatrix {
            columns,
            rows,
            scalar,
            role,
        } if columns == rows => Some((columns, scalar, role)),
        _ => None,
    }
}

/// The size a type name says, `CoopMat8x8` or WGSL's `coop_mat8x8`.
fn size_of(name: &str) -> Option<CooperativeSize> {
    match name {
        "CoopMat8x8" | "coop_mat8x8" => Some(CooperativeSize::Eight),
        "CoopMat16x16" | "coop_mat16x16" => Some(CooperativeSize::Sixteen),
        _ => None,
    }
}

/// Whether `name` is a cooperative matrix type's.
pub(super) fn is_type_name(name: &str) -> bool {
    size_of(name).is_some()
}

/// `CoopMat8x8<f32, A>`: the type, or `None` when `name` is no cooperative
/// matrix's.
pub(super) fn parse_type(
    ctx: &mut Context,
    name: &str,
    seg: &syn::PathSegment,
) -> Option<Result<Handle<Type>, Error>> {
    let size = size_of(name)?;
    Some(matrix_type(ctx, name, size, seg))
}

fn matrix_type(
    ctx: &mut Context,
    name: &str,
    size: CooperativeSize,
    seg: &syn::PathSegment,
) -> Result<Handle<Type>, Error> {
    let unsaid = || Error::CoopType(name.into());
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Err(unsaid());
    };
    let args: Vec<&syn::GenericArgument> = args.args.iter().collect();
    let [syn::GenericArgument::Type(scalar), syn::GenericArgument::Type(role)] = args[..] else {
        return Err(unsaid());
    };
    let scalar = super::lower_scalar_ident(scalar).map_err(|_| unsaid())?;
    if scalar != Scalar::F32 && scalar != Scalar::F16 {
        return Err(unsaid());
    }
    let role = role_of(role).ok_or_else(unsaid)?;
    Ok(ctx.intern_handle_type(TypeInner::CooperativeMatrix {
        columns: size,
        rows: size,
        scalar,
        role,
    }))
}

/// `A`, `B` or `C`, by its own name or by a path to it.
fn role_of(ty: &syn::Type) -> Option<CooperativeRole> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    match path.path.segments.last()?.ident.to_string().as_str() {
        "A" => Some(CooperativeRole::A),
        "B" => Some(CooperativeRole::B),
        "C" => Some(CooperativeRole::C),
        _ => None,
    }
}

/// `CoopMat8x8::<f32, A>::load(&data[offset..], stride)`, its row-major
/// form, and `default()`, the zero matrix. `None` when `ty` names no
/// cooperative matrix. The matrix is spelled out, or named by an alias,
/// since nothing else in the call says what it loads.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_assoc_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    ty_path: &[String],
    seg: &syn::PathSegment,
    item: &str,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    let name = super::last(ty_path);
    let ty = match size_of(&name) {
        Some(size) => matrix_type(ctx, &name, size, seg)?,
        None => match ctx.named_type(ty_path)? {
            Some(ty) if matrix(ctx, ty).is_some() => ty,
            _ => return Ok(None),
        },
    };
    let (size, scalar, role) = matrix(ctx, ty).expect("a cooperative matrix");
    let call = format!("{name}::{item}");
    let row_major = match (item, args) {
        ("default", []) => {
            let handle = function
                .expressions
                .append(Expression::ZeroValue(ty), ctx.span);
            return Ok(Some((handle, ty)));
        }
        ("load", [_, _]) => false,
        ("load_row_major", [_, _]) => true,
        ("default" | "load" | "load_row_major", _) => return Err(Error::WrongArgCount(call)),
        _ => return Err(Error::UnsupportedMethod(call)),
    };
    let pointer = data_pointer(ctx, function, body, &call, args[0], env, scalar, false)?;
    let stride = stride(ctx, function, body, args[1], env)?;
    let handle = emit(
        ctx,
        function,
        body,
        Expression::CooperativeLoad {
            columns: size,
            rows: size,
            role,
            data: CooperativeData {
                pointer,
                stride,
                row_major,
            },
        },
    )?;
    Ok(Some((handle, ty)))
}

/// `a.mul_add(b, c)`, and `m.store(&mut data[offset..], stride)` with its
/// row-major form, which produces nothing.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_method(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    (value, ty): Typed,
    name: &str,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    let (size, scalar, role) = matrix(ctx, ty).expect("the caller checked");
    match (name, args) {
        ("mul_add", [b, c]) => {
            let (b, b_ty) = lower_expr(ctx, function, body, b, env)?;
            let (c, c_ty) = lower_expr(ctx, function, body, c, env)?;
            let fits = |ty, want| matrix(ctx, ty) == Some((size, scalar, want));
            if role != CooperativeRole::A
                || !fits(b_ty, CooperativeRole::B)
                || !fits(c_ty, CooperativeRole::C)
            {
                return Err(Error::CoopRoles);
            }
            let handle = emit(
                ctx,
                function,
                body,
                Expression::CooperativeMultiplyAdd { a: value, b, c },
            )?;
            Ok(Some((handle, c_ty)))
        }
        ("store" | "store_row_major", [data, stride_arg]) => {
            let pointer = data_pointer(ctx, function, body, name, data, env, scalar, true)?;
            let stride = stride(ctx, function, body, stride_arg, env)?;
            body.push(
                Statement::CooperativeStore {
                    target: value,
                    data: CooperativeData {
                        pointer,
                        stride,
                        row_major: name == "store_row_major",
                    },
                },
                ctx.span,
            );
            Ok(None)
        }
        ("mul_add" | "store" | "store_row_major", _) => Err(Error::WrongArgCount(name.into())),
        _ => Err(Error::UnsupportedMethod(name.into())),
    }
}

/// `&data[offset..]`, a pointer to the element at `offset`, and `&data`, to
/// the first. `data` is an array of the matrix's own scalars; a store takes
/// `&mut`, of storage it can write. The end of a range is the CPU's to check,
/// as an index out of bounds is.
#[allow(clippy::too_many_arguments)]
fn data_pointer(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    name: &str,
    expr: &Expr,
    env: &mut Env,
    scalar: Scalar,
    store: bool,
) -> Result<Handle<Expression>, Error> {
    let not_data = || Error::CoopData(name.into());
    let Expr::Reference(reference) = strip_parens(expr) else {
        return Err(not_data());
    };
    if store && reference.mutability.is_none() {
        return Err(not_data());
    }
    let (array, start) = match strip_parens(&reference.expr) {
        Expr::Index(index) => match strip_parens(&index.index) {
            Expr::Range(range) => (&*index.expr, range.start.as_deref()),
            _ => return Err(not_data()),
        },
        whole => (whole, None),
    };
    let place = lower_place(ctx, function, body, array, env)?.ok_or_else(not_data)?;
    let Some((element, size)) = ctx.as_array(place.ty) else {
        return Err(not_data());
    };
    if ctx.as_scalar(element) != Some(scalar) {
        return Err(not_data());
    }
    if store && !place.writable {
        return Err(Error::AssignToReadonly(place.root));
    }
    let bound = match size {
        naga::ArraySize::Constant(len) => Some(len.get()),
        _ => None,
    };
    let index = match start {
        Some(start) => index_expr(ctx, function, body, start, bound, env)?,
        None => IndexKind::Constant(0),
    };
    let access = match index {
        IndexKind::Constant(index) => Expression::AccessIndex {
            base: place.pointer,
            index,
        },
        IndexKind::Dynamic(index) => Expression::Access {
            base: place.pointer,
            index,
        },
    };
    emit(ctx, function, body, access)
}

/// How many scalars apart two columns, or two rows, start: a `u32`.
fn stride(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    expr: &Expr,
    env: &mut Env,
) -> Result<Handle<Expression>, Error> {
    let (stride, ty) = lower_expr_hinted(ctx, function, body, expr, env, Some(Scalar::U32))?;
    match ctx.as_scalar(ty) {
        Some(Scalar::U32) => Ok(stride),
        _ => Err(Error::TypeMismatch),
    }
}
