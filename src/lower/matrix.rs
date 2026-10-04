use naga::{Block, Expression, Function, Handle, Type};
use syn::Expr;

use super::emit::emit;
use super::env::Env;
use super::expr::lower_expr_hinted;
use super::parse_mat_ident;
use super::{Context, Shape, Typed};
use crate::Error;

pub(super) fn lower_mat_ctor(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
) -> Result<Typed, Error> {
    let name = match call.func.as_ref() {
        Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            path.path.segments[0].ident.to_string()
        }
        _ => return Err(Error::UnsupportedExpr("call".into())),
    };
    let (columns, rows, shorthand) =
        parse_mat_ident(&name).ok_or_else(|| Error::BadMatCtor(name.clone()))?;

    if call.args.is_empty() {
        return Err(Error::MatCtorArgs);
    }
    let mut components = Vec::new();
    let mut component_tys = Vec::new();
    for arg in &call.args {
        let (handle, ty) = lower_expr_hinted(ctx, function, body, arg, env, None)?;
        components.push(handle);
        component_tys.push(ty);
    }

    let scalar = match (shorthand, ctx.shape(component_tys[0])) {
        (Some(s), _) => s,
        (None, Shape::Scalar(s) | Shape::Vector(_, s) | Shape::Matrix(_, _, s)) => s,
        (None, Shape::Other) => return Err(Error::TypeMismatch),
    };

    let ty = ctx.intern_matrix(columns, rows, scalar);

    // Identity-ish copy: matN(m) where m already has that type.
    if components.len() == 1 && component_tys[0] == ty {
        return Ok((components[0], ty));
    }

    // Column vectors: matCxR(c0, c1, ...)
    if components.len() == columns as usize
        && component_tys
            .iter()
            .all(|&t| ctx.as_vector(t) == Some((rows, scalar)))
    {
        let handle = emit(ctx, function, body, Expression::Compose { ty, components })?;
        return Ok((handle, ty));
    }

    // Flattened column-major scalars: mat2(a, b, c, d)
    let flat = (columns as usize) * (rows as usize);
    if components.len() == flat
        && component_tys
            .iter()
            .all(|&t| ctx.as_scalar(t) == Some(scalar))
    {
        let mut columns_expr = Vec::new();
        let col_ty = ctx.intern_vector(rows, scalar);
        let col_width = rows as usize;
        for col in 0..columns as usize {
            let start = col * col_width;
            let slice = components[start..start + col_width].to_vec();
            let col_handle = emit(
                ctx,
                function,
                body,
                Expression::Compose {
                    ty: col_ty,
                    components: slice,
                },
            )?;
            columns_expr.push(col_handle);
        }
        let handle = emit(
            ctx,
            function,
            body,
            Expression::Compose {
                ty,
                components: columns_expr,
            },
        )?;
        return Ok((handle, ty));
    }

    Err(Error::MatCtorArgs)
}

/// Result type of `left * right`.
///
/// Naga gives `Multiply` the widest operand table of any operator: scalars,
/// component-wise vectors, vector/scalar and matrix/scalar scaling, the three
/// linear-algebra products, and nothing else.
pub(super) fn multiply_result_ty(
    ctx: &mut Context,
    left: Handle<Type>,
    right: Handle<Type>,
) -> Result<Handle<Type>, Error> {
    use naga::ScalarKind as Sk;

    let numeric = |s: naga::Scalar| matches!(s.kind, Sk::Uint | Sk::Sint | Sk::Float);
    let float = |s: naga::Scalar| s.kind == Sk::Float;
    let bad = || Error::BadOperandTypes("*".into());

    // A cooperative matrix scales by its own scalar, from either side.
    let cooperative = |ty| super::cooperative::matrix(ctx, ty).map(|(_, scalar, _)| scalar);
    match (cooperative(left), cooperative(right)) {
        (Some(scalar), None) if ctx.shape(right) == Shape::Scalar(scalar) => return Ok(left),
        (None, Some(scalar)) if ctx.shape(left) == Shape::Scalar(scalar) => return Ok(right),
        (None, None) => {}
        _ => return Err(bad()),
    }

    match (ctx.shape(left), ctx.shape(right)) {
        (Shape::Scalar(a), Shape::Scalar(b)) if a == b && numeric(a) => Ok(left),
        (Shape::Vector(n, a), Shape::Vector(m, b)) if n == m && a == b && numeric(a) => Ok(left),
        (Shape::Vector(_, a), Shape::Scalar(b)) if a == b && numeric(a) => Ok(left),
        (Shape::Scalar(a), Shape::Vector(_, b)) if a == b && numeric(a) => Ok(right),
        // matCxR * vecC -> vecR
        (Shape::Matrix(columns, rows, a), Shape::Vector(n, b))
            if columns == n && a == b && float(a) =>
        {
            Ok(ctx.intern_vector(rows, a))
        }
        // vecR * matCxR -> vecC
        (Shape::Vector(n, a), Shape::Matrix(columns, rows, b))
            if n == rows && a == b && float(a) =>
        {
            Ok(ctx.intern_vector(columns, a))
        }
        // matKxR * matCxK -> matCxR
        (Shape::Matrix(k, rows, a), Shape::Matrix(columns, k2, b))
            if k == k2 && a == b && float(a) =>
        {
            Ok(ctx.intern_matrix(columns, rows, a))
        }
        (Shape::Matrix(_, _, a), Shape::Scalar(b)) if a == b && float(a) => Ok(left),
        (Shape::Scalar(a), Shape::Matrix(_, _, b)) if a == b && float(a) => Ok(right),
        _ => Err(bad()),
    }
}
