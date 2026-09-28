use naga::{Block, Expression, Function, Handle, MathFunction, Span, Statement, Type};
use syn::Expr;

use super::emit::emit;
use super::env::Env;
use super::expr::{lower_expr, lower_expr_hinted};
use super::matrix::lower_mat_ctor;
use super::parse_mat_ident;
use super::parse_vec_ident;
use super::vector::lower_vec_ctor;
use super::{Context, Shape, Typed};
use crate::Error;

/// The pointer for a `ptr<_, base>` parameter, from `&mut x`, `&x`, or a name
/// that already stands for storage.
fn pointer_arg(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    arg: &Expr,
    env: &mut Env,
    base: Handle<naga::Type>,
) -> Result<Handle<Expression>, Error> {
    let (target, wants_write) = match arg {
        Expr::Reference(reference) => (&*reference.expr, reference.mutability.is_some()),
        other => (other, false),
    };
    let place = super::place::lower_place(ctx, function, body, target, env)?
        .ok_or(Error::InvalidAssignTarget)?;
    if wants_write && !place.writable {
        return Err(Error::AssignToReadonly(place.root));
    }
    if place.ty != base {
        return Err(Error::TypeMismatch);
    }
    Ok(place.pointer)
}

/// The type `T()` names, for a zero value: a vector, matrix, scalar, or struct.
pub(super) fn zero_value_type(
    ctx: &mut Context,
    path: &[String],
) -> Result<Option<Handle<naga::Type>>, Error> {
    let name = super::last(path);
    if let Some((size, shorthand)) = parse_vec_ident(&name) {
        return Ok(Some(
            ctx.intern_vector(size, shorthand.unwrap_or(naga::Scalar::F32)),
        ));
    }
    if let Some((columns, rows, shorthand)) = parse_mat_ident(&name) {
        return Ok(Some(ctx.intern_matrix(
            columns,
            rows,
            shorthand.unwrap_or(naga::Scalar::F32),
        )));
    }
    let scalar = match name.as_str() {
        "f32" => naga::Scalar::F32,
        "u32" | "usize" => naga::Scalar::U32,
        "i32" | "isize" => naga::Scalar::I32,
        "bool" => naga::Scalar::BOOL,
        _ => return ctx.named_type(path),
    };
    Ok(Some(ctx.intern_scalar(scalar)))
}

/// How a call's path reads.
pub(super) enum Callee {
    /// `f(..)`, `brdf::f(..)`, `synaga_shader::dot(..)`: a function, by the
    /// whole path, whose own name is the last segment.
    Function(Vec<String>),
    /// `vec3::splat(..)`, `lighting::Sun::default()`: something a type
    /// provides. `ty` is the path to the type.
    Associated { ty: Vec<String>, item: String },
}

/// Is `path` a function reached through modules, or something on a type?
/// A sibling source or a `crate`/`self`/`super` start says module; so does a
/// head that names no type, which is how `synaga_shader::dot` reads.
pub(super) fn classify_path(ctx: &mut Context, path: &[String]) -> Result<Callee, Error> {
    if let [ty @ .., item] = path {
        if !ty.is_empty() && !ctx.is_module_path(path) && names_type(ctx, ty)? {
            return Ok(Callee::Associated {
                ty: ty.to_vec(),
                item: item.clone(),
            });
        }
    }
    Ok(Callee::Function(path.to_vec()))
}

/// Does `path` name a type: a builtin one, or one the sources declare?
fn names_type(ctx: &mut Context, path: &[String]) -> Result<bool, Error> {
    let name = super::last(path);
    Ok(parse_vec_ident(&name).is_some()
        || parse_mat_ident(&name).is_some()
        || matches!(
            name.as_str(),
            "f32" | "u32" | "i32" | "usize" | "isize" | "bool" | "ray_query"
        )
        || ctx.named_type(path)?.is_some())
}

/// A call in statement position, where builtins that write rather than produce
/// a value are allowed.
pub(super) fn lower_call_stmt(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
) -> Result<(), Error> {
    lower_call_any(ctx, function, body, call, env).map(|_| ())
}

/// A call in value position: it has to produce something.
pub(super) fn lower_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
) -> Result<Typed, Error> {
    lower_call_any(ctx, function, body, call, env)?
        .ok_or_else(|| Error::ValueFromStatement(callee_label(call)))
}

/// What a call's callee is called, for an error.
fn callee_label(call: &syn::ExprCall) -> String {
    match call.func.as_ref() {
        Expr::Path(path) => super::last(&super::path_segments(&path.path)),
        _ => "call".into(),
    }
}

fn barrier(name: &str) -> Option<naga::Barrier> {
    match name {
        "workgroupBarrier" | "workgroup_barrier" => Some(naga::Barrier::WORK_GROUP),
        "storageBarrier" | "storage_barrier" => Some(naga::Barrier::STORAGE),
        "textureBarrier" | "texture_barrier" => Some(naga::Barrier::TEXTURE),
        _ => None,
    }
}

/// `bitcast::<u32>(1.0)`: same bits, a different scalar type.
fn bitcast_target(call: &syn::ExprCall) -> Option<&syn::Type> {
    let Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return None;
    }
    let seg = &path.path.segments[0];
    if seg.ident != "bitcast" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    });
    let ty = types.next()?;
    if types.next().is_some() {
        return None;
    }
    Some(ty)
}

fn lower_bitcast(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
    target_ty: &syn::Type,
) -> Result<Typed, Error> {
    if call.args.len() != 1 {
        return Err(Error::WrongArgCount("bitcast".into()));
    }
    let target = ctx.lower_type(target_ty)?;
    let (value, value_ty) = lower_expr(ctx, function, body, &call.args[0], env)?;
    let to = ctx
        .as_scalar(target)
        .ok_or_else(|| Error::UnsupportedCast("bitcast".into()))?;
    let from = ctx
        .as_scalar(value_ty)
        .ok_or_else(|| Error::UnsupportedCast("bitcast".into()))?;
    if from.width != to.width {
        return Err(Error::TypeMismatch);
    }
    let handle = emit(
        function,
        body,
        Expression::As {
            expr: value,
            kind: to.kind,
            convert: None,
        },
    )?;
    Ok((handle, target))
}

/// A call anywhere: what it produces, or `None` for a call to something that
/// produces nothing, which is fine in statement or tail position.
pub(super) fn lower_call_any(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    if let Some(ty) = bitcast_target(call) {
        return lower_bitcast(ctx, function, body, call, env, ty).map(Some);
    }
    let path = match call.func.as_ref() {
        Expr::Path(path) if path.qself.is_none() => super::path_segments(&path.path),
        _ => return Err(Error::UnsupportedExpr("call".into())),
    };
    let path = match classify_path(ctx, &path)? {
        // `vec3::splat(x)` and `vec4::from(v)` name the type they build.
        Callee::Associated { ty, item } => {
            let args: Vec<&Expr> = call.args.iter().collect();
            return super::method::lower_qualified_call(
                ctx, function, body, &ty, &item, &args, env,
            )
            .map(Some);
        }
        Callee::Function(path) => path,
    };
    // A function the sources declare wins over a builtin of the same name,
    // as a local item shadows a glob import in Rust.
    if let Some(callee) = ctx.function(&path)? {
        return lower_fn_call(ctx, function, body, call, env, callee);
    }
    let name = super::last(&path);
    // `T()` is WGSL's zero value, and the natural spelling for one here too.
    if call.args.is_empty() {
        if let Some(ty) = zero_value_type(ctx, &path)? {
            let handle = function
                .expressions
                .append(Expression::ZeroValue(ty), Span::UNDEFINED);
            return Ok(Some((handle, ty)));
        }
    }
    if parse_vec_ident(&name).is_some() {
        return lower_vec_ctor(ctx, function, body, call, env).map(Some);
    }
    if parse_mat_ident(&name).is_some() {
        return lower_mat_ctor(ctx, function, body, call, env).map(Some);
    }
    if name == "select" {
        return lower_select(ctx, function, body, call, env).map(Some);
    }
    if let Some(barrier) = barrier(&name) {
        if !call.args.is_empty() {
            return Err(Error::WrongArgCount(name));
        }
        body.push(Statement::ControlBarrier(barrier), Span::UNDEFINED);
        return Ok(None);
    }
    if name == "discard" {
        if !call.args.is_empty() {
            return Err(Error::WrongArgCount(name));
        }
        body.push(Statement::Kill, Span::UNDEFINED);
        return Ok(None);
    }
    if let Some(fun) = relational(&name) {
        return lower_relational(ctx, function, body, call, env, &name, fun).map(Some);
    }
    if let Some(spec) = math_spec(&name) {
        return lower_math(ctx, function, body, call, env, &name, spec).map(Some);
    }
    Err(Error::UnknownFunction(name))
}

/// `select(reject, accept, condition)`, in WGSL's argument order: the value
/// picked when the condition holds comes second.
fn lower_select(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
) -> Result<Typed, Error> {
    if call.args.len() != 3 {
        return Err(Error::WrongArgCount("select".into()));
    }
    let (reject, ty) = lower_expr_hinted(ctx, function, body, &call.args[0], env, None)?;
    let hint = ctx.shape(ty).int_hint();
    let (accept, accept_ty) = lower_expr_hinted(ctx, function, body, &call.args[1], env, hint)?;
    if accept_ty != ty {
        return Err(Error::TypeMismatch);
    }
    let (condition, cond_ty) = lower_expr(ctx, function, body, &call.args[2], env)?;
    // A scalar condition picks one whole value; a vector one picks per lane, so
    // it has to line up with the operands.
    let ok = match (ctx.shape(cond_ty), ctx.shape(ty)) {
        (Shape::Scalar(s), _) => s == naga::Scalar::BOOL,
        (Shape::Vector(size, s), Shape::Vector(operand, _)) => {
            s == naga::Scalar::BOOL && size == operand
        }
        _ => false,
    };
    if !ok {
        return Err(Error::TypeMismatch);
    }
    let handle = emit(
        function,
        body,
        Expression::Select {
            condition,
            accept,
            reject,
        },
    )?;
    Ok((handle, ty))
}

/// `all(v)` / `any(v)`: Naga keeps these apart from the math builtins.
fn relational(name: &str) -> Option<naga::RelationalFunction> {
    match name {
        "all" => Some(naga::RelationalFunction::All),
        "any" => Some(naga::RelationalFunction::Any),
        "is_nan" | "isNan" => Some(naga::RelationalFunction::IsNan),
        "is_inf" | "isInf" => Some(naga::RelationalFunction::IsInf),
        _ => None,
    }
}

fn lower_relational(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
    name: &str,
    fun: naga::RelationalFunction,
) -> Result<Typed, Error> {
    let [arg] = call.args.iter().collect::<Vec<_>>()[..] else {
        return Err(Error::WrongArgCount(name.into()));
    };
    let (argument, ty) = lower_expr_hinted(ctx, function, body, arg, env, None)?;
    use naga::RelationalFunction as Rf;
    // `all`/`any` fold a bool vector to one bool; `isNan`/`isInf` test floats
    // component-wise and keep the shape.
    let result = match (fun, ctx.shape(ty)) {
        (Rf::All | Rf::Any, shape) if shape.elem_kind() == Some(naga::ScalarKind::Bool) => {
            ctx.intern_scalar(naga::Scalar::BOOL)
        }
        (Rf::IsNan | Rf::IsInf, shape) if shape.elem_kind() == Some(naga::ScalarKind::Float) => {
            ctx.bool_like(ty)
        }
        _ => return Err(Error::TypeMismatch),
    };
    let handle = emit(function, body, Expression::Relational { fun, argument })?;
    Ok((handle, result))
}

struct MathSpec {
    fun: MathFunction,
    argc: usize,
    result: MathResult,
}

enum MathResult {
    SameAsFirst,
    ScalarOfFirst,
    Transpose,
    /// `u32`, for the pack functions.
    U32,
    /// `vec4<f32>`, for the 4x8 unpack functions.
    Vec4F32,
    /// `vec2<f32>`, for the 2x16 unpack functions.
    Vec2F32,
}

fn math_spec(name: &str) -> Option<MathSpec> {
    use MathFunction as Mf;
    use MathResult::*;
    let (fun, argc, result) = match name {
        "abs" => (Mf::Abs, 1, SameAsFirst),
        "sign" => (Mf::Sign, 1, SameAsFirst),
        "saturate" => (Mf::Saturate, 1, SameAsFirst),
        "sin" => (Mf::Sin, 1, SameAsFirst),
        "cos" => (Mf::Cos, 1, SameAsFirst),
        "tan" => (Mf::Tan, 1, SameAsFirst),
        "asin" => (Mf::Asin, 1, SameAsFirst),
        "acos" => (Mf::Acos, 1, SameAsFirst),
        "atan" => (Mf::Atan, 1, SameAsFirst),
        "floor" => (Mf::Floor, 1, SameAsFirst),
        "ceil" => (Mf::Ceil, 1, SameAsFirst),
        "round" => (Mf::Round, 1, SameAsFirst),
        "fract" => (Mf::Fract, 1, SameAsFirst),
        "sqrt" => (Mf::Sqrt, 1, SameAsFirst),
        "inverse_sqrt" | "inversesqrt" | "inverseSqrt" => (Mf::InverseSqrt, 1, SameAsFirst),
        "inverse" => (Mf::Inverse, 1, SameAsFirst),
        "trunc" => (Mf::Trunc, 1, SameAsFirst),
        "degrees" => (Mf::Degrees, 1, SameAsFirst),
        "radians" => (Mf::Radians, 1, SameAsFirst),
        "sinh" => (Mf::Sinh, 1, SameAsFirst),
        "cosh" => (Mf::Cosh, 1, SameAsFirst),
        "tanh" => (Mf::Tanh, 1, SameAsFirst),
        "asinh" => (Mf::Asinh, 1, SameAsFirst),
        "acosh" => (Mf::Acosh, 1, SameAsFirst),
        "atanh" => (Mf::Atanh, 1, SameAsFirst),
        "quantize_to_f16" | "quantizeToF16" => (Mf::QuantizeToF16, 1, SameAsFirst),
        "count_trailing_zeros" | "countTrailingZeros" => (Mf::CountTrailingZeros, 1, SameAsFirst),
        "count_leading_zeros" | "countLeadingZeros" => (Mf::CountLeadingZeros, 1, SameAsFirst),
        "count_one_bits" | "countOneBits" => (Mf::CountOneBits, 1, SameAsFirst),
        "reverse_bits" | "reverseBits" => (Mf::ReverseBits, 1, SameAsFirst),
        "first_trailing_bit" | "firstTrailingBit" => (Mf::FirstTrailingBit, 1, SameAsFirst),
        "first_leading_bit" | "firstLeadingBit" => (Mf::FirstLeadingBit, 1, SameAsFirst),
        "extract_bits" | "extractBits" => (Mf::ExtractBits, 3, SameAsFirst),
        "insert_bits" | "insertBits" => (Mf::InsertBits, 4, SameAsFirst),
        "step" => (Mf::Step, 2, SameAsFirst),
        "refract" => (Mf::Refract, 3, SameAsFirst),
        "ldexp" => (Mf::Ldexp, 2, SameAsFirst),
        "pack4x8snorm" | "pack_4x8_snorm" => (Mf::Pack4x8snorm, 1, U32),
        "pack4x8unorm" | "pack_4x8_unorm" => (Mf::Pack4x8unorm, 1, U32),
        "pack2x16snorm" | "pack_2x16_snorm" => (Mf::Pack2x16snorm, 1, U32),
        "pack2x16unorm" | "pack_2x16_unorm" => (Mf::Pack2x16unorm, 1, U32),
        "pack2x16float" | "pack_2x16_float" => (Mf::Pack2x16float, 1, U32),
        "unpack4x8snorm" | "unpack_4x8_snorm" => (Mf::Unpack4x8snorm, 1, Vec4F32),
        "unpack4x8unorm" | "unpack_4x8_unorm" => (Mf::Unpack4x8unorm, 1, Vec4F32),
        "unpack2x16snorm" | "unpack_2x16_snorm" => (Mf::Unpack2x16snorm, 1, Vec2F32),
        "unpack2x16unorm" | "unpack_2x16_unorm" => (Mf::Unpack2x16unorm, 1, Vec2F32),
        "unpack2x16float" | "unpack_2x16_float" => (Mf::Unpack2x16float, 1, Vec2F32),
        "normalize" => (Mf::Normalize, 1, SameAsFirst),
        "exp" => (Mf::Exp, 1, SameAsFirst),
        "exp2" => (Mf::Exp2, 1, SameAsFirst),
        "log" => (Mf::Log, 1, SameAsFirst),
        "log2" => (Mf::Log2, 1, SameAsFirst),
        "min" => (Mf::Min, 2, SameAsFirst),
        "max" => (Mf::Max, 2, SameAsFirst),
        "pow" => (Mf::Pow, 2, SameAsFirst),
        "atan2" => (Mf::Atan2, 2, SameAsFirst),
        "reflect" => (Mf::Reflect, 2, SameAsFirst),
        "cross" => (Mf::Cross, 2, SameAsFirst),
        "clamp" => (Mf::Clamp, 3, SameAsFirst),
        "mix" => (Mf::Mix, 3, SameAsFirst),
        "smoothstep" | "smooth_step" => (Mf::SmoothStep, 3, SameAsFirst),
        "fma" => (Mf::Fma, 3, SameAsFirst),
        "face_forward" | "faceforward" => (Mf::FaceForward, 3, SameAsFirst),
        "dot" => (Mf::Dot, 2, ScalarOfFirst),
        "distance" => (Mf::Distance, 2, ScalarOfFirst),
        "length" => (Mf::Length, 1, ScalarOfFirst),
        "transpose" => (Mf::Transpose, 1, Transpose),
        "determinant" => (Mf::Determinant, 1, ScalarOfFirst),
        _ => return None,
    };
    Some(MathSpec { fun, argc, result })
}

/// `y.sin()`, `y.atan2(x)`, `v.dot(w)`: a math builtin with the receiver as
/// its first argument. `None` when `name` is not one, or the arity does not
/// match, so some other method can still claim it.
pub(super) fn lower_math_method(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    receiver: Typed,
    name: &str,
    args: &[&syn::Expr],
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    let Some(spec) = math_spec(name) else {
        return Ok(None);
    };
    if args.len() + 1 != spec.argc {
        return Ok(None);
    }
    let mut hint = ctx.shape(receiver.1).int_hint();
    let mut handles = vec![receiver.0];
    let mut tys = vec![receiver.1];
    for arg in args {
        let (handle, ty) = super::expr::lower_expr_hinted(ctx, function, body, arg, env, hint)?;
        hint = hint.or_else(|| ctx.shape(ty).int_hint());
        handles.push(handle);
        tys.push(ty);
    }
    finish_math(ctx, function, body, spec, &handles, &tys).map(Some)
}

fn lower_math(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
    name: &str,
    spec: MathSpec,
) -> Result<Typed, Error> {
    if call.args.len() != spec.argc {
        return Err(Error::WrongArgCount(name.into()));
    }
    // `clamp(n, 0, 1)`: the first argument fixes the type, the rest follow it.
    let mut hint = None;
    let mut args = Vec::new();
    let mut tys = Vec::new();
    for arg in &call.args {
        let (h, ty) = lower_expr_hinted(ctx, function, body, arg, env, hint)?;
        hint = hint.or_else(|| ctx.shape(ty).int_hint());
        args.push(h);
        tys.push(ty);
    }
    finish_math(ctx, function, body, spec, &args, &tys)
}

fn finish_math(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    spec: MathSpec,
    args: &[Handle<Expression>],
    tys: &[Handle<Type>],
) -> Result<Typed, Error> {
    let result_ty = match spec.result {
        MathResult::SameAsFirst => tys[0],
        MathResult::ScalarOfFirst => {
            if let Some(s) = ctx.as_scalar(tys[0]) {
                ctx.intern_scalar(s)
            } else if let Some((_, s)) = ctx.as_vector(tys[0]) {
                ctx.intern_scalar(s)
            } else if let Some((columns, rows, s)) = ctx.as_matrix(tys[0]) {
                if columns != rows {
                    return Err(Error::TypeMismatch);
                }
                ctx.intern_scalar(s)
            } else {
                return Err(Error::TypeMismatch);
            }
        }
        MathResult::Transpose => {
            let (columns, rows, s) = ctx.as_matrix(tys[0]).ok_or(Error::TypeMismatch)?;
            ctx.intern_matrix(rows, columns, s)
        }
        MathResult::U32 => ctx.intern_scalar(naga::Scalar::U32),
        MathResult::Vec4F32 => ctx.intern_vector(naga::VectorSize::Quad, naga::Scalar::F32),
        MathResult::Vec2F32 => ctx.intern_vector(naga::VectorSize::Bi, naga::Scalar::F32),
    };
    let handle = emit(
        function,
        body,
        Expression::Math {
            fun: spec.fun,
            arg: args[0],
            arg1: args.get(1).copied(),
            arg2: args.get(2).copied(),
            arg3: args.get(3).copied(),
        },
    )?;
    Ok((handle, result_ty))
}

fn lower_fn_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
    callee: Handle<Function>,
) -> Result<Option<Typed>, Error> {
    let (name, expected, ret_ty): (String, Vec<_>, Option<_>) = {
        let func = &ctx.module.functions[callee];
        let expected = func.arguments.iter().map(|a| a.ty).collect();
        (
            func.name.clone().unwrap_or_default(),
            expected,
            func.result.as_ref().map(|r| r.ty),
        )
    };
    let name = name.as_str();

    if expected.len() != call.args.len() {
        return Err(Error::WrongArgCount(name.into()));
    }
    let mut arg_values = Vec::new();
    for (arg, &want) in call.args.iter().zip(expected.iter()) {
        // A pointer parameter takes the storage itself. `&mut x` borrows it;
        // a name that is already a pointer parameter passes straight through,
        // the way a Rust reborrow does.
        if let Some(base) = ctx.pointee(want) {
            arg_values.push(pointer_arg(ctx, function, body, arg, env, base)?);
            continue;
        }
        let hint = ctx.shape(want).int_hint();
        let (h, ty) = lower_expr_hinted(ctx, function, body, arg, env, hint)?;
        if ty != want {
            return Err(Error::TypeMismatch);
        }
        arg_values.push(h);
    }

    // A function with no return type produces nothing to bind.
    let result = ret_ty.map(|_| {
        function
            .expressions
            .append(Expression::CallResult(callee), Span::UNDEFINED)
    });
    body.push(
        Statement::Call {
            function: callee,
            arguments: arg_values,
            result,
        },
        Span::UNDEFINED,
    );
    Ok(result.zip(ret_ty))
}
