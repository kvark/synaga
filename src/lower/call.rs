use naga::{Block, Expression, Function, Handle, MathFunction, Statement, Type};
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
pub(super) fn pointer_arg(
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
/// A vector's scalar comes from its path, `Vec3::<u32>`, or failing that from
/// `hint`, where the value goes.
pub(super) fn zero_value_type(
    ctx: &mut Context,
    path: &[String],
    turbofish: Option<naga::Scalar>,
    hint: Option<naga::Scalar>,
) -> Result<Option<Handle<naga::Type>>, Error> {
    let name = super::last(path);
    if let Some((size, shorthand)) = parse_vec_ident(&name) {
        let scalar = super::vec_scalar(&name, shorthand, turbofish)?.or(hint);
        return Ok(Some(
            ctx.intern_vector(size, scalar.unwrap_or(naga::Scalar::F32)),
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
        "f16" => naga::Scalar::F16,
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
            "f32" | "f16" | "u32" | "i32" | "usize" | "isize" | "bool" | "ray_query" | "RayQuery"
        )
        || super::ray::is_ray_word_type(&name)
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
    lower_call_any(ctx, function, body, call, env, None).map(|_| ())
}

/// A call in value position: it has to produce something. `hint` is the
/// integer scalar where the value goes, if that says one.
pub(super) fn lower_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
    hint: Option<naga::Scalar>,
) -> Result<Typed, Error> {
    lower_call_any(ctx, function, body, call, env, hint)?
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
        "subgroupBarrier" | "subgroup_barrier" => Some(naga::Barrier::SUB_GROUP),
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
        ctx,
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
///
/// `hint` is where the value goes, as Rust's inference would see it: it is
/// what makes `let c: Vec3<u32> = vec3(1, 2, 3)` a `u32` vector, when nothing
/// in the call itself says so.
pub(super) fn lower_call_any(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
    hint: Option<naga::Scalar>,
) -> Result<Option<Typed>, Error> {
    if let Some(ty) = bitcast_target(call) {
        return lower_bitcast(ctx, function, body, call, env, ty).map(Some);
    }
    let syn_path = match call.func.as_ref() {
        Expr::Path(path) if path.qself.is_none() => &path.path,
        _ => return Err(Error::UnsupportedExpr("call".into())),
    };
    let path = super::path_segments(syn_path);
    let path = match classify_path(ctx, &path)? {
        // `Vec3::splat(x)` and `Vec4::<i32>::from(v)` name the type they
        // build, a vector's scalar included.
        Callee::Associated { ty, item } => {
            let ty_segment = &syn_path.segments[syn_path.segments.len() - 2];
            let on = super::method::OnType {
                path: &ty,
                turbofish: super::turbofish_scalar(ty_segment)?,
                hint,
            };
            let args: Vec<&Expr> = call.args.iter().collect();
            if let Some(result) =
                super::method::lower_user_assoc_call(ctx, function, body, &on, &item, &args, env)?
            {
                return Ok(result);
            }
            return super::method::lower_qualified_call(ctx, function, body, on, &item, &args, env)
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
        let last = syn_path.segments.last().expect("a path has a segment");
        let turbofish = super::turbofish_scalar(last)?;
        if let Some(ty) = zero_value_type(ctx, &path, turbofish, hint)? {
            let handle = function
                .expressions
                .append(Expression::ZeroValue(ty), ctx.span);
            return Ok(Some((handle, ty)));
        }
    }
    if parse_vec_ident(&name).is_some() {
        return lower_vec_ctor(ctx, function, body, call, env, hint).map(Some);
    }
    if parse_mat_ident(&name).is_some() {
        return lower_mat_ctor(ctx, function, body, call, env).map(Some);
    }
    if name == "select" {
        return lower_select(ctx, function, body, call, env, hint).map(Some);
    }
    if let Some(barrier) = barrier(&name) {
        if !call.args.is_empty() {
            return Err(Error::WrongArgCount(name));
        }
        body.push(Statement::ControlBarrier(barrier), ctx.span);
        return Ok(None);
    }
    if name == "discard" {
        if !call.args.is_empty() {
            return Err(Error::WrongArgCount(name));
        }
        body.push(Statement::Kill, ctx.span);
        return Ok(None);
    }
    if let Some(fun) = relational(&name) {
        return lower_relational(ctx, function, body, call, env, &name, fun).map(Some);
    }
    if let Some(result) =
        super::subgroup::lower_subgroup_call(ctx, function, body, &name, call, env, hint)?
    {
        return Ok(Some(result));
    }
    if let Some(spec) = math_spec(&name) {
        return lower_math(ctx, function, body, call, env, (&name, spec), hint).map(Some);
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
    hint: Option<naga::Scalar>,
) -> Result<Typed, Error> {
    if call.args.len() != 3 {
        return Err(Error::WrongArgCount("select".into()));
    }
    let (reject, ty) = lower_expr_hinted(ctx, function, body, &call.args[0], env, hint)?;
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
        ctx,
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
    use naga::RelationalFunction as Rf;
    let lanewise = match fun {
        Rf::All | Rf::Any => super::expr::lower_lanewise(ctx, function, body, arg, env)?,
        _ => None,
    };
    let (argument, ty) = match lanewise {
        Some(typed) => typed,
        None => lower_expr_hinted(ctx, function, body, arg, env, None)?,
    };
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
    let handle = emit(
        ctx,
        function,
        body,
        Expression::Relational { fun, argument },
    )?;
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
    /// `u32`, for the pack functions and `dot4_u8_packed`.
    U32,
    /// `i32`, for `dot4_i8_packed`.
    I32,
    /// `vec4<f32>`, for the 4x8 unpack functions.
    Vec4F32,
    /// `vec4<i32>`, for `unpack4x_i8`.
    Vec4I32,
    /// `vec4<u32>`, for `unpack4x_u8`.
    Vec4U32,
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
        "dot4_u8_packed" | "dot4U8Packed" => (Mf::Dot4U8Packed, 2, U32),
        "dot4_i8_packed" | "dot4I8Packed" => (Mf::Dot4I8Packed, 2, I32),
        "pack4x_i8" | "pack4xI8" => (Mf::Pack4xI8, 1, U32),
        "pack4x_u8" | "pack4xU8" => (Mf::Pack4xU8, 1, U32),
        "pack4x_i8_clamp" | "pack4xI8Clamp" => (Mf::Pack4xI8Clamp, 1, U32),
        "pack4x_u8_clamp" | "pack4xU8Clamp" => (Mf::Pack4xU8Clamp, 1, U32),
        "unpack4x_i8" | "unpack4xI8" => (Mf::Unpack4xI8, 1, Vec4I32),
        "unpack4x_u8" | "unpack4xU8" => (Mf::Unpack4xU8, 1, Vec4U32),
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

/// What a math method means, where Rust names it differently from WGSL or
/// means something else by the same name.
enum RustMath {
    /// The builtin, with the receiver as its first argument.
    Builtin(MathSpec),
    /// `x.fract()`, which Rust takes toward zero: `x - x.trunc()`. The GPU's
    /// `fract` is `x - floor(x)`, which differs for a negative `x`.
    Fract,
    /// `v.length_squared()`: `dot(v, v)`.
    LengthSquared,
    /// `x.recip()`: `1 / x`.
    Recip,
    /// `n.unsigned_abs()`: an `i32`'s magnitude as a `u32`, which is right
    /// even for `i32::MIN`.
    UnsignedAbs,
    /// `x.rotate_left(n)` is `(x << n) | (x >> (32 - n))`, and the other way
    /// round for `rotate_right`. WGSL takes a shift's amount modulo 32, which
    /// makes that right for every `n`.
    Rotate { left: bool },
    /// `a.wrapping_mul(b)` and its siblings: the GPU's operator, which wraps
    /// on overflow. Rust's plain operator panics there under overflow checks,
    /// so code that means to wrap, as a hash does, says so.
    Wrapping(naga::BinaryOperator),
    /// `n.wrapping_neg()`: `-n`, or `0 - n` for a `u32`.
    WrappingNeg,
    /// `x.wrapping_shl(n)`: the amount taken modulo 32, as Rust takes it.
    WrappingShift { left: bool },
}

fn rust_math(name: &str) -> Option<RustMath> {
    use MathFunction as Mf;
    use MathResult::*;
    let (fun, argc, result) = match name {
        "ln" => (Mf::Log, 1, SameAsFirst),
        "powf" => (Mf::Pow, 2, SameAsFirst),
        "mul_add" => (Mf::Fma, 3, SameAsFirst),
        "to_degrees" => (Mf::Degrees, 1, SameAsFirst),
        "to_radians" => (Mf::Radians, 1, SameAsFirst),
        // The GPU's `round` takes a half to the even neighbour, which is what
        // Rust calls this one.
        "round_ties_even" => (Mf::Round, 1, SameAsFirst),
        "lerp" => (Mf::Mix, 3, SameAsFirst),
        "signum" => (Mf::Sign, 1, SameAsFirst),
        "fract" => return Some(RustMath::Fract),
        "length_squared" => return Some(RustMath::LengthSquared),
        "recip" => return Some(RustMath::Recip),
        "unsigned_abs" => return Some(RustMath::UnsignedAbs),
        "rotate_left" => return Some(RustMath::Rotate { left: true }),
        "rotate_right" => return Some(RustMath::Rotate { left: false }),
        "wrapping_add" => return Some(RustMath::Wrapping(naga::BinaryOperator::Add)),
        "wrapping_sub" => return Some(RustMath::Wrapping(naga::BinaryOperator::Subtract)),
        "wrapping_mul" => return Some(RustMath::Wrapping(naga::BinaryOperator::Multiply)),
        "wrapping_neg" => return Some(RustMath::WrappingNeg),
        "wrapping_shl" => return Some(RustMath::WrappingShift { left: true }),
        "wrapping_shr" => return Some(RustMath::WrappingShift { left: false }),
        _ => return None,
    };
    Some(RustMath::Builtin(MathSpec { fun, argc, result }))
}

/// Is `name` a math method, as `x.sqrt()` and `v.dot(w)` are?
pub(super) fn is_math_method(name: &str) -> bool {
    rust_math(name).is_some() || math_spec(name).is_some()
}

/// Why a method Rust has cannot be the GPU builtin of the same name, for the
/// two where the difference is more than a line to make up.
fn differs_on_gpu(name: &str, receiver: Shape) -> Option<&'static str> {
    let float = receiver.elem_kind() == Some(naga::ScalarKind::Float);
    match name {
        "round" if float => Some(
            "Rust rounds a half away from zero and the GPU to the even neighbour; \
             `round_ties_even()` is the GPU's",
        ),
        "signum" if float => Some(
            "Rust's `0.0.signum()` is 1 and the GPU's `sign(0.0)` is 0; `sign(x)` is the GPU's",
        ),
        _ => None,
    }
}

/// `x.sqrt()`, `y.atan2(x)`, `v.dot(w)`, `a.lerp(b, t)`: a math builtin with
/// the receiver as its first argument. `None` when `name` is not one, or the
/// arity does not match, so some other method can still claim it.
///
/// The names are Rust's: `f32`'s own methods, and glam's for what only a
/// vector has. Where Rust means something else by a name WGSL also has, the
/// method means what Rust says, since that is what `rustc` checked. WGSL's
/// other names work as methods too, `x.saturate()`, for a shader `rustc`
/// never sees.
pub(super) fn lower_math_method(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    receiver: Typed,
    name: &str,
    args: &[&syn::Expr],
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    let shape = ctx.shape(receiver.1);
    if let Some(why) = differs_on_gpu(name, shape) {
        return Err(Error::DiffersOnGpu(name.into(), why));
    }
    let spec = match rust_math(name) {
        Some(RustMath::Builtin(spec)) => spec,
        Some(special) => {
            return lower_special_math(ctx, function, body, receiver, special, args, env);
        }
        None => match math_spec(name) {
            Some(spec) => spec,
            None => return Ok(None),
        },
    };
    if args.len() + 1 != spec.argc {
        return Ok(None);
    }
    let mut hint = shape.int_hint();
    let mut handles = vec![receiver.0];
    let mut tys = vec![receiver.1];
    for arg in args {
        let (mut handle, mut ty) =
            super::expr::lower_expr_hinted(ctx, function, body, arg, env, hint)?;
        // `v.powf(2.0)` raises every lane to one power, which the GPU's `pow`
        // wants as a vector.
        if let (MathFunction::Pow, Shape::Vector(size, _), Shape::Scalar(_)) =
            (spec.fun, shape, ctx.shape(ty))
        {
            handle = emit(
                ctx,
                function,
                body,
                Expression::Splat {
                    size,
                    value: handle,
                },
            )?;
            ty = receiver.1;
        }
        hint = hint.or_else(|| ctx.shape(ty).int_hint());
        handles.push(handle);
        tys.push(ty);
    }
    finish_math(ctx, function, body, spec, &handles, &tys).map(Some)
}

/// The methods that are a few operations rather than one builtin. `None` for
/// the wrong number of arguments, as for a builtin.
fn lower_special_math(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    (value, ty): Typed,
    special: RustMath,
    args: &[&syn::Expr],
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    use naga::ScalarKind as Kind;
    let shape = ctx.shape(ty);
    let kind = shape.elem_kind();
    let fits = match (&special, args) {
        (RustMath::Fract | RustMath::LengthSquared | RustMath::Recip, []) => {
            kind == Some(Kind::Float)
        }
        (RustMath::UnsignedAbs, []) => kind == Some(Kind::Sint),
        (RustMath::Rotate { .. } | RustMath::WrappingShift { .. }, [_]) => {
            matches!(shape, Shape::Scalar(s) if s == naga::Scalar::U32 || s == naga::Scalar::I32)
        }
        (RustMath::Wrapping(_), [_]) | (RustMath::WrappingNeg, []) => {
            matches!(kind, Some(Kind::Sint | Kind::Uint))
        }
        _ => return Ok(None),
    };
    if !fits {
        return Err(Error::TypeMismatch);
    }
    let math = |fun, arg1| Expression::Math {
        fun,
        arg: value,
        arg1,
        arg2: None,
        arg3: None,
    };
    let typed = match special {
        RustMath::UnsignedAbs => {
            let magnitude = emit(ctx, function, body, math(MathFunction::Abs, None))?;
            let expr = Expression::As {
                expr: magnitude,
                kind: naga::ScalarKind::Uint,
                convert: Some(4),
            };
            let uint = match shape {
                Shape::Vector(size, _) => ctx.intern_vector(size, naga::Scalar::U32),
                _ => ctx.intern_scalar(naga::Scalar::U32),
            };
            (emit(ctx, function, body, expr)?, uint)
        }
        RustMath::Rotate { left } => {
            use naga::BinaryOperator as Op;
            let u32_ = naga::Scalar::U32;
            let (amount, amount_ty) =
                lower_expr_hinted(ctx, function, body, args[0], env, Some(u32_))?;
            if ctx.shape(amount_ty) != Shape::Scalar(u32_) {
                return Err(Error::BadShiftType);
            }
            let reinterpret = |function: &mut Function, body: &mut Block, expr, kind| {
                let convert = None;
                emit(
                    ctx,
                    function,
                    body,
                    Expression::As {
                        expr,
                        kind,
                        convert,
                    },
                )
            };
            let literal = |function: &mut Function, n| {
                let literal = Expression::Literal(naga::Literal::U32(n));
                function.expressions.append(literal, ctx.span)
            };
            // An `i32` rotates its bits, which `>>` on it would not keep: it
            // drags the sign in.
            let signed = kind == Some(Kind::Sint);
            let bits = match signed {
                true => reinterpret(function, body, value, Kind::Uint)?,
                false => value,
            };
            // Rust takes the amount modulo 32. WGSL's shifts do too, but Naga
            // writes some backends' shifts as ones that are undefined from 32
            // up, so each amount is kept below it here.
            let (first, second) = if left {
                (Op::ShiftLeft, Op::ShiftRight)
            } else {
                (Op::ShiftRight, Op::ShiftLeft)
            };
            let mask = literal(function, 31);
            let near_amount = super::emit::binary(ctx, function, body, Op::And, amount, mask)?;
            let near = super::emit::binary(ctx, function, body, first, bits, near_amount)?;
            let width = literal(function, 32);
            let rest = super::emit::binary(ctx, function, body, Op::Subtract, width, amount)?;
            let mask = literal(function, 31);
            let far_amount = super::emit::binary(ctx, function, body, Op::And, rest, mask)?;
            let far = super::emit::binary(ctx, function, body, second, bits, far_amount)?;
            let rotated = super::emit::binary(ctx, function, body, Op::InclusiveOr, near, far)?;
            match signed {
                true => (reinterpret(function, body, rotated, Kind::Sint)?, ty),
                false => (rotated, ty),
            }
        }
        RustMath::Wrapping(op) => {
            let hint = shape.int_hint();
            let (rhs, rhs_ty) = lower_expr_hinted(ctx, function, body, args[0], env, hint)?;
            if ctx.shape(rhs_ty) != shape {
                return Err(Error::TypeMismatch);
            }
            (
                super::emit::binary(ctx, function, body, op, value, rhs)?,
                ty,
            )
        }
        RustMath::WrappingNeg => {
            let negated = match kind {
                Some(Kind::Sint) => emit(
                    ctx,
                    function,
                    body,
                    Expression::Unary {
                        op: naga::UnaryOperator::Negate,
                        expr: value,
                    },
                )?,
                // A `u32` has no `-`: `0 - n` wraps the same way.
                _ => {
                    let zero = function
                        .expressions
                        .append(Expression::ZeroValue(ty), ctx.span);
                    super::emit::binary(
                        ctx,
                        function,
                        body,
                        naga::BinaryOperator::Subtract,
                        zero,
                        value,
                    )?
                }
            };
            (negated, ty)
        }
        RustMath::WrappingShift { left } => {
            let u32_ = naga::Scalar::U32;
            let (amount, amount_ty) =
                lower_expr_hinted(ctx, function, body, args[0], env, Some(u32_))?;
            if ctx.shape(amount_ty) != Shape::Scalar(u32_) {
                return Err(Error::BadShiftType);
            }
            // WGSL takes the amount modulo 32 too, but as for a rotation,
            // Naga writes some backends' shifts as ones that are undefined
            // from 32 up.
            let mask = function
                .expressions
                .append(Expression::Literal(naga::Literal::U32(31)), ctx.span);
            let amount =
                super::emit::binary(ctx, function, body, naga::BinaryOperator::And, amount, mask)?;
            let op = match left {
                true => naga::BinaryOperator::ShiftLeft,
                false => naga::BinaryOperator::ShiftRight,
            };
            (
                super::emit::binary(ctx, function, body, op, value, amount)?,
                ty,
            )
        }
        RustMath::Fract => {
            let whole = emit(ctx, function, body, math(MathFunction::Trunc, None))?;
            let op = naga::BinaryOperator::Subtract;
            (
                super::emit::binary(ctx, function, body, op, value, whole)?,
                ty,
            )
        }
        RustMath::LengthSquared => {
            let Shape::Vector(_, scalar) = shape else {
                return Err(Error::TypeMismatch);
            };
            let handle = emit(ctx, function, body, math(MathFunction::Dot, Some(value)))?;
            (handle, ctx.intern_scalar(scalar))
        }
        RustMath::Recip => {
            let scalar = shape.scalar().ok_or(Error::TypeMismatch)?;
            let one = naga::Literal::one(scalar).ok_or(Error::TypeMismatch)?;
            let mut one = function
                .expressions
                .append(Expression::Literal(one), ctx.span);
            if let Shape::Vector(size, _) = shape {
                one = emit(ctx, function, body, Expression::Splat { size, value: one })?;
            }
            let op = naga::BinaryOperator::Divide;
            (
                super::emit::binary(ctx, function, body, op, one, value)?,
                ty,
            )
        }
        RustMath::Builtin(_) => unreachable!("a builtin is not special"),
    };
    Ok(Some(typed))
}

fn lower_math(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
    (name, spec): (&str, MathSpec),
    hint: Option<naga::Scalar>,
) -> Result<Typed, Error> {
    if call.args.len() != spec.argc {
        return Err(Error::WrongArgCount(name.into()));
    }
    // `clamp(n, 0, 1)`: the first argument fixes the type, the rest follow it.
    // Where the result is the first argument's type, or its scalar, where the
    // result goes says the type as well: `let n: u32 = max(1, 2)`.
    let mut hint = match spec.result {
        MathResult::SameAsFirst | MathResult::ScalarOfFirst => hint,
        _ => argument_scalar(spec.fun),
    };
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

/// The scalar a builtin's arguments are, where its signature fixes one rather
/// than following them: an unsuffixed literal takes it, as Rust's would.
fn argument_scalar(fun: MathFunction) -> Option<naga::Scalar> {
    use MathFunction as Mf;
    match fun {
        Mf::Dot4I8Packed
        | Mf::Dot4U8Packed
        | Mf::Pack4xU8
        | Mf::Pack4xU8Clamp
        | Mf::Unpack4xI8
        | Mf::Unpack4xU8
        | Mf::Unpack4x8snorm
        | Mf::Unpack4x8unorm
        | Mf::Unpack2x16snorm
        | Mf::Unpack2x16unorm
        | Mf::Unpack2x16float => Some(naga::Scalar::U32),
        Mf::Pack4xI8 | Mf::Pack4xI8Clamp => Some(naga::Scalar::I32),
        _ => None,
    }
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
        MathResult::I32 => ctx.intern_scalar(naga::Scalar::I32),
        MathResult::Vec4F32 => ctx.intern_vector(naga::VectorSize::Quad, naga::Scalar::F32),
        MathResult::Vec4I32 => ctx.intern_vector(naga::VectorSize::Quad, naga::Scalar::I32),
        MathResult::Vec4U32 => ctx.intern_vector(naga::VectorSize::Quad, naga::Scalar::U32),
        MathResult::Vec2F32 => ctx.intern_vector(naga::VectorSize::Bi, naga::Scalar::F32),
    };
    let handle = emit(
        ctx,
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
    let args: Vec<&Expr> = call.args.iter().collect();
    call_function(ctx, function, body, callee, None, &args, env)
}

/// A call to `callee`: `first`, a receiver already lowered, if there is one,
/// then `args`.
pub(super) fn call_function(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    callee: Handle<Function>,
    first: Option<Handle<Expression>>,
    args: &[&Expr],
    env: &mut Env,
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

    let skip = usize::from(first.is_some());
    if expected.len() != args.len() + skip {
        return Err(Error::WrongArgCount(name.into()));
    }
    let mut arg_values: Vec<_> = first.into_iter().collect();
    for (&arg, &want) in args.iter().zip(expected.iter().skip(skip)) {
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
            .append(Expression::CallResult(callee), ctx.span)
    });
    body.push(
        Statement::Call {
            function: callee,
            arguments: arg_values,
            result,
        },
        ctx.span,
    );
    Ok(result.zip(ret_ty))
}
