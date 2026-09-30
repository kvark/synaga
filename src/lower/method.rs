//! The spellings a shader needs when it is also being type-checked by Rust.
//!
//! Rust cannot give one piece of memory a hundred overlapping names, cannot
//! overload a function on arity, and cannot make `a < b` yield one `bool` per
//! lane. So a checkable shader writes `v.xyz()`, `vec3::splat(x)`,
//! `v.extend(w)` and `a.cmple(b)` where WGSL writes `v.xyz`, `vec3(x)`,
//! `vec3(v, w)` and `a <= b`.
//!
//! This module reads those, and produces exactly what the WGSL spellings do.

use naga::{BinaryOperator, Block, Expression, Function, Scalar, VectorSize};
use syn::Expr;

use super::emit::emit;
use super::env::Env;
use super::expr::{bin_result_ty, lower_expr, lower_expr_hinted};
use super::place::swizzle_components;
use super::{parse_vec_ident, Context, Shape, Typed};
use crate::Error;

/// A method call. What the receiver is decides what the method means: an
/// atomic's `load` is not a texture's.
///
/// Some receivers are storage rather than a value. An atomic changes in place
/// and a ray query advances, so for those the receiver is found as a place
/// first, and its type says which it is. Anything else is loaded, as it would
/// be in any other expression.
pub(super) fn lower_method_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprMethodCall,
    env: &mut Env,
    hint: Option<Scalar>,
) -> Result<Typed, Error> {
    lower_method_any(ctx, function, body, call, env, hint)?
        .ok_or_else(|| Error::ValueFromStatement(call.method.to_string()))
}

/// A method call anywhere: what it produces, or `None` for one that only
/// acts, like a texture's `store`, which is fine in statement or tail
/// position. `hint` is where the value goes, which is all `v.cast()` has to
/// say what it casts to.
pub(super) fn lower_method_any(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprMethodCall,
    env: &mut Env,
    hint: Option<Scalar>,
) -> Result<Option<Typed>, Error> {
    let name = call.method.to_string();
    let args: Vec<&Expr> = call.args.iter().collect();

    let receiver = match super::place::lower_place(ctx, function, body, &call.receiver, env)? {
        Some(place) => {
            // `emit_end.set(n)` writes a whole resource, which is what
            // `*emit_end = n` would, if Rust could assign through a `static`.
            if name == "set" && is_resource(function, &place) {
                return lower_set(ctx, function, body, place, &args, env).map(|()| None);
            }
            if let Some(method) = user_method(ctx, place.ty, &name)? {
                let receiver = Receiver::Place(place);
                return lower_user_method(ctx, function, body, method, &name, receiver, &args, env);
            }
            match ctx.module.types[place.ty].inner {
                naga::TypeInner::Atomic(scalar) => {
                    return super::atomic::lower_atomic_method(
                        ctx, function, body, place, scalar, &name, &args, env,
                    );
                }
                naga::TypeInner::RayQuery { .. } => {
                    return super::ray::lower_ray_method(
                        ctx, function, body, place, &name, &args, env,
                    );
                }
                // A runtime-sized array has a length only where it lives.
                naga::TypeInner::Array {
                    size: naga::ArraySize::Dynamic,
                    ..
                } if name == "len" && args.is_empty() => {
                    let handle = emit(function, body, Expression::ArrayLength(place.pointer))?;
                    return Ok(Some((handle, ctx.intern_scalar(Scalar::U32))));
                }
                _ => {}
            }
            let ty = place.ty;
            (super::place::load(function, body, &place)?, ty)
        }
        None => {
            let value = lower_expr(ctx, function, body, &call.receiver, env)?;
            if let Some(method) = user_method(ctx, value.1, &name)? {
                let receiver = Receiver::Value(value);
                return lower_user_method(ctx, function, body, method, &name, receiver, &args, env);
            }
            value
        }
    };
    if super::texture::is_image(ctx, receiver.1) {
        return super::texture::lower_texture_method(
            ctx, function, body, receiver, &name, &args, env,
        );
    }
    if name == "cast" {
        return lower_cast(ctx, function, body, receiver, call, hint).map(Some);
    }
    if let Some(typed) =
        super::nominal::lower_flags_method(ctx, function, body, receiver, &name, &args, env)?
    {
        return Ok(Some(typed));
    }
    lower_value_method(ctx, function, body, receiver, &name, &args, env).map(Some)
}

/// What a method call's receiver turned out to be: storage, or a value.
enum Receiver {
    Place(super::place::Place),
    Value(Typed),
}

/// The method `name` the sources define on `ty`, unless `ty` has one of that
/// name of its own: a vector's `dot` is the vector's, as an inherent method
/// comes before a trait's in Rust.
fn user_method(
    ctx: &mut Context,
    ty: naga::Handle<naga::Type>,
    name: &str,
) -> Result<Option<super::Method>, Error> {
    let own = match ctx.shape(ty) {
        Shape::Other => ctx.flags_of(ty).is_some() && super::nominal::is_flags_method(name),
        _ => is_value_method(name),
    };
    if own || super::texture::is_image(ctx, ty) {
        return Ok(None);
    }
    ctx.find_method(ty, name)
}

/// A method of a vector, scalar or matrix, by its name alone.
fn is_value_method(name: &str) -> bool {
    swizzle_components(name).is_some()
        || compare_op(name).is_some()
        || super::call::is_math_method(name)
        || matches!(
            name,
            "len"
                | "extend"
                | "truncate"
                | "to_bits"
                | "all"
                | "any"
                | "element_sum"
                | "cast"
                | "get_mut"
                | "set"
        )
}

/// `receiver.name(args)`, for a method the sources define.
#[allow(clippy::too_many_arguments)]
fn lower_user_method(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    method: super::Method,
    name: &str,
    receiver: Receiver,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    let Some(kind) = method.receiver else {
        return Err(Error::AssociatedFunction(name.into()));
    };
    let first = match (kind, receiver) {
        (super::ReceiverKind::Value, Receiver::Value((value, _))) => value,
        (super::ReceiverKind::Value, Receiver::Place(place)) => {
            super::place::load(function, body, &place)?
        }
        (super::ReceiverKind::Mut, Receiver::Place(place)) => {
            if !place.writable {
                return Err(Error::AssignToReadonly(place.root));
            }
            if place.space != naga::AddressSpace::Function {
                return Err(Error::MutSelfNotLocal(name.into()));
            }
            place.pointer
        }
        (super::ReceiverKind::Mut, Receiver::Value(_)) => {
            return Err(Error::MutSelfNotLocal(name.into()))
        }
    };
    let callee = ctx.method_function(method.entry)?;
    super::call::call_function(ctx, function, body, callee, Some(first), args, env)
}

/// `Material::from_metallic_roughness(..)`, `Self::new(..)`, or a method
/// called through its type, `Quat::inverse(q)`: an associated function the
/// sources define. `None` when the type has none of that name, and the
/// builtins get their turn.
pub(super) fn lower_user_assoc_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    on: &OnType,
    name: &str,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Option<Option<Typed>>, Error> {
    let Some(ty) = super::call::zero_value_type(ctx, on.path, on.turbofish, on.hint)? else {
        return Ok(None);
    };
    // A vector's `splat` and `from` are its own.
    if ctx.shape(ty) != Shape::Other && matches!(name, "splat" | "from" | "from_bits") {
        return Ok(None);
    }
    let Some(method) = ctx.find_method(ty, name)? else {
        return Ok(None);
    };
    let (first, rest) = match method.receiver {
        None => (None, args),
        Some(kind) => {
            let [receiver, rest @ ..] = args else {
                return Err(Error::WrongArgCount(name.into()));
            };
            let first = match kind {
                super::ReceiverKind::Value => {
                    let (value, value_ty) = lower_expr(ctx, function, body, receiver, env)?;
                    if value_ty != ty {
                        return Err(Error::TypeMismatch);
                    }
                    value
                }
                super::ReceiverKind::Mut => {
                    super::call::pointer_arg(ctx, function, body, receiver, env, ty)?
                }
            };
            (Some(first), rest)
        }
    };
    let callee = ctx.method_function(method.entry)?;
    super::call::call_function(ctx, function, body, callee, first, rest, env).map(Some)
}

/// Is `place` a resource itself, rather than a part of one or a local?
fn is_resource(function: &Function, place: &super::place::Place) -> bool {
    matches!(
        function.expressions[place.pointer],
        Expression::GlobalVariable(_)
    )
}

/// `resource.set(value)`: a store of the whole of what the resource holds.
fn lower_set(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    place: super::place::Place,
    args: &[&Expr],
    env: &mut Env,
) -> Result<(), Error> {
    let [value] = args else {
        return Err(Error::WrongArgCount("set".into()));
    };
    if !place.writable {
        return Err(Error::AssignToReadonly(place.root));
    }
    let hint = ctx.shape(place.ty).int_hint();
    let (value, value_ty) = lower_expr_hinted(ctx, function, body, value, env, hint)?;
    if value_ty != place.ty {
        return Err(Error::TypeMismatch);
    }
    body.push(
        naga::Statement::Store {
            pointer: place.pointer,
            value,
        },
        naga::Span::UNDEFINED,
    );
    Ok(())
}

/// `v.cast::<i32>()`: every lane converted, as `vec3<i32>(v)` converts them.
/// Without a turbofish, the scalar is the one where the value goes, as `rustc`
/// infers it; with nothing there either, it is `f32`, as a bare `Vec3` is.
fn lower_cast(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    (value, ty): Typed,
    call: &syn::ExprMethodCall,
    hint: Option<Scalar>,
) -> Result<Typed, Error> {
    if !call.args.is_empty() {
        return Err(Error::WrongArgCount("cast".into()));
    }
    let Shape::Vector(size, _) = ctx.shape(ty) else {
        return Err(Error::UnsupportedMethod("cast".into()));
    };
    let turbofish = match &call.turbofish {
        Some(args) => Some(super::angle_scalar(args, "cast")?),
        None => None,
    };
    let scalar = turbofish.or(hint).unwrap_or(Scalar::F32);
    let handle = emit(
        function,
        body,
        Expression::As {
            expr: value,
            kind: scalar.kind,
            convert: Some(scalar.width),
        },
    )?;
    Ok((handle, ctx.intern_vector(size, scalar)))
}

/// `v.xyz()`, `v.extend(w)`, `a.cmple(b)` and the rest, on a value.
fn lower_value_method(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    (base, base_ty): Typed,
    name: &str,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Typed, Error> {
    let name = name.to_string();

    // A swizzle takes no arguments and is named only by its components.
    if args.is_empty() {
        if let Some(components) = swizzle_components(&name) {
            return super::vector::swizzle(ctx, function, body, base, base_ty, &components, &name);
        }
    }

    match (name.as_str(), args) {
        // A fixed array's length is part of its type; `usize` is `u32` here.
        ("len", []) if ctx.as_array(base_ty).is_some() => {
            let Some((_, naga::ArraySize::Constant(len))) = ctx.as_array(base_ty) else {
                return Err(Error::UnsupportedMethod(name));
            };
            let handle = function.expressions.append(
                Expression::Literal(naga::Literal::U32(len.get())),
                naga::Span::UNDEFINED,
            );
            Ok((handle, ctx.intern_scalar(Scalar::U32)))
        }
        ("extend", [value]) => {
            let Shape::Vector(size, scalar) = ctx.shape(base_ty) else {
                return Err(Error::UnsupportedMethod(name));
            };
            let wider = match size {
                VectorSize::Bi => VectorSize::Tri,
                VectorSize::Tri => VectorSize::Quad,
                VectorSize::Quad => return Err(Error::UnsupportedMethod(name)),
            };
            let hint = Shape::Scalar(scalar).int_hint();
            let (value, value_ty) = lower_expr_hinted(ctx, function, body, value, env, hint)?;
            if ctx.shape(value_ty) != Shape::Scalar(scalar) {
                return Err(Error::TypeMismatch);
            }
            let ty = ctx.intern_vector(wider, scalar);
            let handle = emit(
                function,
                body,
                Expression::Compose {
                    ty,
                    components: vec![base, value],
                },
            )?;
            Ok((handle, ty))
        }
        // `x.to_bits()`: a float's bits as a `u32`, `bitcast<u32>(x)` in WGSL.
        ("to_bits", []) if ctx.shape(base_ty) == Shape::Scalar(Scalar::F32) => {
            bitcast(ctx, function, body, base, Scalar::U32)
        }
        ("truncate", []) => {
            let Shape::Vector(size, _) = ctx.shape(base_ty) else {
                return Err(Error::UnsupportedMethod(name));
            };
            let keep = match size {
                VectorSize::Bi => return Err(Error::UnsupportedMethod(name)),
                VectorSize::Tri => vec![0, 1],
                VectorSize::Quad => vec![0, 1, 2],
            };
            super::vector::swizzle(ctx, function, body, base, base_ty, &keep, &name)
        }
        // `mask.all()`: a `Vec3<bool>` folded to one `bool`, as glam's `BVec3`
        // does it.
        ("all" | "any", []) if matches!(ctx.shape(base_ty), Shape::Vector(_, s) if s == Scalar::BOOL) =>
        {
            let fun = match name.as_str() {
                "all" => naga::RelationalFunction::All,
                _ => naga::RelationalFunction::Any,
            };
            let handle = emit(
                function,
                body,
                Expression::Relational {
                    fun,
                    argument: base,
                },
            )?;
            Ok((handle, ctx.intern_scalar(Scalar::BOOL)))
        }
        // `v.element_sum()`, as glam names it: the dot product with ones.
        ("element_sum", []) => {
            let Shape::Vector(size, scalar) = ctx.shape(base_ty) else {
                return Err(Error::UnsupportedMethod(name));
            };
            let one = match scalar.kind {
                naga::ScalarKind::Float => naga::Literal::F32(1.0),
                naga::ScalarKind::Sint => naga::Literal::I32(1),
                naga::ScalarKind::Uint => naga::Literal::U32(1),
                _ => return Err(Error::UnsupportedMethod(name)),
            };
            let one = function
                .expressions
                .append(Expression::Literal(one), naga::Span::UNDEFINED);
            let ones = emit(function, body, Expression::Splat { size, value: one })?;
            let expr = Expression::Math {
                fun: naga::MathFunction::Dot,
                arg: base,
                arg1: Some(ones),
                arg2: None,
                arg3: None,
            };
            Ok((emit(function, body, expr)?, ctx.intern_scalar(scalar)))
        }
        (cmp, [rhs]) if compare_op(cmp).is_some() => {
            let op = compare_op(cmp).expect("checked above");
            let (left, left_ty) = (base, base_ty);
            let hint = ctx.shape(left_ty).int_hint();
            let (right, right_ty) = lower_expr_hinted(ctx, function, body, rhs, env, hint)?;
            let ty = bin_result_ty(ctx, op, left_ty, right_ty)?;
            let handle = emit(function, body, Expression::Binary { op, left, right })?;
            Ok((handle, ty))
        }
        _ => {
            // `y.asin()`, `v.dot(w)`: the same builtins as the free functions,
            // with the receiver as the first argument, under the names `f32`
            // and the vectors have them.
            if let Some(typed) = super::call::lower_math_method(
                ctx,
                function,
                body,
                (base, base_ty),
                &name,
                args,
                env,
            )? {
                return Ok(typed);
            }
            Err(Error::UnsupportedMethod(name))
        }
    }
}

/// The same bits, read as another scalar of the same width.
fn bitcast(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    value: naga::Handle<Expression>,
    to: Scalar,
) -> Result<Typed, Error> {
    let expr = Expression::As {
        expr: value,
        kind: to.kind,
        convert: None,
    };
    Ok((emit(function, body, expr)?, ctx.intern_scalar(to)))
}

/// The lane-wise comparisons, which Rust's operators cannot express.
fn compare_op(name: &str) -> Option<BinaryOperator> {
    Some(match name {
        "cmpeq" => BinaryOperator::Equal,
        "cmpne" => BinaryOperator::NotEqual,
        "cmplt" => BinaryOperator::Less,
        "cmple" => BinaryOperator::LessEqual,
        "cmpgt" => BinaryOperator::Greater,
        "cmpge" => BinaryOperator::GreaterEqual,
        _ => return None,
    })
}

/// `vec3::splat(x)`, `vec4::from(v)`, `vec4::ZERO`: a call or a constant
/// qualified by the type it belongs to.
/// The type a qualified call or constant is on, as its path spells it.
pub(super) struct OnType<'a> {
    pub path: &'a [String],
    /// A vector's scalar, when a turbofish says it: `Vec3::<u32>::splat(1)`.
    pub turbofish: Option<Scalar>,
    /// Where the value goes, for a vector whose path says no scalar.
    pub hint: Option<Scalar>,
}

impl OnType<'_> {
    /// The vector's scalar, from its name, its turbofish or where it goes.
    fn scalar(&self, name: &str, shorthand: Option<Scalar>) -> Result<Option<Scalar>, Error> {
        Ok(super::vec_scalar(name, shorthand, self.turbofish)?.or(self.hint))
    }
}

pub(super) fn lower_qualified_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    on: OnType,
    method: &str,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Typed, Error> {
    let ty_name = &super::last(on.path);
    // An enum's derived `Default` is its `#[default]` variant, whatever that
    // variant's discriminant is.
    if method == "default" && args.is_empty() {
        let default = ctx.scope.enums.get(ty_name).and_then(|e| e.default);
        if let Some(value) = default {
            let ty = ctx
                .nominal_type(ty_name)?
                .ok_or_else(|| Error::EnumRepr(ty_name.clone()))?;
            let handle = function.expressions.append(
                Expression::Literal(naga::Literal::U32(value)),
                naga::Span::UNDEFINED,
            );
            return Ok((handle, ty));
        }
    }
    if let Some(typed) =
        super::nominal::lower_flags_call(ctx, function, body, ty_name, method, args, env)?
    {
        return Ok(typed);
    }
    if let Some(value) = super::ray::flags_call(ty_name, method) {
        if !args.is_empty() {
            return Err(Error::WrongArgCount(format!("{ty_name}::{method}")));
        }
        return Ok(u32_literal(ctx, function, value));
    }
    // `T::default()` is how Rust spells a zero value, and WGSL's `T()` is the
    // same thing. Any type may have one, so this comes before the vector names.
    if method == "default" && args.is_empty() {
        if let Some(ty) = super::call::zero_value_type(ctx, on.path, on.turbofish, on.hint)? {
            if let Some(unseen) = super::structure::unseen_default(ctx, ty) {
                return Err(Error::UnseenDefault(ty_name.clone(), unseen));
            }
            let handle = function
                .expressions
                .append(Expression::ZeroValue(ty), naga::Span::UNDEFINED);
            return Ok((handle, ty));
        }
    }

    // `f32::from_bits(n)`: a `u32`'s bits read as a float, which WGSL spells
    // `bitcast<f32>(n)`.
    if ty_name == "f32" && method == "from_bits" {
        let [bits] = args else {
            return Err(Error::WrongArgCount(format!("{ty_name}::{method}")));
        };
        let u32_ = Some(Scalar::U32);
        let (bits, bits_ty) = lower_expr_hinted(ctx, function, body, bits, env, u32_)?;
        if ctx.shape(bits_ty) != Shape::Scalar(Scalar::U32) {
            return Err(Error::TypeMismatch);
        }
        return bitcast(ctx, function, body, bits, Scalar::F32);
    }

    let Some((size, shorthand)) = parse_vec_ident(ty_name) else {
        return Err(Error::UnsupportedMethod(format!("{ty_name}::{method}")));
    };
    let shorthand = on.scalar(ty_name, shorthand)?;

    match (method, args) {
        ("splat", [value]) => {
            let hint = shorthand.and_then(|s| Shape::Scalar(s).int_hint());
            let (value, value_ty) = lower_expr_hinted(ctx, function, body, value, env, hint)?;
            let Shape::Scalar(scalar) = ctx.shape(value_ty) else {
                return Err(Error::TypeMismatch);
            };
            if matches!(shorthand, Some(want) if want != scalar) {
                return Err(Error::TypeMismatch);
            }
            let ty = ctx.intern_vector(size, scalar);
            let handle = emit(function, body, Expression::Splat { size, value })?;
            Ok((handle, ty))
        }
        // `vec4::from(v)` converts a vector's components, which a shader spells
        // as a cast. Rust's `as` only works on primitives, so vectors take this.
        ("from", [value]) => {
            let (value, value_ty) = lower_expr(ctx, function, body, value, env)?;
            let Shape::Vector(from_size, _) = ctx.shape(value_ty) else {
                return Err(Error::TypeMismatch);
            };
            if from_size != size {
                return Err(Error::TypeMismatch);
            }
            let scalar = shorthand.unwrap_or(Scalar::F32);
            let ty = ctx.intern_vector(size, scalar);
            let handle = emit(
                function,
                body,
                Expression::As {
                    expr: value,
                    kind: scalar.kind,
                    convert: Some(scalar.width),
                },
            )?;
            Ok((handle, ty))
        }
        _ => Err(Error::UnsupportedMethod(format!("{ty_name}::{method}"))),
    }
}

/// `Vec4::ZERO` and `Vec4::ONE`, which name a value rather than call anything,
/// `Mode::Variance`, which names an enum's discriminant, and `u32::MAX`, a
/// primitive's own.
pub(super) fn lower_qualified_const(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    on: OnType,
    constant: &str,
) -> Result<Typed, Error> {
    let ty_name = &super::last(on.path);
    if let Some(typed) = super::nominal::lower_const(ctx, function, ty_name, constant)? {
        return Ok(typed);
    }
    if let Some(value) = super::ray::typed_const(ty_name, constant) {
        return Ok(u32_literal(ctx, function, value));
    }
    if let Some(literal) = super::constant::scalar_const(ty_name, constant) {
        let ty = ctx.intern_scalar(literal.scalar());
        let handle = function
            .expressions
            .append(Expression::Literal(literal), naga::Span::UNDEFINED);
        return Ok((handle, ty));
    }
    let Some((size, shorthand)) = parse_vec_ident(ty_name) else {
        return Err(Error::UnknownIdent(format!("{ty_name}::{constant}")));
    };
    let scalar = on.scalar(ty_name, shorthand)?.unwrap_or(Scalar::F32);
    let ty = ctx.intern_vector(size, scalar);
    match constant {
        "ZERO" => {
            let handle = function
                .expressions
                .append(Expression::ZeroValue(ty), naga::Span::UNDEFINED);
            Ok((handle, ty))
        }
        "ONE" => {
            let one = match scalar.kind {
                naga::ScalarKind::Float => naga::Literal::F32(1.0),
                naga::ScalarKind::Sint => naga::Literal::I32(1),
                naga::ScalarKind::Uint => naga::Literal::U32(1),
                _ => naga::Literal::Bool(true),
            };
            let value = function
                .expressions
                .append(Expression::Literal(one), naga::Span::UNDEFINED);
            let handle = emit(function, body, Expression::Splat { size, value })?;
            Ok((handle, ty))
        }
        _ => Err(Error::UnknownIdent(format!("{ty_name}::{constant}"))),
    }
}

/// `value` as a `u32` literal.
fn u32_literal(ctx: &mut Context, function: &mut Function, value: u32) -> Typed {
    let handle = function.expressions.append(
        Expression::Literal(naga::Literal::U32(value)),
        naga::Span::UNDEFINED,
    );
    (handle, ctx.intern_scalar(Scalar::U32))
}
