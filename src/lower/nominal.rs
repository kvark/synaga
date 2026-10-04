//! Enums and `bitflags!` sets: types the host names, which are a `u32` on
//! the GPU.
//!
//! A `#[repr(u32)]` enum and a flags set around a `u32` lay out as that
//! `u32`, so a struct the host shares can hold one, and a shader compares or
//! tests it the way Rust does: `params.mode == Mode::Depth`,
//! `params.flags.contains(Flags::SPACE)`. On the GPU the type is a `u32` that
//! keeps the Rust type's name. The name is what tells a flags set's `!`,
//! which stays within the declared flags, from a `u32`'s, which flips all
//! thirty-two bits.
//!
//! A set is declared with `bitflags!`, either whole (`pub struct Flags: u32`)
//! or on a newtype the source declares itself (`impl Flags: u32`), which is
//! how the newtype can derive `Shared` for the host.

use naga::{BinaryOperator, Block, Expression, Function, Handle, Literal, Scalar, Type};
use naga::{TypeInner, UnaryOperator};
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Token};

use super::emit::{binary, emit};
use super::env::Env;
use super::expr::lower_expr_hinted;
use super::{Context, Typed};
use crate::Error;

/// A fieldless enum, as the shader sees it.
pub(crate) struct EnumInfo {
    /// Only a `#[repr(u32)]` enum is sure to be the `u32` it is on the GPU.
    pub repr_u32: bool,
    /// The discriminant `Default` gives, when `Default` is derived.
    pub default: Option<u32>,
}

/// A `bitflags!` set.
pub(crate) struct FlagsInfo {
    /// The named flags, in order.
    pub flags: Vec<(String, u32)>,
    /// Every declared bit, the unnamed ones (`const _ = ..`) included.
    pub all: u32,
    /// `impl Flags: u32` rather than `struct Flags: u32`: the sources declare
    /// the type themselves, with its `#[repr]` and derives.
    pub external: bool,
    /// `#[repr(transparent)]`, which is what makes the set the `u32` it holds.
    pub transparent: bool,
    /// A derived `Default` is the empty set, which is zero.
    pub derives_default: bool,
}

impl FlagsInfo {
    pub fn flag(&self, name: &str) -> Option<u32> {
        self.flags.iter().find(|(n, _)| n == name).map(|&(_, v)| v)
    }
}

pub(crate) fn enum_info(item: &syn::ItemEnum, variants: &[(String, u32)]) -> EnumInfo {
    let default = super::structure::derives_default(&item.attrs)
        .then(|| {
            item.variants
                .iter()
                .zip(variants)
                .find(|(v, _)| v.attrs.iter().any(|a| a.path().is_ident("default")))
                .map(|(_, &(_, value))| value)
        })
        .flatten();
    EnumInfo {
        repr_u32: has_repr(&item.attrs, "u32"),
        default,
    }
}

pub(crate) fn has_repr(attrs: &[syn::Attribute], wanted: &str) -> bool {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("repr"))
        .any(|attr| {
            let mut found = false;
            let _ = attr.parse_nested_meta(|meta| {
                found |= meta.path.is_ident(wanted);
                Ok(())
            });
            found
        })
}

/// Is `mac` a `bitflags!` invocation, by whatever path it is reached?
pub(crate) fn is_bitflags(mac: &syn::Macro) -> bool {
    mac.path
        .segments
        .last()
        .is_some_and(|s| s.ident == "bitflags")
}

/// The sets a `bitflags!` invocation declares.
pub(crate) fn parse_bitflags(mac: &syn::Macro) -> Result<Vec<(String, FlagsInfo)>, Error> {
    let decls: Decls = mac.parse_body().map_err(Error::from)?;
    decls
        .0
        .into_iter()
        .map(|decl| {
            let name = decl.ident.to_string();
            if !matches!(&decl.bits, syn::Type::Path(p) if p.path.is_ident("u32")) {
                return Err(Error::Bitflags(format!(
                    "set `{name}` is not a `u32`, the only one the GPU shares"
                )));
            }
            let mut flags = Vec::new();
            let mut all = 0;
            for (flag, value) in &decl.flags {
                let value = flag_value(value, &flags)?;
                all |= value;
                if let Some(flag) = flag {
                    flags.push((flag.to_string(), value));
                }
            }
            let info = FlagsInfo {
                flags,
                all,
                external: decl.external,
                transparent: has_repr(&decl.attrs, "transparent"),
                derives_default: super::structure::derives_default(&decl.attrs),
            };
            Ok((name, info))
        })
        .collect()
}

struct Decl {
    attrs: Vec<syn::Attribute>,
    external: bool,
    ident: syn::Ident,
    bits: syn::Type,
    flags: Vec<(Option<syn::Ident>, Expr)>,
}

struct Decls(Vec<Decl>);

impl Parse for Decls {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut decls = Vec::new();
        while !input.is_empty() {
            let attrs = input.call(syn::Attribute::parse_outer)?;
            let external = input.peek(Token![impl]);
            if external {
                input.parse::<Token![impl]>()?;
            } else {
                input.parse::<syn::Visibility>()?;
                input.parse::<Token![struct]>()?;
            }
            let ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let bits = input.parse()?;
            let content;
            syn::braced!(content in input);
            let mut flags = Vec::new();
            while !content.is_empty() {
                content.call(syn::Attribute::parse_outer)?;
                content.parse::<Token![const]>()?;
                let flag = if content.peek(Token![_]) {
                    content.parse::<Token![_]>()?;
                    None
                } else {
                    Some(content.parse()?)
                };
                content.parse::<Token![=]>()?;
                let value = content.parse()?;
                content.parse::<Token![;]>()?;
                flags.push((flag, value));
            }
            decls.push(Decl {
                attrs,
                external,
                ident,
                bits,
                flags,
            });
        }
        Ok(Decls(decls))
    }
}

/// A flag's bits: a literal, `1 << n`, `|` and `&` of those, `!` of one, or
/// an earlier flag's `Self::A.bits()`.
fn flag_value(expr: &Expr, earlier: &[(String, u32)]) -> Result<u32, Error> {
    let unsupported = || Error::Bitflags(format!("flag value `{}`", quote_expr(expr)));
    Ok(match expr {
        Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(value),
            ..
        }) => value.base10_parse().map_err(Error::from)?,
        Expr::Paren(inner) => flag_value(&inner.expr, earlier)?,
        Expr::Group(inner) => flag_value(&inner.expr, earlier)?,
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Not(_)) => {
            !flag_value(&unary.expr, earlier)?
        }
        Expr::Binary(bin) => {
            let left = flag_value(&bin.left, earlier)?;
            let right = flag_value(&bin.right, earlier)?;
            match bin.op {
                syn::BinOp::BitOr(_) => left | right,
                syn::BinOp::BitAnd(_) => left & right,
                syn::BinOp::Shl(_) => left.checked_shl(right).ok_or_else(unsupported)?,
                _ => return Err(unsupported()),
            }
        }
        Expr::MethodCall(call) if call.method == "bits" && call.args.is_empty() => {
            let Expr::Path(path) = &*call.receiver else {
                return Err(unsupported());
            };
            let name = super::last(&super::path_segments(&path.path));
            earlier
                .iter()
                .find(|(n, _)| *n == name)
                .map(|&(_, v)| v)
                .ok_or_else(unsupported)?
        }
        _ => return Err(unsupported()),
    })
}

fn quote_expr(expr: &Expr) -> String {
    use quote::ToTokens;
    expr.to_token_stream().to_string()
}

impl Context {
    /// The GPU type of the enum or flags set `name`, if it is one.
    pub(super) fn nominal_type(&mut self, name: &str) -> Result<Option<Handle<Type>>, Error> {
        if let Some(info) = self.scope.flags.get(name) {
            if !info.transparent {
                return Err(Error::FlagsRepr(name.into()));
            }
        } else if let Some(info) = self.scope.enums.get(name) {
            if !info.repr_u32 {
                return Err(Error::EnumRepr(name.into()));
            }
        } else {
            return Ok(None);
        }
        Ok(Some(self.intern_named_u32(name)))
    }

    pub(super) fn intern_named_u32(&mut self, name: &str) -> Handle<Type> {
        self.module.types.insert(
            Type {
                name: Some(name.into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            self.span,
        )
    }

    /// The Rust type a named `u32` stands for.
    fn nominal_name(&self, ty: Handle<Type>) -> Option<&str> {
        match &self.module.types[ty] {
            Type {
                name: Some(name),
                inner: TypeInner::Scalar(Scalar::U32),
            } => Some(name),
            _ => None,
        }
    }

    /// The flags set `ty` is, if it is one.
    pub(super) fn flags_of(&self, ty: Handle<Type>) -> Option<&FlagsInfo> {
        self.nominal_name(ty).and_then(|n| self.scope.flags.get(n))
    }

    /// The enum `ty` is, if it is one.
    pub(super) fn enum_of(&self, ty: Handle<Type>) -> Option<&EnumInfo> {
        self.nominal_name(ty).and_then(|n| self.scope.enums.get(n))
    }
}

fn literal(ctx: &Context, function: &mut Function, value: u32) -> Handle<Expression> {
    function
        .expressions
        .append(Expression::Literal(Literal::U32(value)), ctx.span)
}

/// `Mode::Depth` and `Flags::SPACE`: the discriminant or the bits, as the
/// type they belong to. An enum that is not `#[repr(u32)]` has only its
/// discriminant, a plain `u32`, which is how an older shader wrote
/// `Mode::Depth as u32`.
pub(super) fn lower_const(
    ctx: &mut Context,
    function: &mut Function,
    ty_name: &str,
    item: &str,
) -> Result<Option<Typed>, Error> {
    let value = if let Some(info) = ctx.scope.flags.get(ty_name) {
        match info.flag(item) {
            Some(value) => value,
            None => return Ok(None),
        }
    } else if let Some(value) = ctx.scope.enum_variant(ty_name, item) {
        if !ctx.scope.enums.get(ty_name).is_some_and(|e| e.repr_u32) {
            let ty = ctx.intern_scalar(Scalar::U32);
            return Ok(Some((literal(ctx, function, value), ty)));
        }
        value
    } else {
        return Ok(None);
    };
    let ty = ctx.intern_named_u32(ty_name);
    Ok(Some((literal(ctx, function, value), ty)))
}

/// `Flags::empty()`, `Flags::all()` and the `from_bits` that cannot fail.
pub(super) fn lower_flags_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    ty_name: &str,
    method: &str,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    let Some(all) = ctx.scope.flags.get(ty_name).map(|info| info.all) else {
        return Ok(None);
    };
    let ty = ctx.intern_named_u32(ty_name);
    let value = match (method, args) {
        ("empty", []) => literal(ctx, function, 0),
        ("all", []) => literal(ctx, function, all),
        ("from_bits_retain", [bits]) => bits_arg(ctx, function, body, bits, env)?,
        // The bits no flag declares are dropped, as Rust drops them.
        ("from_bits_truncate", [bits]) => {
            let bits = bits_arg(ctx, function, body, bits, env)?;
            let mask = literal(ctx, function, all);
            binary(ctx, function, body, BinaryOperator::And, bits, mask)?
        }
        // `Flags::default()` is the zero value, where a derived `Default` is.
        _ => return Ok(None),
    };
    Ok(Some((value, ty)))
}

fn bits_arg(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    bits: &Expr,
    env: &mut Env,
) -> Result<Handle<Expression>, Error> {
    let (bits, bits_ty) = lower_expr_hinted(ctx, function, body, bits, env, Some(Scalar::U32))?;
    if ctx.module.types[bits_ty].inner != TypeInner::Scalar(Scalar::U32) {
        return Err(Error::TypeMismatch);
    }
    Ok(bits)
}

/// `flags.contains(other)` and the rest of what a set answers, if the
/// receiver is a set.
/// Is `name` one of the methods `bitflags!` gives a set, which come before
/// any the sources define?
pub(super) fn is_flags_method(name: &str) -> bool {
    matches!(
        name,
        "bits"
            | "is_empty"
            | "is_all"
            | "contains"
            | "intersects"
            | "union"
            | "intersection"
            | "symmetric_difference"
            | "difference"
            | "complement"
            | "insert"
            | "remove"
            | "toggle"
            | "set"
            | "iter"
            | "iter_names"
    )
}

pub(super) fn lower_flags_method(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    (value, ty): Typed,
    name: &str,
    args: &[&Expr],
    env: &mut Env,
) -> Result<Option<Typed>, Error> {
    use BinaryOperator as Bo;
    let Some(all) = ctx.flags_of(ty).map(|info| info.all) else {
        return Ok(None);
    };
    let mut other = |ctx: &mut Context, function: &mut Function, body: &mut Block| {
        let [arg] = args else {
            return Err(Error::WrongArgCount(name.into()));
        };
        let (arg, arg_ty) = super::expr::lower_expr(ctx, function, body, arg, env)?;
        if arg_ty != ty {
            return Err(Error::TypeMismatch);
        }
        Ok(arg)
    };
    let boolean = ctx.intern_scalar(Scalar::BOOL);
    let result = match name {
        "bits" if args.is_empty() => (value, ctx.intern_scalar(Scalar::U32)),
        "is_empty" if args.is_empty() => {
            let zero = literal(ctx, function, 0);
            (
                binary(ctx, function, body, Bo::Equal, value, zero)?,
                boolean,
            )
        }
        "is_all" if args.is_empty() => {
            let mask = literal(ctx, function, all);
            let held = binary(ctx, function, body, Bo::And, value, mask)?;
            (binary(ctx, function, body, Bo::Equal, held, mask)?, boolean)
        }
        "contains" => {
            let other = other(ctx, function, body)?;
            let held = binary(ctx, function, body, Bo::And, value, other)?;
            (
                binary(ctx, function, body, Bo::Equal, held, other)?,
                boolean,
            )
        }
        "intersects" => {
            let other = other(ctx, function, body)?;
            let held = binary(ctx, function, body, Bo::And, value, other)?;
            let zero = literal(ctx, function, 0);
            (
                binary(ctx, function, body, Bo::NotEqual, held, zero)?,
                boolean,
            )
        }
        "union" => {
            let other = other(ctx, function, body)?;
            (
                binary(ctx, function, body, Bo::InclusiveOr, value, other)?,
                ty,
            )
        }
        "intersection" => {
            let other = other(ctx, function, body)?;
            (binary(ctx, function, body, Bo::And, value, other)?, ty)
        }
        "symmetric_difference" => {
            let other = other(ctx, function, body)?;
            (
                binary(ctx, function, body, Bo::ExclusiveOr, value, other)?,
                ty,
            )
        }
        "difference" => {
            let other = other(ctx, function, body)?;
            (difference(ctx, function, body, value, other)?, ty)
        }
        "complement" if args.is_empty() => (complement(ctx, function, body, value, all)?, ty),
        _ => return Err(Error::UnsupportedMethod(name.into())),
    };
    Ok(Some(result))
}

/// `!flags`: the declared flags that are not set, as Rust's `complement`.
pub(super) fn complement(
    ctx: &Context,
    function: &mut Function,
    body: &mut Block,
    value: Handle<Expression>,
    all: u32,
) -> Result<Handle<Expression>, Error> {
    let flipped = emit(
        ctx,
        function,
        body,
        Expression::Unary {
            op: UnaryOperator::BitwiseNot,
            expr: value,
        },
    )?;
    let mask = literal(ctx, function, all);
    binary(ctx, function, body, BinaryOperator::And, flipped, mask)
}

/// `a - b`: the flags of `a` that `b` does not have.
pub(super) fn difference(
    ctx: &Context,
    function: &mut Function,
    body: &mut Block,
    left: Handle<Expression>,
    right: Handle<Expression>,
) -> Result<Handle<Expression>, Error> {
    let _span = ctx.span;
    let flipped = emit(
        ctx,
        function,
        body,
        Expression::Unary {
            op: UnaryOperator::BitwiseNot,
            expr: right,
        },
    )?;
    binary(ctx, function, body, BinaryOperator::And, left, flipped)
}
