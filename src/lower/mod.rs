use core::num::NonZeroU32;
use std::collections::HashSet;

use naga::{
    AddressSpace, ArraySize, Block, Expression, Function, FunctionArgument, FunctionResult, Handle,
    Module, Scalar, ScalarKind, Span, Statement, Type, TypeInner, VectorSize,
};
use syn::{FnArg, Item, ItemFn, ReturnType, Signature};

use crate::Error;

mod call;
mod constant;
mod emit;
mod entry;
mod env;
mod expr;
mod global;
mod matrix;
mod method;
mod place;
mod ray;
mod stmt;
mod structure;
mod texture;
mod vector;

use emit::item_kind;
use env::{Env, Slot};
use stmt::{always_jumps, lower_block};

/// A lowered expression and the type it evaluates to. Every `lower_*` that
/// produces a value hands back one of these.
pub(super) type Typed = (Handle<Expression>, Handle<Type>);

/// Coarse shape of a type, as far as operators and constructors care.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Shape {
    Scalar(Scalar),
    Vector(VectorSize, Scalar),
    /// Columns, rows, component scalar.
    Matrix(VectorSize, VectorSize, Scalar),
    /// Structs and anything else without component-wise operators.
    Other,
}

impl Shape {
    /// Component scalar of a scalar, vector, or matrix.
    pub(super) fn scalar(self) -> Option<Scalar> {
        match self {
            Shape::Scalar(s) | Shape::Vector(_, s) | Shape::Matrix(_, _, s) => Some(s),
            Shape::Other => None,
        }
    }

    /// Component kind of a scalar or vector. Matrices are excluded because Naga
    /// treats them separately in every operator rule that uses this.
    pub(super) fn elem_kind(self) -> Option<ScalarKind> {
        match self {
            Shape::Scalar(s) | Shape::Vector(_, s) => Some(s.kind),
            Shape::Matrix(..) | Shape::Other => None,
        }
    }

    /// The scalar an untyped integer literal should take on in this context,
    /// mirroring Rust's integer literal inference. Floats are excluded: Rust
    /// would not turn `1` into `1.0` either.
    pub(super) fn int_hint(self) -> Option<Scalar> {
        match self.scalar() {
            Some(s) if matches!(s.kind, ScalarKind::Sint | ScalarKind::Uint) => Some(s),
            _ => None,
        }
    }
}

pub struct Context {
    pub module: Module,
    pub(super) globals: Vec<global::GlobalInfo>,
    pub(super) structs: Vec<(String, Handle<Type>)>,
    pub(super) consts: Vec<constant::ConstInfo>,
    /// Set by `lower_type` when it unwraps an address-space wrapper, and taken
    /// by the global being declared. A type mentions its space at most once,
    /// and only a global asks.
    pub(super) pending_space: Option<AddressSpace>,
    /// Locals in the function currently being lowered that are assigned or
    /// passed as storage. Everything else can stay a value.
    pub(super) addressed: HashSet<String>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            module: Module::default(),
            globals: Vec::new(),
            structs: Vec::new(),
            consts: Vec::new(),
            pending_space: None,
            addressed: HashSet::new(),
        }
    }

    pub fn lower_file(&mut self, file: syn::File) -> Result<(), Error> {
        for item in file.items {
            let (name, line) = item_location(&item);
            self.lower_item(item).map_err(|source| Error::At {
                item: name,
                line,
                source: Box::new(source),
            })?;
        }
        Ok(())
    }

    fn lower_item(&mut self, item: Item) -> Result<(), Error> {
        match item {
            // A shader module is also an ordinary Rust module, so it carries
            // the `use` that brings the shader prelude into scope and the
            // `mod` that lists its siblings. Neither says anything about the
            // shader.
            Item::Use(_) | Item::Mod(_) => Ok(()),
            Item::Fn(func) => self.lower_fn(func),
            Item::Static(st) => global::lower_static(self, st),
            Item::ForeignMod(fm) => global::lower_foreign_mod(self, fm),
            Item::Struct(st) => structure::lower_struct_item(self, st),
            Item::Const(c) => constant::lower_const_item(self, c),
            other => Err(Error::UnsupportedItem(item_kind(&other))),
        }
    }

    /// Intern a type that has no component structure of its own: an image or
    /// a sampler.
    pub(super) fn intern_handle_type(&mut self, inner: TypeInner) -> Handle<Type> {
        self.module
            .types
            .insert(Type { name: None, inner }, Span::UNDEFINED)
    }

    pub(super) fn intern_scalar(&mut self, scalar: Scalar) -> Handle<Type> {
        self.module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(scalar),
            },
            Span::UNDEFINED,
        )
    }

    pub(super) fn intern_vector(&mut self, size: VectorSize, scalar: Scalar) -> Handle<Type> {
        self.module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector { size, scalar },
            },
            Span::UNDEFINED,
        )
    }

    pub(super) fn shape(&self, ty: Handle<Type>) -> Shape {
        match self.module.types[ty].inner {
            TypeInner::Scalar(scalar) => Shape::Scalar(scalar),
            TypeInner::Vector { size, scalar } => Shape::Vector(size, scalar),
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => Shape::Matrix(columns, rows, scalar),
            _ => Shape::Other,
        }
    }

    /// `bool` for a scalar operand, `vecN<bool>` for a vector one: the result
    /// type of a comparison.
    pub(super) fn bool_like(&mut self, ty: Handle<Type>) -> Handle<Type> {
        match self.shape(ty) {
            Shape::Vector(size, _) => self.intern_vector(size, Scalar::BOOL),
            _ => self.intern_scalar(Scalar::BOOL),
        }
    }

    pub(super) fn as_scalar(&self, ty: Handle<Type>) -> Option<Scalar> {
        match self.module.types[ty].inner {
            TypeInner::Scalar(scalar) => Some(scalar),
            _ => None,
        }
    }

    pub(super) fn as_vector(&self, ty: Handle<Type>) -> Option<(VectorSize, Scalar)> {
        match self.module.types[ty].inner {
            TypeInner::Vector { size, scalar } => Some((size, scalar)),
            _ => None,
        }
    }

    pub(super) fn intern_matrix(
        &mut self,
        columns: VectorSize,
        rows: VectorSize,
        scalar: Scalar,
    ) -> Handle<Type> {
        self.module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Matrix {
                    columns,
                    rows,
                    scalar,
                },
            },
            Span::UNDEFINED,
        )
    }

    pub(super) fn as_matrix(&self, ty: Handle<Type>) -> Option<(VectorSize, VectorSize, Scalar)> {
        match self.module.types[ty].inner {
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => Some((columns, rows, scalar)),
            _ => None,
        }
    }

    pub(super) fn lower_type(&mut self, ty: &syn::Type) -> Result<Handle<Type>, Error> {
        let path = match ty {
            syn::Type::Path(path) if path.qself.is_none() => path,
            // `[T]` is a runtime-sized array: what a storage buffer holds.
            syn::Type::Slice(slice) => {
                let base = self.lower_type(&slice.elem)?;
                return self.intern_array(base, ArraySize::Dynamic);
            }
            // `[T; N]` is a fixed array.
            syn::Type::Array(array) => {
                let base = self.lower_type(&array.elem)?;
                let len = self.array_len(&array.len)?;
                return self.intern_array(base, ArraySize::Constant(len));
            }
            syn::Type::Paren(inner) => return self.lower_type(&inner.elem),
            // `&mut T` is WGSL's `ptr<function, T>`: an out-parameter.
            syn::Type::Reference(reference) => {
                if reference.lifetime.is_some() {
                    return Err(Error::UnsupportedType("lifetime".into()));
                }
                let base = self.lower_type(&reference.elem)?;
                return Ok(self.intern_handle_type(TypeInner::Pointer {
                    base,
                    space: AddressSpace::Function,
                }));
            }
            _ => return Err(Error::UnsupportedType("non-path type".into())),
        };
        let seg = path
            .path
            .segments
            .last()
            .ok_or_else(|| Error::UnsupportedType("empty path".into()))?;
        let name = seg.ident.to_string();

        // Textures and samplers take their own argument shapes —
        // `texture_storage_2d<Format, Access>` has two — so they are resolved
        // before the one-argument rule below.
        if name == "binding_array" {
            // The optional count is a const argument, so the element type is
            // picked out rather than taken as the only argument.
            let args = type_args_only(seg);
            let [base] = args[..] else {
                return Err(Error::UnsupportedType("binding_array".into()));
            };
            let base = self.lower_type(base)?;
            let size = match binding_array_len(seg)? {
                Some(len) => ArraySize::Constant(len),
                None => ArraySize::Dynamic,
            };
            return Ok(self.intern_handle_type(TypeInner::BindingArray { base, size }));
        }

        if let Some(result) = texture::parse_handle_type(self, &name, &collect_type_args(seg)?) {
            return result;
        }
        if let Some(ty) = ray::parse_ray_type(self, &name) {
            return Ok(ty);
        }
        if let Some(ty) = ray::special_struct(self, &name) {
            return Ok(ty);
        }

        let type_arg = match &seg.arguments {
            syn::PathArguments::None => None,
            syn::PathArguments::AngleBracketed(args) if args.args.len() == 1 => {
                match args.args.first() {
                    Some(syn::GenericArgument::Type(inner)) => Some(inner),
                    _ => return Err(Error::UnsupportedType(name)),
                }
            }
            _ => return Err(Error::UnsupportedType(name)),
        };

        // The address space is part of the type: `Uniform<T>` and friends wrap
        // what they hold, so a global is checkable Rust without an attribute.
        if let Some(space) = space_wrapper(&name) {
            let [inner] = type_args_only(seg)[..] else {
                return Err(Error::UnsupportedType(name));
            };
            let ty = self.lower_type(inner)?;
            self.pending_space = Some(space);
            return Ok(ty);
        }

        // `binding_array<T>` is a bound array of resources: a texture array in
        // a descriptor set, not memory.
        if name == "atomic" {
            let scalar = match type_arg {
                Some(inner) => lower_scalar_ident(inner)?,
                None => return Err(Error::UnsupportedType("atomic".into())),
            };
            return Ok(self.intern_handle_type(TypeInner::Atomic(scalar)));
        }

        if let Some((size, shorthand)) = parse_vec_ident(&name) {
            let scalar = match (shorthand, type_arg) {
                (Some(scalar), None) => scalar,
                (None, None) => Scalar::F32,
                (None, Some(inner)) => lower_scalar_ident(inner)?,
                (Some(_), Some(_)) => return Err(Error::UnsupportedType(name)),
            };
            return Ok(self.intern_vector(size, scalar));
        }

        if let Some((columns, rows, shorthand)) = parse_mat_ident(&name) {
            let scalar = match (shorthand, type_arg) {
                (Some(scalar), None) => scalar,
                (None, None) => Scalar::F32,
                (None, Some(inner)) => lower_scalar_ident(inner)?,
                (Some(_), Some(_)) => return Err(Error::UnsupportedType(name)),
            };
            return Ok(self.intern_matrix(columns, rows, scalar));
        }

        if type_arg.is_some() {
            return Err(Error::UnsupportedType(name));
        }
        // `usize` has no shader meaning, but `[T; N]` and `[T]` index by it and
        // nothing can change that, so a checkable shader has to be able to
        // write `arr[i as usize]`. An index is 32-bit on a GPU, so it is `u32`.
        match name.as_str() {
            "f32" => Ok(self.intern_scalar(Scalar::F32)),
            "u32" | "usize" => Ok(self.intern_scalar(Scalar::U32)),
            "i32" | "isize" => Ok(self.intern_scalar(Scalar::I32)),
            "bool" => Ok(self.intern_scalar(Scalar::BOOL)),
            other => self
                .struct_by_name(other)
                .ok_or_else(|| Error::UnsupportedType(other.into())),
        }
    }

    /// Functions and entry points share one namespace, as they do in WGSL.
    /// Without this, two `fn f` end up as `f` and `f_1` in the output and calls
    /// silently pick the first.
    pub(super) fn claim_fn_name(&self, name: &str) -> Result<(), Error> {
        let taken = self
            .module
            .functions
            .iter()
            .any(|(_, f)| f.name.as_deref() == Some(name))
            || self.module.entry_points.iter().any(|e| e.name == name);
        if taken {
            return Err(Error::DuplicateFunction(name.into()));
        }
        Ok(())
    }

    /// An array length: a literal, or a `const` naming one.
    fn array_len(&self, len: &syn::Expr) -> Result<NonZeroU32, Error> {
        let value = match len {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(int),
                ..
            }) => int.base10_parse::<u32>().map_err(Error::from)?,
            syn::Expr::Path(path) => {
                let name = path
                    .path
                    .get_ident()
                    .ok_or_else(|| Error::UnsupportedType("array length".into()))?
                    .to_string();
                let info = self
                    .consts
                    .iter()
                    .find(|c| c.name == name)
                    .ok_or(Error::UnknownIdent(name))?;
                match self.module.global_expressions[info.init_expr] {
                    naga::Expression::Literal(naga::Literal::U32(v)) => v,
                    naga::Expression::Literal(naga::Literal::I32(v)) if v >= 0 => v as u32,
                    _ => return Err(Error::UnsupportedType("array length".into())),
                }
            }
            _ => return Err(Error::UnsupportedType("array length".into())),
        };
        NonZeroU32::new(value).ok_or_else(|| Error::UnsupportedType("zero-length array".into()))
    }

    pub(super) fn struct_by_name(&mut self, name: &str) -> Option<Handle<Type>> {
        let declared = self
            .structs
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, h)| *h);
        // `RayDesc` and `RayIntersection` are Naga's, generated on first use.
        declared.or_else(|| ray::special_struct(self, name))
    }

    /// What `ty` points at, if it is a pointer.
    pub(super) fn pointee(&self, ty: Handle<Type>) -> Option<Handle<Type>> {
        match self.module.types[ty].inner {
            TypeInner::Pointer { base, .. } => Some(base),
            _ => None,
        }
    }

    pub(super) fn as_array(&self, ty: Handle<Type>) -> Option<(Handle<Type>, ArraySize)> {
        match self.module.types[ty].inner {
            TypeInner::Array { base, size, .. } => Some((base, size)),
            _ => None,
        }
    }

    /// Byte distance between consecutive elements of `base`, which Naga needs
    /// baked into the array type.
    pub(super) fn stride_of(&self, base: Handle<Type>) -> Result<u32, Error> {
        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(self.module.to_ctx())
            .map_err(|e| Error::UnsupportedType(e.to_string()))?;
        Ok(layouter[base].to_stride())
    }

    pub(super) fn intern_array(
        &mut self,
        base: Handle<Type>,
        size: ArraySize,
    ) -> Result<Handle<Type>, Error> {
        let stride = self.stride_of(base)?;
        Ok(self.module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array { base, size, stride },
            },
            Span::UNDEFINED,
        ))
    }

    pub(super) fn as_struct(&self, ty: Handle<Type>) -> Option<&[naga::StructMember]> {
        match &self.module.types[ty].inner {
            TypeInner::Struct { members, .. } => Some(members),
            _ => None,
        }
    }

    fn lower_fn(&mut self, item: ItemFn) -> Result<(), Error> {
        if !item.sig.generics.params.is_empty() {
            return Err(Error::UnsupportedItem(format!(
                "generic function `{}`",
                item.sig.ident
            )));
        }
        if item.sig.asyncness.is_some() || item.sig.abi.is_some() {
            return Err(Error::UnsupportedItem(format!(
                "async/extern function `{}`",
                item.sig.ident
            )));
        }

        let info = entry::parse_fn_attrs(&item.attrs)?;
        if info.stage.is_some() {
            return entry::lower_entry(self, item, info);
        }
        if info.workgroup_size.is_some() || info.return_binding.is_some() {
            return Err(Error::UnsupportedItem(
                "entry-point attribute on a regular function".into(),
            ));
        }

        let name = item.sig.ident.to_string();
        self.claim_fn_name(&name)?;
        // A function with no return type produces nothing, as in Rust; calls to
        // it are statements.
        let result = match &item.sig.output {
            ReturnType::Type(_, ty) if is_unit(ty) => None,
            ReturnType::Type(_, ty) => Some(FunctionResult {
                ty: self.lower_type(ty)?,
                binding: None,
            }),
            ReturnType::Default => None,
        };

        let mut function = Function {
            name: Some(name),
            arguments: Vec::new(),
            result,
            ..Default::default()
        };

        let mut env = Env::default();
        global::bind_globals(self, &mut function, &mut env);
        lower_signature(self, &mut function, &item.sig, &mut env)?;
        let mut body = Block::new();
        env.push_scope();
        self.addressed = stmt::addressed_names(&item.block);
        let tail = lower_block(self, &mut function, &mut body, &item.block, &mut env)?;
        env.pop_scope();
        match tail {
            Some((value, _)) => {
                body.push(Statement::Return { value: Some(value) }, Span::UNDEFINED)
            }
            None if function.result.is_some() && !always_jumps(&body) => {
                return Err(Error::MissingReturn(function.name.unwrap_or_default()))
            }
            None => {}
        }
        function.body = body;
        self.module.functions.append(function, Span::UNDEFINED);
        Ok(())
    }
}

/// How to name an item in an error, and the line it opens on.
fn item_location(item: &Item) -> (String, usize) {
    use syn::spanned::Spanned;
    let name = match item {
        Item::Fn(f) => format!("`fn {}`", f.sig.ident),
        Item::Struct(s) => format!("`struct {}`", s.ident),
        Item::Static(s) => format!("`static {}`", s.ident),
        Item::Const(c) => format!("`const {}`", c.ident),
        Item::ForeignMod(_) => "`extern` block".to_string(),
        other => item_kind(other),
    };
    (name, item.span().start().line)
}

/// Parse `vec2` / `Vec3` / `vec4f` / `vec3i` / `vec2u`.
/// `None` scalar means "default f32, or infer from constructor args".
pub(super) fn parse_vec_ident(name: &str) -> Option<(VectorSize, Option<Scalar>)> {
    match name {
        "vec2" | "Vec2" => Some((VectorSize::Bi, None)),
        "vec3" | "Vec3" => Some((VectorSize::Tri, None)),
        "vec4" | "Vec4" => Some((VectorSize::Quad, None)),
        "vec2f" | "Vec2f" => Some((VectorSize::Bi, Some(Scalar::F32))),
        "vec3f" | "Vec3f" => Some((VectorSize::Tri, Some(Scalar::F32))),
        "vec4f" | "Vec4f" => Some((VectorSize::Quad, Some(Scalar::F32))),
        "vec2i" | "Vec2i" => Some((VectorSize::Bi, Some(Scalar::I32))),
        "vec3i" | "Vec3i" => Some((VectorSize::Tri, Some(Scalar::I32))),
        "vec4i" | "Vec4i" => Some((VectorSize::Quad, Some(Scalar::I32))),
        "vec2u" | "Vec2u" => Some((VectorSize::Bi, Some(Scalar::U32))),
        "vec3u" | "Vec3u" => Some((VectorSize::Tri, Some(Scalar::U32))),
        "vec4u" | "Vec4u" => Some((VectorSize::Quad, Some(Scalar::U32))),
        _ => None,
    }
}

/// Parse `mat2` / `Mat4` / `mat2x3` / `mat4f`.
/// Scalar `None` means default `f32` (or infer from constructor args).
pub(super) fn parse_mat_ident(name: &str) -> Option<(VectorSize, VectorSize, Option<Scalar>)> {
    fn pair(c: char, r: char) -> Option<(VectorSize, VectorSize)> {
        let dim = |ch| match ch {
            '2' => Some(VectorSize::Bi),
            '3' => Some(VectorSize::Tri),
            '4' => Some(VectorSize::Quad),
            _ => None,
        };
        Some((dim(c)?, dim(r)?))
    }

    let (stem, scalar) = match name.as_bytes().last().copied() {
        Some(b'f' | b'F') if name.len() > 1 => (&name[..name.len() - 1], Some(Scalar::F32)),
        _ => (name, None),
    };

    match stem {
        "mat2" | "Mat2" => Some((VectorSize::Bi, VectorSize::Bi, scalar)),
        "mat3" | "Mat3" => Some((VectorSize::Tri, VectorSize::Tri, scalar)),
        "mat4" | "Mat4" => Some((VectorSize::Quad, VectorSize::Quad, scalar)),
        other => {
            let rest = other
                .strip_prefix("mat")
                .or_else(|| other.strip_prefix("Mat"))?;
            let b = rest.as_bytes();
            if b.len() == 3 && b[1] == b'x' {
                let (columns, rows) = pair(b[0] as char, b[2] as char)?;
                Some((columns, rows, scalar))
            } else {
                None
            }
        }
    }
}

/// The address space `Uniform<T>` and its siblings stand for.
fn space_wrapper(name: &str) -> Option<AddressSpace> {
    Some(match name {
        "Uniform" => AddressSpace::Uniform,
        "Storage" => AddressSpace::Storage {
            access: naga::StorageAccess::LOAD,
        },
        "StorageMut" => AddressSpace::Storage {
            access: naga::StorageAccess::LOAD.union(naga::StorageAccess::STORE),
        },
        "Workgroup" => AddressSpace::WorkGroup,
        "Private" => AddressSpace::Private,
        _ => return None,
    })
}

/// The type arguments of `seg`, ignoring any const ones.
fn type_args_only(seg: &syn::PathSegment) -> Vec<&syn::Type> {
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Vec::new();
    };
    args.args
        .iter()
        .filter_map(|arg| match arg {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect()
}

/// The count in `binding_array<T, N>`, which `syn` parses as a const argument.
fn binding_array_len(seg: &syn::PathSegment) -> Result<Option<NonZeroU32>, Error> {
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Ok(None);
    };
    for arg in &args.args {
        let value = match arg {
            syn::GenericArgument::Const(syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(int),
                ..
            })) => int.base10_parse::<u32>().map_err(Error::from)?,
            syn::GenericArgument::Type(_) => continue,
            _ => return Err(Error::UnsupportedType("binding_array".into())),
        };
        return NonZeroU32::new(value)
            .map(Some)
            .ok_or_else(|| Error::UnsupportedType("zero-length binding_array".into()));
    }
    Ok(None)
}

/// Every angle-bracketed type argument of `seg`, in order.
fn collect_type_args(seg: &syn::PathSegment) -> Result<Vec<&syn::Type>, Error> {
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return Ok(Vec::new());
    };
    args.args
        .iter()
        .map(|arg| match arg {
            syn::GenericArgument::Type(ty) => Ok(ty),
            _ => Err(Error::UnsupportedType(seg.ident.to_string())),
        })
        .collect()
}

fn lower_scalar_ident(ty: &syn::Type) -> Result<Scalar, Error> {
    let ident = match ty {
        syn::Type::Path(path) if path.qself.is_none() => path
            .path
            .get_ident()
            .ok_or_else(|| Error::UnsupportedType("path type".into()))?,
        _ => return Err(Error::UnsupportedType("non-scalar type argument".into())),
    };
    match ident.to_string().as_str() {
        "f32" => Ok(Scalar::F32),
        "u32" | "usize" => Ok(Scalar::U32),
        "i32" | "isize" => Ok(Scalar::I32),
        "bool" => Ok(Scalar::BOOL),
        other => Err(Error::UnsupportedType(other.into())),
    }
}

/// Is this the unit type, `()`?
pub(super) fn is_unit(ty: &syn::Type) -> bool {
    matches!(ty, syn::Type::Tuple(t) if t.elems.is_empty())
}

pub(super) fn lower_signature(
    ctx: &mut Context,
    function: &mut Function,
    sig: &Signature,
    env: &mut Env,
) -> Result<(), Error> {
    for arg in &sig.inputs {
        match arg {
            FnArg::Receiver(_) => return Err(Error::Receiver),
            FnArg::Typed(pat_ty) => {
                let name = match &*pat_ty.pat {
                    syn::Pat::Ident(ident)
                        if ident.by_ref.is_none() && ident.mutability.is_none() =>
                    {
                        ident.ident.to_string()
                    }
                    _ => return Err(Error::PatternParam),
                };
                let ty = ctx.lower_type(&pat_ty.ty)?;
                let index = function.arguments.len() as u32;
                function.arguments.push(FunctionArgument {
                    name: Some(name.clone()),
                    ty,
                    binding: None,
                });
                let expr = function
                    .expressions
                    .append(Expression::FunctionArgument(index), Span::UNDEFINED);
                // A pointer parameter names storage the caller owns, so it
                // binds as a place: `r.field = x` writes through it, and `&T`
                // marks the write as not allowed.
                match ctx.pointee(ty) {
                    Some(base) => {
                        let writable = matches!(
                            &*pat_ty.ty,
                            syn::Type::Reference(r) if r.mutability.is_some()
                        );
                        env.push_in(
                            name,
                            Slot::Ptr(expr),
                            base,
                            writable,
                            AddressSpace::Function,
                        );
                    }
                    None => env.push(name, Slot::Value(expr), ty),
                }
            }
        }
    }
    Ok(())
}
