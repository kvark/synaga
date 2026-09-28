use core::num::NonZeroU32;
use std::collections::{HashMap, HashSet};

use naga::{
    AddressSpace, ArraySize, Block, Expression, Function, FunctionArgument, FunctionResult, Handle,
    Module, Scalar, ScalarKind, Span, Type, TypeInner, VectorSize,
};
use syn::{FnArg, Item, ItemFn, ReturnType, Signature};

use crate::build::Bindings;
use crate::Error;

mod atomic;
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
mod scope;
mod stmt;
mod structure;
mod texture;
mod vector;

use emit::item_kind;
use env::{Env, Slot};
use scope::{Lowered, Ns, Scope, State};

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
    pub(super) consts: Vec<constant::ConstInfo>,
    /// Every item of every source, and how names reach them.
    pub(super) scope: Scope,
    /// The source whose item is being lowered, which is where its names are
    /// resolved from.
    pub(super) current: usize,
    /// The source of the item that failed, for [`crate::SourceError::index`].
    /// An item can fail while another one that needed it is being lowered,
    /// so this is not always the source that was being worked through.
    pub failed_source: Option<usize>,
    pub(super) cfg: crate::Cfg,
    /// Set by `lower_type` when it unwraps an address-space wrapper, and taken
    /// by the global being declared. A type mentions its space at most once,
    /// and only a global asks.
    pub(super) pending_space: Option<AddressSpace>,
    /// Locals in the function currently being lowered that are assigned or
    /// passed as storage. Everything else can stay a value.
    pub(super) addressed: HashSet<String>,
    /// Who the build says assigns bindings, if it said. Without it, a missing
    /// binding is left for Naga's validation to find.
    pub(super) bindings: Option<Bindings>,
    /// The structs that say `#[repr(C)]`, which the host may share.
    pub(super) host_reprs: HashMap<Handle<Type>, structure::HostRepr>,
    /// The size and alignment `rustc` gives each of those a buffer holds,
    /// once checked against the GPU's: a struct holding one needs it.
    pub(super) host_layouts: HashMap<Handle<Type>, structure::HostLayout>,
    /// The structs that derive `Default`. Any other `Default` is written by
    /// hand, where the shader cannot see it.
    pub(super) derived_defaults: HashSet<Handle<Type>>,
}

impl Context {
    pub fn new(cfg: crate::Cfg, bindings: Option<Bindings>) -> Self {
        Self {
            module: Module::default(),
            globals: Vec::new(),
            consts: Vec::new(),
            scope: Scope::default(),
            current: 0,
            failed_source: None,
            cfg,
            pending_space: None,
            addressed: HashSet::new(),
            bindings,
            host_reprs: HashMap::new(),
            host_layouts: HashMap::new(),
            derived_defaults: HashSet::new(),
        }
    }

    /// Lower every item of `files`, each named by the module it is, in an
    /// order that puts what an item needs before it.
    ///
    /// Globals and constants go first: every function binds the globals as it
    /// starts, so they have to exist by then. Everything else follows in
    /// source order, and anything an item needs that is not lowered yet is
    /// lowered right there. Naga wants a function after the functions it
    /// calls, and this is what gives it that.
    pub fn lower_sources(&mut self, files: Vec<(Option<String>, syn::File)>) -> Result<(), Error> {
        self.scope = Scope::index(files, &self.cfg).map_err(|err| {
            self.failed_source = Some(err.source);
            err.error
        })?;
        let (first, rest): (Vec<usize>, Vec<usize>) = (0..self.scope.entries.len())
            .partition(|&i| self.scope.entries[i].is_static_or_const());
        for index in first.into_iter().chain(rest) {
            self.ensure(index)?;
        }
        Ok(())
    }

    /// Lower entry `index` if it is not already, and say what it became.
    fn ensure(&mut self, index: usize) -> Result<Lowered, Error> {
        let entry = &mut self.scope.entries[index];
        match entry.state {
            State::Done(lowered) => return Ok(lowered),
            State::InProgress => {
                let name = entry.name.clone().map(|(_, n)| n).unwrap_or_default();
                return Err(Error::Cycle(name));
            }
            State::Pending => {}
        }
        entry.state = State::InProgress;
        let item = entry.item.take().expect("a pending entry keeps its item");
        let source = entry.source;
        let (label, line) = item_location(&item);

        // Lowering one item can start another, so whatever belongs to the
        // item in progress is set aside and put back afterwards.
        let current = std::mem::replace(&mut self.current, source);
        let addressed = std::mem::take(&mut self.addressed);
        let pending_space = self.pending_space.take();
        let result = self.lower_item(item);
        self.current = current;
        self.addressed = addressed;
        self.pending_space = pending_space;

        match result {
            Ok(lowered) => {
                self.scope.entries[index].state = State::Done(lowered);
                Ok(lowered)
            }
            // Already placed inside the item that failed: that is where the
            // problem is, not in whatever needed it.
            Err(err @ Error::At { .. }) => Err(err),
            Err(err) => {
                self.failed_source.get_or_insert(source);
                Err(Error::At {
                    item: label,
                    line,
                    source: Box::new(err),
                })
            }
        }
    }

    fn lower_item(&mut self, item: Item) -> Result<Lowered, Error> {
        match item {
            // A shader module is also an ordinary Rust module, so it may list
            // its siblings with `mod`, or keep something of its own in an
            // inline one. Neither is part of the shader.
            Item::Mod(_) => Ok(Lowered::Nothing),
            Item::Fn(func) => self.lower_fn(func),
            Item::Static(st) => global::lower_static(self, st).map(|()| Lowered::Static),
            Item::ForeignMod(fm) => global::lower_foreign_mod(self, fm).map(|()| Lowered::Static),
            Item::Struct(st) => structure::lower_struct_item(self, st).map(Lowered::Type),
            Item::Const(c) => constant::lower_const_item(self, c).map(Lowered::Const),
            // `type Color = vec4;` names a type another way.
            Item::Type(alias) => {
                if !alias.generics.params.is_empty() {
                    return Err(Error::UnsupportedItem("generic type alias".into()));
                }
                self.lower_type(&alias.ty).map(Lowered::Type)
            }
            other => Err(Error::UnsupportedItem(item_kind(&other))),
        }
    }

    /// Resolve `path` in namespace `ns` from the item being lowered, lowering
    /// what it names if nothing has yet.
    pub(super) fn resolve(&mut self, ns: Ns, path: &[String]) -> Result<Option<Lowered>, Error> {
        match self.scope.resolve(self.current, ns, path)? {
            Some(index) => self.ensure(index).map(Some),
            None => Ok(None),
        }
    }

    /// The function `path` names, if it names one the sources declare.
    pub(super) fn function(&mut self, path: &[String]) -> Result<Option<Handle<Function>>, Error> {
        match self.resolve(Ns::Value, path)? {
            Some(Lowered::Function(handle)) => Ok(Some(handle)),
            Some(Lowered::EntryPoint) => Err(Error::CallToEntryPoint(last(path))),
            _ => Ok(None),
        }
    }

    /// The `const` `path` names, as an index into `consts`.
    pub(super) fn constant(&mut self, path: &[String]) -> Result<Option<usize>, Error> {
        match self.resolve(Ns::Value, path)? {
            Some(Lowered::Const(index)) => Ok(Some(index)),
            _ => Ok(None),
        }
    }

    /// The value of `core::f32::consts::PI` and its siblings, by that path or
    /// by the name a `use` brought in.
    pub(super) fn float_const(&self, path: &[String]) -> Option<f32> {
        match path {
            [name] => {
                let outside = self.scope.external_path(self.current, name)?;
                constant::std_float(&outside)
            }
            _ => constant::std_float(path),
        }
    }

    /// Is `path` headed by one of the sources, as `brdf::sample` is, rather
    /// than by a type, as `vec3::splat` is?
    pub(super) fn is_module_path(&self, path: &[String]) -> bool {
        self.scope.is_module_path(self.current, path)
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
        let full = path_segments(&path.path);
        // What follows reads WGSL's names; `Texture2D` and the rest map onto them.
        let wgsl = wgsl_type_name(&name);

        // Textures and samplers take their own argument shapes —
        // `TextureStorage2D<Format, Access>` has two — so they are resolved
        // before the one-argument rule below.
        if wgsl == "binding_array" {
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

        if let Some(result) = texture::parse_handle_type(self, wgsl, &collect_type_args(seg)?) {
            return result;
        }
        if let Some(ty) = ray::parse_ray_type(self, wgsl) {
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

        // `synaga_shader::AtomicU32` and its sibling, which are the standard
        // atomics without an `Ordering`.
        let atomic = match name.as_str() {
            "AtomicU32" => Some(Scalar::U32),
            "AtomicI32" => Some(Scalar::I32),
            _ => None,
        };
        if let Some(scalar) = atomic {
            if type_arg.is_some() {
                return Err(Error::UnsupportedType(name));
            }
            return Ok(self.intern_handle_type(TypeInner::Atomic(scalar)));
        }
        // What `compare_exchange_weak` hands back: Naga's own struct, so its
        // fields are the ones the backends know.
        if name == "CompareExchange" {
            let scalar = match type_arg {
                Some(inner) => lower_scalar_ident(inner)?,
                None => return Err(Error::UnsupportedType(name)),
            };
            return Ok(self.module.generate_predeclared_type(
                naga::PredeclaredType::AtomicCompareExchangeWeakResult(scalar),
            ));
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
                .named_type(&full)?
                .ok_or_else(|| Error::UnsupportedType(other.into())),
        }
    }

    /// A host asks for an entry point by name, so two in different modules
    /// cannot share one. Other functions are reached through their module and
    /// may.
    pub(super) fn claim_entry_point_name(&self, name: &str) -> Result<(), Error> {
        if self.module.entry_points.iter().any(|e| e.name == name) {
            return Err(Error::DuplicateFunction(name.into()));
        }
        Ok(())
    }

    /// An array length: a literal, or a `const` naming one.
    fn array_len(&mut self, len: &syn::Expr) -> Result<NonZeroU32, Error> {
        let value = self
            .const_u32(len)?
            .ok_or_else(|| Error::UnsupportedType("array length".into()))?;
        NonZeroU32::new(value).ok_or_else(|| Error::UnsupportedType("zero-length array".into()))
    }

    /// A `u32` known before the shader runs: an integer literal, or a `const`
    /// naming one. `None` for anything else, which each caller refuses in its
    /// own words.
    pub(super) fn const_u32(&mut self, expr: &syn::Expr) -> Result<Option<u32>, Error> {
        match constant::strip_parens(expr) {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(int),
                ..
            }) => int.base10_parse::<u32>().map(Some).map_err(Error::from),
            syn::Expr::Path(path) => {
                let segments = path_segments(&path.path);
                let index = self
                    .constant(&segments)?
                    .ok_or_else(|| Error::UnknownIdent(last(&segments)))?;
                // A `const` may be defined as another one, so follow the chain
                // down to the literal.
                let mut init = self.consts[index].init_expr;
                loop {
                    match self.module.global_expressions[init] {
                        naga::Expression::Literal(naga::Literal::U32(v)) => return Ok(Some(v)),
                        naga::Expression::Literal(naga::Literal::I32(v)) if v >= 0 => {
                            return Ok(Some(v as u32))
                        }
                        naga::Expression::Constant(other) => {
                            init = self.module.constants[other].init
                        }
                        _ => return Ok(None),
                    }
                }
            }
            _ => Ok(None),
        }
    }

    /// The type `path` names: a struct or alias the sources declare, or one of
    /// the structs Naga predeclares.
    pub(super) fn named_type(&mut self, path: &[String]) -> Result<Option<Handle<Type>>, Error> {
        if let Some(Lowered::Type(handle)) = self.resolve(Ns::Type, path)? {
            return Ok(Some(handle));
        }
        // `RayDesc` and `RayIntersection` are Naga's, generated on first use.
        Ok(ray::special_struct(self, &last(path)))
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

    fn lower_fn(&mut self, item: ItemFn) -> Result<Lowered, Error> {
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
            return entry::lower_entry(self, item, info).map(|()| Lowered::EntryPoint);
        }
        if info.workgroup_size.is_some() || info.return_binding.is_some() {
            return Err(Error::UnsupportedItem(
                "entry-point attribute on a regular function".into(),
            ));
        }

        let name = item.sig.ident.to_string();
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
        stmt::lower_body(self, &mut function, &mut body, &item.block, &mut env)?;
        env.pop_scope();
        function.body = body;
        let handle = self.module.functions.append(function, Span::UNDEFINED);
        Ok(Lowered::Function(handle))
    }
}

/// The identifiers of `path`, in order.
pub(super) fn path_segments(path: &syn::Path) -> Vec<String> {
    path.segments.iter().map(|s| s.ident.to_string()).collect()
}

/// The last identifier of a path, which is the item's own name.
pub(super) fn last(path: &[String]) -> String {
    path.last().cloned().unwrap_or_default()
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

/// Parse a vector's name: the type `Vec3`, its constructor `vec3`, or one of
/// WGSL's shorthands, `vec3f`, `vec3i` and `vec3u`, which fix the scalar.
/// `None` scalar means a type argument says, or the constructor's arguments
/// do, or it is `f32`.
pub(super) fn parse_vec_ident(name: &str) -> Option<(VectorSize, Option<Scalar>)> {
    let (digit, scalar) = match name.strip_prefix("Vec") {
        Some(rest) => (rest, None),
        None => {
            let rest = name.strip_prefix("vec")?;
            match rest.as_bytes() {
                [_] => (rest, None),
                [_, b'f'] => (&rest[..1], Some(Scalar::F32)),
                [_, b'i'] => (&rest[..1], Some(Scalar::I32)),
                [_, b'u'] => (&rest[..1], Some(Scalar::U32)),
                _ => return None,
            }
        }
    };
    let size = match digit {
        "2" => VectorSize::Bi,
        "3" => VectorSize::Tri,
        "4" => VectorSize::Quad,
        _ => return None,
    };
    Some((size, scalar))
}

/// The WGSL name of a handle type, for one spelled the Rust way:
/// `TextureStorage2DArray` is `texture_storage_2d_array`. Anything else comes
/// back as it was, so a struct of the shader's own called `TextureParams` is
/// never taken for a texture.
pub(super) fn wgsl_type_name(name: &str) -> &str {
    match name {
        "Texture1D" => "texture_1d",
        "Texture2D" => "texture_2d",
        "Texture2DArray" => "texture_2d_array",
        "Texture3D" => "texture_3d",
        "TextureCube" => "texture_cube",
        "TextureCubeArray" => "texture_cube_array",
        "TextureMultisampled2D" => "texture_multisampled_2d",
        "TextureDepth2D" => "texture_depth_2d",
        "TextureDepth2DArray" => "texture_depth_2d_array",
        "TextureDepthCube" => "texture_depth_cube",
        "TextureDepthCubeArray" => "texture_depth_cube_array",
        "TextureDepthMultisampled2D" => "texture_depth_multisampled_2d",
        "TextureStorage1D" => "texture_storage_1d",
        "TextureStorage2D" => "texture_storage_2d",
        "TextureStorage2DArray" => "texture_storage_2d_array",
        "TextureStorage3D" => "texture_storage_3d",
        "Sampler" => "sampler",
        "SamplerComparison" => "sampler_comparison",
        "AccelerationStructure" => "acceleration_structure",
        "RayQuery" => "ray_query",
        "BindingArray" => "binding_array",
        other => other,
    }
}

/// The scalar a turbofish names, as `vec3::<u32>(..)` and `Vec3::<u32>::ZERO`
/// do: the type argument a type position would take, moved into an
/// expression.
pub(super) fn turbofish_scalar(seg: &syn::PathSegment) -> Result<Option<Scalar>, Error> {
    match &seg.arguments {
        syn::PathArguments::None => Ok(None),
        syn::PathArguments::AngleBracketed(args) => {
            angle_scalar(args, &seg.ident.to_string()).map(Some)
        }
        _ => Err(Error::UnsupportedType(seg.ident.to_string())),
    }
}

/// The one scalar in `<u32>`, after `what`: `Vec3::<u32>`, `v.cast::<u32>()`.
pub(super) fn angle_scalar(
    args: &syn::AngleBracketedGenericArguments,
    what: &str,
) -> Result<Scalar, Error> {
    match (args.args.len(), args.args.first()) {
        (1, Some(syn::GenericArgument::Type(ty))) => lower_scalar_ident(ty),
        _ => Err(Error::UnsupportedType(what.into())),
    }
}

/// A vector's scalar, from its name or a turbofish. `vec3u::<u32>` says it
/// twice, which is refused rather than checked for agreement.
pub(super) fn vec_scalar(
    name: &str,
    shorthand: Option<Scalar>,
    turbofish: Option<Scalar>,
) -> Result<Option<Scalar>, Error> {
    match (shorthand, turbofish) {
        (Some(_), Some(_)) => Err(Error::UnsupportedType(name.into())),
        (shorthand, turbofish) => Ok(shorthand.or(turbofish)),
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
