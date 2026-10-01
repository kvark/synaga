use naga::{
    AddressSpace, Expression, Function, GlobalVariable, Handle, MemoryDecorations, ResourceBinding,
    StorageAccess, Type,
};
use syn::{Attribute, Expr, ForeignItem, ItemForeignMod, ItemStatic, LitInt};

use super::constant::strip_parens;
use super::env::{Env, Slot};
use super::Context;
use crate::build::Bindings;
use crate::Error;

pub(crate) struct GlobalInfo {
    pub name: String,
    pub handle: Handle<GlobalVariable>,
    pub ty: Handle<Type>,
    pub writable: bool,
    pub space: AddressSpace,
}

#[derive(Clone, Copy)]
enum SpaceKind {
    Uniform,
    Storage {
        write: bool,
    },
    /// Shared across a workgroup, zero-initialised each dispatch.
    Workgroup,
    /// Private to each invocation.
    Private,
}

struct ResourceInfo {
    group: Option<u32>,
    binding: Option<u32>,
    space: Option<SpaceKind>,
}

pub(super) fn bind_globals(ctx: &Context, function: &mut Function, env: &mut Env) {
    for g in &ctx.globals {
        let expr = function
            .expressions
            .append(Expression::GlobalVariable(g.handle), ctx.span);
        // A handle names the resource itself; there is nothing to load from it,
        // and Naga wants the `GlobalVariable` expression passed straight to the
        // image builtins.
        let slot = if super::texture::is_handle(ctx, g.ty) {
            Slot::Value(expr)
        } else {
            Slot::Ptr(expr)
        };
        env.push_in(g.name.clone(), slot, g.ty, g.writable, g.space);
    }
}

pub(super) fn lower_static(ctx: &mut Context, item: ItemStatic) -> Result<(), Error> {
    let name = item.ident.to_string();
    ctx.pending_space = None;
    let ty = ctx.lower_type(&item.ty)?;
    let from_type = ctx.pending_space.take();
    // A buffer's contents are the host's too. `usize` indexes well enough in
    // a shader, but the host lays it out wider than the GPU does.
    if matches!(
        from_type,
        Some(naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. })
    ) {
        if let Some(int) = ctx.scope.pointer_sized(&item.ty) {
            return Err(Error::PointerSized(name, int, fixed_width(int)));
        }
    }
    let from_init = initializer_binding(ctx, &name, &item.expr)?;
    insert_global(ctx, name, ty, &item.attrs, from_type, from_init)
}

/// The 32-bit integer a shared `usize` or `isize` should be.
pub(super) fn fixed_width(int: &str) -> &'static str {
    match int {
        "isize" => "i32",
        _ => "u32",
    }
}

/// Where a static's initialiser says it binds.
///
/// `group(G).binding(B)` says, as WGSL's `@group(G) @binding(B)` does.
/// `binding()` leaves it to the host, or to nothing for the shader's own
/// memory, and so does `()`, the placeholder of the spelling that says it
/// with attributes. The numbers are integer literals or `const`s, so a host
/// can share them.
fn initializer_binding(
    ctx: &mut Context,
    name: &str,
    init: &Expr,
) -> Result<Option<ResourceBinding>, Error> {
    let unsupported = || Error::UnsupportedInitializer(name.into());
    match strip_parens(init) {
        Expr::Tuple(unit) if unit.elems.is_empty() => Ok(None),
        Expr::Call(call) if call.args.is_empty() && is_named(&call.func, "binding") => Ok(None),
        Expr::MethodCall(call) if call.method == "binding" && call.args.len() == 1 => {
            let Expr::Call(group) = strip_parens(&call.receiver) else {
                return Err(unsupported());
            };
            if group.args.len() != 1 || !is_named(&group.func, "group") {
                return Err(unsupported());
            }
            Ok(Some(ResourceBinding {
                group: binding_number(ctx, &group.args[0])?,
                binding: binding_number(ctx, &call.args[0])?,
            }))
        }
        _ => Err(unsupported()),
    }
}

/// Does `func` name `name`, as `binding` and `synaga_shader::binding` do?
fn is_named(func: &Expr, name: &str) -> bool {
    match func {
        Expr::Path(path) if path.qself.is_none() => path
            .path
            .segments
            .last()
            .is_some_and(|seg| seg.ident == name),
        _ => false,
    }
}

fn binding_number(ctx: &mut Context, expr: &Expr) -> Result<u32, Error> {
    ctx.const_u32(expr)?.ok_or_else(|| {
        let text = quote::ToTokens::to_token_stream(expr).to_string();
        Error::UnsupportedBindingNumber(text.replace(" :: ", "::"))
    })
}

pub(super) fn lower_foreign_mod(ctx: &mut Context, item: ItemForeignMod) -> Result<(), Error> {
    for foreign in item.items {
        match foreign {
            ForeignItem::Static(st) => {
                let name = st.ident.to_string();
                ctx.pending_space = None;
                let ty = ctx.lower_type(&st.ty)?;
                let from_type = ctx.pending_space.take();
                insert_global(ctx, name, ty, &st.attrs, from_type, None)?;
            }
            other => {
                return Err(Error::UnsupportedItem(foreign_kind(&other)));
            }
        }
    }
    Ok(())
}

fn insert_global(
    ctx: &mut Context,
    name: String,
    ty: Handle<Type>,
    attrs: &[Attribute],
    from_type: Option<AddressSpace>,
    from_init: Option<ResourceBinding>,
) -> Result<(), Error> {
    let info = parse_resource_attrs(attrs)?;
    // Both or neither: a host that assigns bindings itself (Blade matches
    // globals up by name at pipeline creation) wants them left unset, but half
    // a binding is a typo.
    let from_attrs = match (info.group, info.binding) {
        (Some(group), Some(binding)) => Some(ResourceBinding { group, binding }),
        (None, None) => None,
        _ => return Err(Error::HalfBinding(name)),
    };
    let binding = match (from_attrs, from_init) {
        (Some(_), Some(_)) => return Err(Error::BindingTwice(name)),
        (from_attrs, from_init) => from_attrs.or(from_init),
    };

    if ctx.globals.iter().any(|g| g.name == name) {
        return Err(Error::DuplicateGlobal(name));
    }

    // A texture or sampler is a handle, not a buffer: it has no address space
    // to choose and is never written through an assignment.
    if super::texture::is_handle(ctx, ty) {
        if info.space.is_some() {
            return Err(Error::UnexpectedAddressSpace(name));
        }
        return finish_global(ctx, name, ty, AddressSpace::Handle, false, binding);
    }

    // `Uniform<T>` and its siblings say the space in the type. An attribute
    // says the same thing the older way; saying both is a contradiction
    // waiting to happen.
    if let Some(space) = from_type {
        if info.space.is_some() {
            return Err(Error::DuplicateAttribute("address space".into()));
        }
        let writable = match space {
            AddressSpace::Storage { access } => access.contains(StorageAccess::STORE),
            AddressSpace::WorkGroup | AddressSpace::Private => true,
            _ => false,
        };
        if !matches!(space, AddressSpace::Storage { .. }) && has_runtime_array(ctx, ty) {
            return Err(Error::RuntimeArrayNotStorage(name));
        }
        return finish_global(ctx, name, ty, space, writable, binding);
    }

    let (space, writable) = match info.space.unwrap_or(SpaceKind::Uniform) {
        SpaceKind::Uniform => (AddressSpace::Uniform, false),
        SpaceKind::Storage { write } => {
            let access = if write {
                StorageAccess::LOAD | StorageAccess::STORE
            } else {
                StorageAccess::LOAD
            };
            (AddressSpace::Storage { access }, write)
        }
        SpaceKind::Workgroup => (AddressSpace::WorkGroup, true),
        SpaceKind::Private => (AddressSpace::Private, true),
    };

    // WGSL puts runtime-sized arrays in storage only. Naga notices too, but as
    // an alignment complaint about a stride nobody wrote.
    if !matches!(space, AddressSpace::Storage { .. }) && has_runtime_array(ctx, ty) {
        return Err(Error::RuntimeArrayNotStorage(name));
    }

    finish_global(ctx, name, ty, space, writable, binding)
}

fn finish_global(
    ctx: &mut Context,
    name: String,
    ty: Handle<Type>,
    space: AddressSpace,
    writable: bool,
    binding: Option<ResourceBinding>,
) -> Result<(), Error> {
    // Only resources are bound; workgroup and private memory belongs to the
    // shader itself. A build that said who assigns bindings holds each
    // resource to it here, where the error can name the line: Naga would
    // notice a missing binding too, but only as a complaint about the module.
    let resource = matches!(
        space,
        AddressSpace::Uniform | AddressSpace::Storage { .. } | AddressSpace::Handle
    );
    match (binding, ctx.bindings) {
        (Some(_), _) if !resource => return Err(Error::UnexpectedBinding(name)),
        (None, Some(Bindings::Explicit)) if resource => {
            return Err(Error::MissingResourceBinding(name))
        }
        (Some(_), Some(Bindings::Host)) => return Err(Error::HostAssignedBinding(name)),
        _ => {}
    }

    if matches!(space, AddressSpace::Uniform | AddressSpace::Storage { .. }) {
        super::structure::check_shared(ctx, ty)?;
    }
    let handle = ctx.module.global_variables.append(
        GlobalVariable {
            name: Some(name.clone()),
            space,
            binding,
            ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        ctx.span,
    );
    ctx.globals.push(GlobalInfo {
        name,
        handle,
        ty,
        writable,
        space,
    });
    Ok(())
}

/// Is `ty` a runtime-sized array, or a struct ending in one?
fn has_runtime_array(ctx: &Context, ty: Handle<Type>) -> bool {
    if matches!(ctx.as_array(ty), Some((_, naga::ArraySize::Dynamic))) {
        return true;
    }
    match ctx.as_struct(ty).and_then(|members| members.last()) {
        Some(last) => has_runtime_array(ctx, last.ty),
        None => false,
    }
}

fn parse_resource_attrs(attrs: &[Attribute]) -> Result<ResourceInfo, Error> {
    let mut info = ResourceInfo {
        group: None,
        binding: None,
        space: None,
    };
    for attr in attrs {
        if attr.path().is_ident("group") {
            if info.group.is_some() {
                return Err(Error::DuplicateAttribute("group".into()));
            }
            info.group = Some(parse_u32_arg(attr, "group")?);
        } else if attr.path().is_ident("binding") {
            if info.binding.is_some() {
                return Err(Error::DuplicateAttribute("binding".into()));
            }
            info.binding = Some(parse_u32_arg(attr, "binding")?);
        } else if attr.path().is_ident("uniform") {
            set_space(&mut info.space, SpaceKind::Uniform)?;
        } else if attr.path().is_ident("storage") {
            let write = parse_storage_write(attr)?;
            set_space(&mut info.space, SpaceKind::Storage { write })?;
        } else if attr.path().is_ident("workgroup") {
            set_space(&mut info.space, SpaceKind::Workgroup)?;
        } else if attr.path().is_ident("private") {
            set_space(&mut info.space, SpaceKind::Private)?;
        }
    }
    Ok(info)
}

fn set_space(slot: &mut Option<SpaceKind>, space: SpaceKind) -> Result<(), Error> {
    if slot.is_some() {
        return Err(Error::DuplicateAttribute("address space".into()));
    }
    *slot = Some(space);
    Ok(())
}

fn parse_u32_arg(attr: &Attribute, what: &str) -> Result<u32, Error> {
    let lit: LitInt = attr
        .parse_args()
        .map_err(|e| Error::UnsupportedBinding(e.to_string()))?;
    lit.base10_parse()
        .map_err(|_| Error::UnsupportedBinding(what.into()))
}

fn parse_storage_write(attr: &Attribute) -> Result<bool, Error> {
    if attr.meta.require_path_only().is_ok() {
        return Ok(false);
    }
    let ident: syn::Ident = attr
        .parse_args()
        .map_err(|e| Error::UnsupportedBinding(e.to_string()))?;
    match ident.to_string().as_str() {
        "read" => Ok(false),
        "read_write" | "write" => Ok(true),
        other => Err(Error::UnsupportedBinding(other.into())),
    }
}

fn foreign_kind(item: &ForeignItem) -> String {
    match item {
        ForeignItem::Fn(_) => "extern fn",
        ForeignItem::Static(_) => "extern static",
        ForeignItem::Type(_) => "extern type",
        ForeignItem::Macro(_) => "extern macro",
        _ => "extern item",
    }
    .into()
}
