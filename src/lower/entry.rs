use naga::{
    Binding, BuiltIn, EntryPoint, Function, FunctionResult, Handle, Interpolation, Sampling,
    ScalarKind, ShaderStage, Type,
};
use syn::{Attribute, FnArg, ItemFn, LitInt, Meta, ReturnType};

use super::env::Env;
use super::{is_unit, lower_signature, Context};
use crate::build::Bindings;
use crate::Error;

#[derive(Default)]
pub(super) struct StageInfo {
    pub stage: Option<ShaderStage>,
    pub workgroup_size: Option<[u32; 3]>,
}

pub(super) fn parse_fn_attrs(attrs: &[Attribute]) -> Result<StageInfo, Error> {
    let mut info = StageInfo::default();
    for attr in attrs {
        if attr.path().is_ident("entry_point") {
            if info.stage.is_some() {
                return Err(Error::DuplicateAttribute("entry_point".into()));
            }
            parse_entry_point(attr, &mut info)?;
        } else if attr.path().is_ident("output") {
            // A bare return value is the stage's one default output, and
            // anything else is a struct whose fields say where they go.
            return Err(Error::OutputAttribute);
        } else if let Some(old) = ["vertex", "fragment", "compute", "workgroup_size"]
            .into_iter()
            .find(|old| attr.path().is_ident(old))
        {
            // Left alone, this would quietly turn the entry point into a helper.
            return Err(Error::OldStageAttribute(old.into()));
        }
    }
    Ok(info)
}

/// `#[entry_point(vertex)]`, `#[entry_point(fragment)]`, or
/// `#[entry_point(compute, threads(x, y, z))]`.
fn parse_entry_point(attr: &Attribute, info: &mut StageInfo) -> Result<(), Error> {
    let args = attr
        .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        .map_err(|e| Error::UnsupportedBinding(e.to_string()))?;
    let mut args = args.iter();
    info.stage = Some(match args.next() {
        Some(Meta::Path(path)) if path.is_ident("vertex") => ShaderStage::Vertex,
        Some(Meta::Path(path)) if path.is_ident("fragment") => ShaderStage::Fragment,
        Some(Meta::Path(path)) if path.is_ident("compute") => ShaderStage::Compute,
        _ => return Err(Error::UnknownStage),
    });
    for arg in args {
        match arg {
            Meta::List(list) if list.path.is_ident("threads") => {
                if info.workgroup_size.is_some() {
                    return Err(Error::DuplicateAttribute("threads".into()));
                }
                info.workgroup_size = Some(parse_threads(list)?);
            }
            _ => return Err(Error::UnsupportedBinding("entry_point argument".into())),
        }
    }
    Ok(())
}

fn parse_threads(list: &syn::MetaList) -> Result<[u32; 3], Error> {
    let lits = list
        .parse_args_with(syn::punctuated::Punctuated::<LitInt, syn::Token![,]>::parse_terminated)
        .map_err(|e| Error::UnsupportedBinding(e.to_string()))?;
    if lits.is_empty() || lits.len() > 3 {
        return Err(Error::UnsupportedBinding("threads".into()));
    }
    let mut size = [1u32, 1, 1];
    for (i, lit) in lits.iter().enumerate() {
        size[i] = lit
            .base10_parse()
            .map_err(|_| Error::UnsupportedBinding("threads".into()))?;
        if size[i] == 0 {
            return Err(Error::UnsupportedBinding("threads(0)".into()));
        }
    }
    Ok(size)
}

/// Parse `#[location(N)]` / `#[builtin(name)]`, plus an optional `#[flat]`.
///
/// Used for both entry-point arguments and the fields of an I/O struct.
pub(super) fn parse_io_binding(attrs: &[Attribute]) -> Result<Option<Binding>, Error> {
    let mut found = None;
    let mut flat = false;
    for attr in attrs {
        if attr.path().is_ident("builtin") || attr.path().is_ident("location") {
            if found.is_some() {
                return Err(Error::DuplicateAttribute("location/builtin".into()));
            }
            found = Some(parse_plain_binding(attr)?);
        } else if attr.path().is_ident("flat") {
            flat = true;
        }
    }
    if flat {
        match &mut found {
            Some(Binding::Location { interpolation, .. }) => {
                *interpolation = Some(Interpolation::Flat)
            }
            _ => return Err(Error::UnsupportedBinding("flat".into())),
        }
    }
    Ok(found)
}

fn parse_plain_binding(attr: &Attribute) -> Result<Binding, Error> {
    if attr.path().is_ident("builtin") {
        let ident: syn::Ident = attr
            .parse_args()
            .map_err(|e| Error::UnsupportedBinding(e.to_string()))?;
        Ok(Binding::BuiltIn(map_builtin(&ident.to_string())?))
    } else if attr.path().is_ident("location") {
        let lit: LitInt = attr
            .parse_args()
            .map_err(|e| Error::UnsupportedBinding(e.to_string()))?;
        let location = lit
            .base10_parse()
            .map_err(|_| Error::UnsupportedBinding("location".into()))?;
        Ok(Binding::Location {
            location,
            interpolation: None,
            sampling: None,
            blend_src: None,
            per_primitive: false,
        })
    } else {
        Err(Error::UnsupportedBinding("binding".into()))
    }
}

fn map_builtin(name: &str) -> Result<BuiltIn, Error> {
    Ok(match name {
        "position" => BuiltIn::Position { invariant: false },
        "vertex_index" => BuiltIn::VertexIndex,
        "instance_index" => BuiltIn::InstanceIndex,
        "global_invocation_id" => BuiltIn::GlobalInvocationId,
        "local_invocation_id" => BuiltIn::LocalInvocationId,
        "local_invocation_index" => BuiltIn::LocalInvocationIndex,
        "workgroup_id" => BuiltIn::WorkGroupId,
        "workgroup_size" => BuiltIn::WorkGroupSize,
        "num_workgroups" => BuiltIn::NumWorkGroups,
        "front_facing" => BuiltIn::FrontFacing,
        "frag_depth" => BuiltIn::FragDepth,
        other => return Err(Error::UnsupportedBinding(other.into())),
    })
}

/// The builtin a parameter is by its name alone: `global_invocation_id:
/// Vec3<u32>` needs no `#[builtin]`. Only what the stage takes as input
/// counts, so a vertex shader's `position` is still an attribute that has to
/// say its `#[location]`.
fn input_builtin(stage: ShaderStage, name: &str) -> Option<BuiltIn> {
    use BuiltIn as B;
    let builtin = map_builtin(name).ok()?;
    let takes = match stage {
        ShaderStage::Vertex => matches!(builtin, B::VertexIndex | B::InstanceIndex),
        ShaderStage::Fragment => matches!(builtin, B::Position { .. } | B::FrontFacing),
        ShaderStage::Compute => matches!(
            builtin,
            B::GlobalInvocationId
                | B::LocalInvocationId
                | B::LocalInvocationIndex
                | B::WorkGroupId
                | B::NumWorkGroups
        ),
        _ => false,
    };
    takes.then_some(builtin)
}

/// Where a bare value an entry point returns goes: a vertex shader's is its
/// position, the one thing it has to produce, and a fragment shader's is its
/// first colour target. Anything else is returned in a struct whose fields
/// say where they go, so a function never carries its result's binding.
fn default_output(stage: ShaderStage) -> Option<Binding> {
    match stage {
        ShaderStage::Vertex => Some(Binding::BuiltIn(BuiltIn::Position { invariant: false })),
        ShaderStage::Fragment => Some(Binding::Location {
            location: 0,
            interpolation: None,
            sampling: None,
            blend_src: None,
            per_primitive: false,
        }),
        _ => None,
    }
}

/// Only vertex outputs and fragment inputs are interpolated, so only they need
/// an interpolation mode.
fn needs_interpolation(stage: ShaderStage, is_input: bool) -> bool {
    matches!(
        (stage, is_input),
        (ShaderStage::Vertex, false) | (ShaderStage::Fragment, true)
    )
}

/// Give a float `Location` binding the default every shading language shares:
/// perspective-correct, center-sampled.
///
/// Applied to every location binding, not just the interpolated ones, because
/// that is what Naga's own WGSL frontend does — and because the backend prints
/// nothing for the default, so it stays invisible where it does not apply.
/// Integers cannot be interpolated at all, so they are left alone and
/// `check_interpolation` asks for `#[flat]` where one is required.
pub(super) fn apply_default_interpolation(ctx: &Context, ty: Handle<Type>, binding: &mut Binding) {
    let Binding::Location {
        interpolation: interpolation @ None,
        sampling,
        ..
    } = binding
    else {
        return;
    };
    if ctx.shape(ty).scalar().map(|s| s.kind) == Some(ScalarKind::Float) {
        *interpolation = Some(Interpolation::Perspective);
        *sampling = Some(Sampling::Center);
    }
}

/// An integer varying has to say `#[flat]`; nothing else can be meant, but WGSL
/// still wants it written down.
fn check_interpolation(
    binding: &Binding,
    name: &str,
    stage: ShaderStage,
    is_input: bool,
) -> Result<(), Error> {
    if !needs_interpolation(stage, is_input) {
        return Ok(());
    }
    if let Binding::Location {
        interpolation: None,
        ..
    } = binding
    {
        return Err(Error::MissingFlat(name.into()));
    }
    Ok(())
}

/// Does this type carry its own per-field bindings, as a vertex-output or
/// fragment-input struct does?
fn is_io_struct(ctx: &Context, ty: naga::Handle<naga::Type>) -> bool {
    match ctx.as_struct(ty) {
        Some(members) => members.iter().all(|m| m.binding.is_some()),
        None => false,
    }
}

/// Float fields get the default interpolation when the struct is declared;
/// integers cannot be interpolated at all, so they have to say `#[flat]`.
fn check_io_struct(
    ctx: &Context,
    ty: Handle<Type>,
    stage: ShaderStage,
    is_input: bool,
) -> Result<(), Error> {
    for member in ctx.as_struct(ty).into_iter().flatten() {
        if let Some(binding) = &member.binding {
            let name = member.name.clone().unwrap_or_default();
            check_interpolation(binding, &name, stage, is_input)?;
        }
    }
    Ok(())
}

pub(super) fn lower_entry(ctx: &mut Context, item: ItemFn, info: StageInfo) -> Result<(), Error> {
    let stage = info.stage.expect("stage present");
    let name = item.sig.ident.to_string();
    ctx.claim_entry_point_name(&name)?;

    if stage != ShaderStage::Compute && info.workgroup_size.is_some() {
        return Err(Error::UnexpectedWorkgroupSize);
    }
    let workgroup_size = if stage == ShaderStage::Compute {
        info.workgroup_size.ok_or(Error::MissingWorkgroupSize)?
    } else {
        [0, 0, 0]
    };

    let result = match &item.sig.output {
        // A fragment may write only depth, so it has nothing to return.
        // A vertex stage still has to produce a position.
        ReturnType::Default => {
            if stage == ShaderStage::Vertex {
                return Err(Error::MissingReturnType(name));
            }
            None
        }
        ReturnType::Type(_, ty) if is_unit(ty) => {
            if stage == ShaderStage::Vertex {
                return Err(Error::MissingReturnType(name));
            }
            None
        }
        ReturnType::Type(_, ty) => {
            let result_ty = ctx.lower_type(ty)?;
            if is_io_struct(ctx, result_ty) {
                check_io_struct(ctx, result_ty, stage, false)?;
                Some(FunctionResult {
                    ty: result_ty,
                    binding: None,
                })
            } else {
                let mut binding = default_output(stage)
                    .ok_or_else(|| Error::ComputeReturnsValue(name.clone()))?;
                apply_default_interpolation(ctx, result_ty, &mut binding);
                check_interpolation(&binding, &name, stage, false)?;
                Some(FunctionResult {
                    ty: result_ty,
                    binding: Some(binding),
                })
            }
        }
    };

    let mut function = Function {
        name: Some(name),
        arguments: Vec::new(),
        result,
        ..Default::default()
    };

    let mut env = Env::default();
    super::global::bind_globals(ctx, &mut function, &mut env);
    lower_signature(ctx, &mut function, &item.sig, &mut env)?;

    for (arg, fn_arg) in function.arguments.iter_mut().zip(item.sig.inputs.iter()) {
        let FnArg::Typed(pat_ty) = fn_arg else {
            return Err(Error::Receiver);
        };
        match parse_io_binding(&pat_ty.attrs)? {
            Some(mut binding) => {
                apply_default_interpolation(ctx, arg.ty, &mut binding);
                let name = arg.name.clone().unwrap_or_default();
                check_interpolation(&binding, &name, stage, true)?;
                arg.binding = Some(binding);
            }
            // A struct argument carries its interface on its fields — either
            // written there, or left for the host to fill in (Blade matches
            // vertex attributes up by field name). Half-bound structs are
            // already refused where they are declared.
            None if ctx.as_struct(arg.ty).is_some() => {
                if is_io_struct(ctx, arg.ty) {
                    check_io_struct(ctx, arg.ty, stage, true)?;
                } else if ctx.bindings == Some(Bindings::Explicit) {
                    return Err(Error::UnboundStructArg(
                        arg.name.clone().unwrap_or_default(),
                    ));
                }
            }
            None => {
                let name = arg.name.clone().unwrap_or_default();
                let builtin = input_builtin(stage, &name).ok_or(Error::MissingArgBinding(name))?;
                arg.binding = Some(Binding::BuiltIn(builtin));
            }
        }
    }

    let mut body = naga::Block::new();
    env.push_scope();
    ctx.addressed = super::stmt::addressed_names(&item.block, &ctx.scope.mut_self_methods);
    super::stmt::lower_body(ctx, &mut function, &mut body, &item.block, &mut env)?;
    env.pop_scope();
    function.body = body;

    ctx.module.entry_points.push(EntryPoint {
        name: function.name.clone().unwrap_or_default(),
        stage,
        early_depth_test: None,
        workgroup_size,
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });
    Ok(())
}
