//! Texture and sampler types, and the builtins that work on them.
//!
//! Spelled as WGSL spells them — `texture_2d<f32>`, `texture_storage_2d<Rgba8Unorm, Write>`,
//! `sampler` — so a WGSL shader ports across without renaming. The builtins take
//! either their WGSL name or a snake_case one (`textureLoad` / `texture_load`).

use naga::{
    Block, Expression, Function, Handle, ImageClass, ImageDimension, ImageQuery, SampleLevel,
    Scalar, ScalarKind, Span, Statement, StorageAccess, StorageFormat, Type, TypeInner, VectorSize,
};
use syn::Expr;

use super::emit::emit;
use super::env::Env;
use super::expr::lower_expr;
use super::{Context, Typed};
use crate::Error;

/// `texture_2d<f32>` and friends, parsed from the type name plus its generic
/// arguments. Returns `None` for anything that is not an image or sampler.
pub(super) fn parse_handle_type(
    ctx: &mut Context,
    name: &str,
    args: &[&syn::Type],
) -> Option<Result<Handle<Type>, Error>> {
    if name == "sampler" || name == "sampler_comparison" {
        if !args.is_empty() {
            return Some(Err(Error::UnsupportedType(name.into())));
        }
        return Some(Ok(ctx.intern_handle_type(TypeInner::Sampler {
            comparison: name == "sampler_comparison",
        })));
    }

    let rest = name.strip_prefix("texture_")?;
    let (rest, arrayed) = match rest.strip_suffix("_array") {
        Some(head) => (head, true),
        None => (rest, false),
    };

    // Storage textures carry their format and access mode as type arguments.
    if let Some(dim) = rest.strip_prefix("storage_") {
        let Some(dim) = parse_dim(dim) else {
            return Some(Err(Error::UnsupportedType(name.into())));
        };
        let [format, access] = args else {
            return Some(Err(Error::UnsupportedType(format!(
                "{name} needs a format and an access mode"
            ))));
        };
        let class = match (parse_format(format), parse_access(access)) {
            (Some(format), Some(access)) => ImageClass::Storage { format, access },
            _ => return Some(Err(Error::UnsupportedType(name.into()))),
        };
        return Some(Ok(ctx.intern_handle_type(TypeInner::Image {
            dim,
            arrayed,
            class,
        })));
    }

    // Depth textures take no type argument; sampled ones take their component.
    if let Some(dim) = rest.strip_prefix("depth_multisampled_") {
        return Some(image(
            ctx,
            dim,
            arrayed,
            args,
            ImageClass::Depth { multi: true },
        ));
    }
    if let Some(dim) = rest.strip_prefix("depth_") {
        return Some(image(
            ctx,
            dim,
            arrayed,
            args,
            ImageClass::Depth { multi: false },
        ));
    }
    let (dim, multi) = match rest.strip_prefix("multisampled_") {
        Some(dim) => (dim, true),
        None => (rest, false),
    };
    parse_dim(dim)?;
    let [component] = args else {
        return Some(Err(Error::UnsupportedType(format!(
            "{name} needs a component type"
        ))));
    };
    let kind = match component_kind(component) {
        Some(kind) => kind,
        None => return Some(Err(Error::UnsupportedType(name.into()))),
    };
    Some(image(
        ctx,
        dim,
        arrayed,
        &[],
        ImageClass::Sampled { kind, multi },
    ))
}

fn image(
    ctx: &mut Context,
    dim: &str,
    arrayed: bool,
    extra_args: &[&syn::Type],
    class: ImageClass,
) -> Result<Handle<Type>, Error> {
    if !extra_args.is_empty() {
        return Err(Error::UnsupportedType("texture type arguments".into()));
    }
    let dim = parse_dim(dim).ok_or_else(|| Error::UnsupportedType(format!("texture_{dim}")))?;
    Ok(ctx.intern_handle_type(TypeInner::Image {
        dim,
        arrayed,
        class,
    }))
}

fn parse_dim(name: &str) -> Option<ImageDimension> {
    match name {
        "1d" => Some(ImageDimension::D1),
        "2d" => Some(ImageDimension::D2),
        "3d" => Some(ImageDimension::D3),
        "cube" => Some(ImageDimension::Cube),
        _ => None,
    }
}

fn type_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) if path.qself.is_none() => {
            path.path.get_ident().map(|i| i.to_string())
        }
        _ => None,
    }
}

fn component_kind(ty: &syn::Type) -> Option<ScalarKind> {
    match type_ident(ty)?.as_str() {
        "f32" => Some(ScalarKind::Float),
        "i32" => Some(ScalarKind::Sint),
        "u32" => Some(ScalarKind::Uint),
        _ => None,
    }
}

fn parse_access(ty: &syn::Type) -> Option<StorageAccess> {
    match type_ident(ty)?.as_str() {
        "Read" | "read" => Some(StorageAccess::LOAD),
        "Write" | "write" => Some(StorageAccess::STORE),
        "ReadWrite" | "read_write" => Some(StorageAccess::LOAD | StorageAccess::STORE),
        _ => None,
    }
}

/// WGSL's format names, accepted as written or in Rust's `CamelCase`.
fn parse_format(ty: &syn::Type) -> Option<StorageFormat> {
    use StorageFormat as Sf;
    let name = type_ident(ty)?.to_ascii_lowercase();
    Some(match name.as_str() {
        "r8unorm" => Sf::R8Unorm,
        "r8snorm" => Sf::R8Snorm,
        "r8uint" => Sf::R8Uint,
        "r8sint" => Sf::R8Sint,
        "r16uint" => Sf::R16Uint,
        "r16sint" => Sf::R16Sint,
        "r16float" => Sf::R16Float,
        "rg8unorm" => Sf::Rg8Unorm,
        "rg8snorm" => Sf::Rg8Snorm,
        "rg8uint" => Sf::Rg8Uint,
        "rg8sint" => Sf::Rg8Sint,
        "r32uint" => Sf::R32Uint,
        "r32sint" => Sf::R32Sint,
        "r32float" => Sf::R32Float,
        "rg16uint" => Sf::Rg16Uint,
        "rg16sint" => Sf::Rg16Sint,
        "rg16float" => Sf::Rg16Float,
        "rgba8unorm" => Sf::Rgba8Unorm,
        "rgba8snorm" => Sf::Rgba8Snorm,
        "rgba8uint" => Sf::Rgba8Uint,
        "rgba8sint" => Sf::Rgba8Sint,
        "rgb10a2unorm" => Sf::Rgb10a2Unorm,
        "rg11b10float" | "rg11b10ufloat" => Sf::Rg11b10Ufloat,
        "rg32uint" => Sf::Rg32Uint,
        "rg32sint" => Sf::Rg32Sint,
        "rg32float" => Sf::Rg32Float,
        "rgba16uint" => Sf::Rgba16Uint,
        "rgba16sint" => Sf::Rgba16Sint,
        "rgba16float" => Sf::Rgba16Float,
        "rgba32uint" => Sf::Rgba32Uint,
        "rgba32sint" => Sf::Rgba32Sint,
        "rgba32float" => Sf::Rgba32Float,
        _ => return None,
    })
}

/// Images, samplers and acceleration structures live in the handle address
/// space: they name a resource rather than memory, so there is no space to
/// choose and nothing to load through a pointer.
pub(super) fn is_handle(ctx: &Context, ty: Handle<Type>) -> bool {
    matches!(
        ctx.module.types[ty].inner,
        TypeInner::Image { .. }
            | TypeInner::Sampler { .. }
            | TypeInner::AccelerationStructure { .. }
    ) || matches!(
        ctx.module.types[ty].inner,
        // A binding array of textures is a handle; one of buffers is storage,
        // and takes an address space like any other buffer.
        TypeInner::BindingArray { base, .. } if is_handle(ctx, base)
    )
}

/// Which texture builtin `name` is, under either spelling.
pub(super) fn texture_builtin(name: &str) -> Option<TextureOp> {
    Some(match name {
        "textureSample" | "texture_sample" => TextureOp::Sample,
        "textureSampleLevel" | "texture_sample_level" => TextureOp::SampleLevel,
        "textureSampleCompare" | "texture_sample_compare" => TextureOp::SampleCompare,
        "textureSampleCompareLevel" | "texture_sample_compare_level" => {
            TextureOp::SampleCompareLevel
        }
        "textureLoad" | "texture_load" => TextureOp::Load,
        // The storage form is the same operation; Rust just cannot give one
        // name two arities, so a checkable shader spells it apart.
        "textureLoadStorage" | "texture_load_storage" => TextureOp::Load,
        "textureStore" | "texture_store" => TextureOp::Store,
        "textureDimensions"
        | "texture_dimensions"
        | "textureDimensionsLevel"
        | "texture_dimensions_level" => TextureOp::Dimensions,
        "textureNumLevels" | "texture_num_levels" => TextureOp::NumLevels,
        "textureNumLayers" | "texture_num_layers" => TextureOp::NumLayers,
        "textureNumSamples" | "texture_num_samples" => TextureOp::NumSamples,
        _ => return None,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TextureOp {
    Sample,
    SampleLevel,
    SampleCompare,
    SampleCompareLevel,
    Load,
    Store,
    Dimensions,
    NumLevels,
    NumLayers,
    NumSamples,
}

impl TextureOp {
    /// `textureStore` writes rather than produces; it is a statement.
    pub(super) fn is_statement(self) -> bool {
        self == TextureOp::Store
    }
}

struct ImageInfo {
    dim: ImageDimension,
    arrayed: bool,
    class: ImageClass,
}

fn image_info(ctx: &Context, ty: Handle<Type>, name: &str) -> Result<ImageInfo, Error> {
    match ctx.module.types[ty].inner {
        TypeInner::Image {
            dim,
            arrayed,
            class,
        } => Ok(ImageInfo {
            dim,
            arrayed,
            class,
        }),
        _ => Err(Error::NotATexture(name.into())),
    }
}

/// Number of coordinate components a `dim` image indexes with.
fn coord_size(dim: ImageDimension) -> Option<VectorSize> {
    match dim {
        ImageDimension::D1 => None,
        ImageDimension::D2 => Some(VectorSize::Bi),
        ImageDimension::D3 | ImageDimension::Cube => Some(VectorSize::Tri),
    }
}

pub(super) fn lower_texture_call(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    call: &syn::ExprCall,
    env: &mut Env,
    name: &str,
    op: TextureOp,
) -> Result<Typed, Error> {
    let mut args = call.args.iter();
    let image_arg = args
        .next()
        .ok_or_else(|| Error::WrongArgCount(name.into()))?;
    let (image, image_ty) = lower_expr(ctx, function, body, image_arg, env)?;
    let info = image_info(ctx, image_ty, name)?;

    match op {
        TextureOp::Dimensions
        | TextureOp::NumLevels
        | TextureOp::NumLayers
        | TextureOp::NumSamples => {
            let query = match op {
                TextureOp::Dimensions => {
                    let level = match args.next() {
                        Some(level) => Some(lower_expr(ctx, function, body, level, env)?.0),
                        None => None,
                    };
                    ImageQuery::Size { level }
                }
                TextureOp::NumLevels => ImageQuery::NumLevels,
                TextureOp::NumLayers => ImageQuery::NumLayers,
                _ => ImageQuery::NumSamples,
            };
            expect_end(&mut args, name)?;
            let ty = match (op, coord_size(info.dim)) {
                (TextureOp::Dimensions, Some(size)) => ctx.intern_vector(size, Scalar::U32),
                _ => ctx.intern_scalar(Scalar::U32),
            };
            let handle = emit(function, body, Expression::ImageQuery { image, query })?;
            Ok((handle, ty))
        }
        TextureOp::Load => {
            let coordinate = next_arg(ctx, function, body, &mut args, env, name)?;
            let array_index = if info.arrayed {
                Some(next_arg(ctx, function, body, &mut args, env, name)?)
            } else {
                None
            };
            // Sampled and depth images take a mip level; multisampled ones take
            // a sample index; storage images take neither.
            let (level, sample) = match info.class {
                ImageClass::Storage { .. } => (None, None),
                ImageClass::Sampled { multi: true, .. } | ImageClass::Depth { multi: true } => (
                    None,
                    Some(next_arg(ctx, function, body, &mut args, env, name)?),
                ),
                _ => (
                    Some(next_arg(ctx, function, body, &mut args, env, name)?),
                    None,
                ),
            };
            expect_end(&mut args, name)?;
            let ty = load_result_ty(ctx, &info);
            let handle = emit(
                function,
                body,
                Expression::ImageLoad {
                    image,
                    coordinate,
                    array_index,
                    sample,
                    level,
                },
            )?;
            Ok((handle, ty))
        }
        TextureOp::Store => {
            let coordinate = next_arg(ctx, function, body, &mut args, env, name)?;
            let array_index = if info.arrayed {
                Some(next_arg(ctx, function, body, &mut args, env, name)?)
            } else {
                None
            };
            let value = next_arg(ctx, function, body, &mut args, env, name)?;
            expect_end(&mut args, name)?;
            body.push(
                Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                },
                Span::UNDEFINED,
            );
            // Nothing to hand back; `lower_stmt_expr` keeps this out of value
            // position, and the caller here only uses the type on that path.
            Ok((value, ctx.intern_scalar(Scalar::U32)))
        }
        _ => lower_sample(ctx, function, body, &mut args, env, name, op, image, &info),
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_sample<'a>(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    args: &mut impl Iterator<Item = &'a Expr>,
    env: &mut Env,
    name: &str,
    op: TextureOp,
    image: Handle<Expression>,
    info: &ImageInfo,
) -> Result<Typed, Error> {
    let sampler = next_arg(ctx, function, body, args, env, name)?;
    let coordinate = next_arg(ctx, function, body, args, env, name)?;
    let array_index = if info.arrayed {
        Some(next_arg(ctx, function, body, args, env, name)?)
    } else {
        None
    };
    let compare = matches!(op, TextureOp::SampleCompare | TextureOp::SampleCompareLevel);
    let depth_ref = if compare {
        Some(next_arg(ctx, function, body, args, env, name)?)
    } else {
        None
    };
    let level = match op {
        TextureOp::Sample => SampleLevel::Auto,
        TextureOp::SampleLevel => {
            SampleLevel::Exact(next_arg(ctx, function, body, args, env, name)?)
        }
        // A comparison sample is always at the base level; `textureSampleCompare`
        // picks its own in hardware.
        TextureOp::SampleCompare => SampleLevel::Auto,
        _ => SampleLevel::Zero,
    };
    expect_end(args, name)?;

    let ty = if compare {
        ctx.intern_scalar(Scalar::F32)
    } else {
        sample_result_ty(ctx, info)
    };
    let handle = emit(
        function,
        body,
        Expression::ImageSample {
            image,
            sampler,
            gather: None,
            coordinate,
            array_index,
            offset: None,
            level,
            depth_ref,
            clamp_to_edge: false,
        },
    )?;
    Ok((handle, ty))
}

/// Sampling a colour image gives a `vec4` of its component type; a depth image
/// gives a bare `f32`.
fn sample_result_ty(ctx: &mut Context, info: &ImageInfo) -> Handle<Type> {
    match info.class {
        ImageClass::Depth { .. } => ctx.intern_scalar(Scalar::F32),
        ImageClass::Sampled { kind, .. } => ctx.intern_vector(VectorSize::Quad, scalar_of(kind)),
        ImageClass::Storage { format, .. } => ctx.intern_vector(VectorSize::Quad, format.into()),
        // External textures are a video-interop class with their own sampling
        // rules; nothing here builds one.
        ImageClass::External => ctx.intern_vector(VectorSize::Quad, Scalar::F32),
    }
}

fn load_result_ty(ctx: &mut Context, info: &ImageInfo) -> Handle<Type> {
    sample_result_ty(ctx, info)
}

fn scalar_of(kind: ScalarKind) -> Scalar {
    match kind {
        ScalarKind::Sint => Scalar::I32,
        ScalarKind::Uint => Scalar::U32,
        _ => Scalar::F32,
    }
}

fn next_arg<'a>(
    ctx: &mut Context,
    function: &mut Function,
    body: &mut Block,
    args: &mut impl Iterator<Item = &'a Expr>,
    env: &mut Env,
    name: &str,
) -> Result<Handle<Expression>, Error> {
    let arg = args
        .next()
        .ok_or_else(|| Error::WrongArgCount(name.into()))?;
    Ok(lower_expr(ctx, function, body, arg, env)?.0)
}

fn expect_end<'a>(args: &mut impl Iterator<Item = &'a Expr>, name: &str) -> Result<(), Error> {
    match args.next() {
        Some(_) => Err(Error::WrongArgCount(name.into())),
        None => Ok(()),
    }
}
