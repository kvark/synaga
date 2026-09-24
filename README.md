# synaga

Native Rust to [Naga](https://docs.rs/naga) IR transpiler.

Parses a restricted Rust dialect with [`syn`](https://docs.rs/syn) on stable and
builds a `naga::Module` by hand. No `rustc_private`, no nightly.

## v0 dialect

- free functions
- scalars: `f32`, `u32`, `i32`, `bool`
- vectors: `vec2`/`vec3`/`vec4` (default `f32`), `vecN<T>`, `vec2f`/`vec3i`/`vec4u`
- matrices: `mat2`/`mat3`/`mat4` (square `f32`), `matCxR`, `mat4f`, `mat2x3<f32>`
- constructors: `vec3(x, y, z)`, splat `vec3(x)`, mix `vec3(xy, z)`
- matrix constructors: column vectors `mat4(c0,c1,c2,c3)` or column-major scalars
- components: `.x`/`.y`/`.z`/`.w` or `.r`/`.g`/`.b`/`.a`, swizzle `.xy`/`.zyx`,
  index `v[0]` / `v[i]` / `m[0]`
- literals, unary `-`/`!`, arithmetic / compare / bitwise / shift ops (scalar splat on mix)
- casts: `a as f32`, `v as vec3<u32>` (same width, component-wise)
- `let` / `let x: T = …`, and `let x: T;` assigned later (WGSL's bare `var x: T;`)
- `if` / `else` / `else if` as statement or value
- `x = e` and compound `+=`/`-=`/`*=`/`/=`/… on any place: a local, a writable
  global, a field `s.a`, a component `v.x` / `v[i]`, a matrix column `m[0]`
- `loop` / `while` / `for x in a..b` / `a..=b` / `break` / `continue`
- implicit tail expressions and `return`
- entry points: `#[vertex]` / `#[fragment]` / `#[compute]` + `#[workgroup_size(x,y,z)]`
- bindings: `#[location(N)]`, `#[builtin(name)]` on args; `#[output(builtin(..))]` / `#[output(location(N))]` on the fn
- `select(reject, accept, condition)`, in WGSL's argument order
- calls to earlier free functions, including ones that return nothing
- out-parameters: `&mut T` is WGSL's `ptr<function, T>`; `&T` is the same pointer
  with writes refused
- `const NAME: T = …` at module level (literals and vector/matrix constructors)
- math builtins: `dot`, `cross`, `normalize`, `length`, `abs`, `min`, `max`, `clamp`,
  `mix`, `step`, `sin`, `cos`, `pow`, `transpose`, `determinant`, the bit-twiddling
  set (`countOneBits`, `reverseBits`, `extractBits`, …) and the packing set
  (`pack4x8snorm`, `unpack4x8unorm`, …) — each also spelled snake_case
- `all`, `any`, `isNan`, `isInf`; `arrayLength(buf)`; `discard()`;
  `workgroupBarrier()` / `storageBarrier()`
- globals: `#[group(N)] #[binding(M)] static x: T = ();` (init ignored) or `extern { static x: T; }`;
  both attributes may be dropped for a host that assigns bindings itself — see [`validate_unbound`](#host-assigned-bindings)
- address spaces: uniform (default / `#[uniform]`), `#[storage]` (read),
  `#[storage(read_write)]`, `#[workgroup]`, `#[private]`
- atomics: `atomic<u32>` / `atomic<i32>`, `atomicAdd` / `atomicStore` / `atomicLoad` /
  `atomicMax` / … taking the variable directly rather than a reference
- ray queries: `acceleration_structure`, `ray_query`, `RayDesc`, `RayIntersection`,
  `rayQueryInitialize` / `Proceed` / `GetCommittedIntersection` / …, and the
  predeclared `RAY_FLAG_*` and `RAY_QUERY_INTERSECTION_*` names
- `binding_array<T>` and `binding_array<T, N>`
- zero values: `T()`, or `T::default()`, for a struct, vector, matrix or scalar
- structs: `struct S { a: vec3, b: f32 }`, literals `S { a, b: x }`, field access `s.a`
- arrays: `[T; N]` and literals `[a, b, c]`; `[T]` for a runtime-sized storage buffer
- textures and samplers: `texture_2d<f32>`, `texture_storage_2d<Rgba8Unorm, Write>`,
  `texture_depth_2d`, `sampler`, `sampler_comparison`, arrayed and multisampled variants
- texture builtins: `textureSample`, `textureSampleLevel`, `textureSampleCompare`,
  `textureLoad`, `textureStore`, `textureDimensions`, `textureNumLevels`, … — each also
  spelled snake_case (`texture_load`)
- I/O structs: `#[location]` / `#[builtin]` on struct fields, for vertex outputs,
  fragment inputs, and multiple render targets

Not yet: labeled loops, `break` values, `switch`, forward calls, methods, generics,
cooperative matrices, `f16`, `const` arithmetic (Naga wants constants already
folded). Swizzles are values, so `v.xy = a` is rejected — as it is in WGSL.
Assignment to function arguments is rejected. Vector compare yields a `vecN<bool>`.

### Typing

Operand rules follow Naga's, so a program the frontend accepts is a module its
validator accepts — `-x` needs a signed or float operand, `a & b` an integer or
bool one, a shift amount is always `u32`, and so on.

Untyped integer literals take their type from context the way Rust's inference
would: `n << 1`, `n * 2`, `clamp(n, 0, 10)`, `f(1)`, `let n: u32 = 1` and
`vec3u(1, 2, 3)` all work whatever integer type is in play. There is no
`1` to `1.0` conversion, again as in Rust.

A function with a return type has to return on every path; `if c { a }` as a
whole body is rejected rather than quietly falling off the end.

A function you declare shadows a math builtin of the same name.

### Validation

`validate` uses Naga's default flags and no extra capabilities; `validate_unbound`
drops `ValidationFlags::BINDINGS` for a host that assigns them; `validate_with`
takes both, which is what a ray query needs (`Capabilities::RAY_QUERY`) and what
a host validating against a real device wants anyway.

The module is the product. `to_wgsl` is `feature = "wgsl"`, for a client that
wants to read the shader as text. It prints a ray query as the WGSL builtins
(`rayQueryInitialize` and the rest) and adds `enable wgpu_ray_query`. Naga's
own backend has no spelling for the operation and panics, so the module is
rewritten into calls first and the stand-in functions are removed from the text.

### Errors

`validate` and `validate_unbound` return a `ValidationError` that prints Naga's
whole source chain. Naga puts the useful part there: the top level says only
that a global is invalid, and the reason ("the array stride 4 is not a multiple
of the required alignment 16") is one `source()` down.

### Places

`s.a`, `v.x`, `v[i]` and `m[0]` are lowered as pointers rather than as
components picked out of a loaded value. That is what makes them assignable,
and it means reading one field of a uniform buffer loads that field instead of
the whole struct. A function argument is a value, so its fields can be read but
not written.

### Host-assigned bindings

Some engines leave `@group`/`@binding` out of the shader and fill them in at
pipeline creation, matching globals up by name — [Blade][blade] does, and
asserts the module has none. Drop both attributes for that, and validate with
`validate_unbound`, which is `validate` minus `ValidationFlags::BINDINGS`:

```rust
let module = synaga::parse_str(src)?;
let info = synaga::validate_unbound(&module)?;
```

Blade takes a `naga::Module` directly (`ShaderDesc::naga_module`), so a module
built here needs no WGSL round trip to reach it.

[blade]: https://github.com/kvark/blade

### Interpolation

A float `#[location]` binding gets the default every shading language shares —
perspective-correct, center-sampled — on entry-point arguments, results and
struct fields alike. This is the same rule Naga's own WGSL frontend applies, so
a shader ported from WGSL produces the same module as the WGSL did; the backend
prints nothing for the default, so it stays invisible where it does not apply.

Integers cannot be interpolated, so an integer `#[location]` that *is* — a
vertex output or a fragment input — has to say `#[flat]`:

```rust
struct VsOut {
    #[builtin(position)] pos: vec4,
    #[location(0)] uv: vec2,
    #[location(1)] #[flat] material: u32,
}
```

A struct is either plain data or a shader interface: binding some fields and not
others is rejected. An entry point returning a bound struct does not take
`#[output(...)]`, and a struct argument whose fields carry no bindings is left
for the host to fill in — which is how Blade supplies vertex attributes.

### Blade

The dialect was built against [Blade][blade]'s shaders, which is why it covers
what it covers. Bunnymark, egui, skin, debug-blit, the a-trous denoiser, the
colour and quaternion helpers, the random-number generator, and the particle and
post-process compute passes have all been ported and checked — as modules `rustc`
compiles, transpiled by a build script, with three of them compared interface for
interface against the WGSL they came from, parsed by Naga.

Those ports are in the git history rather than the tree; there is no reason to
carry someone else's shaders here until there is something to do with them.
`tests/blade_basics.rs` keeps the constructs they needed. Of Blade's 37 shaders,
36 use only what the dialect covers; the exception is its cooperative-matrix
matmul example.

Rust keywords are the one thing that forces a rename: Blade's `fn fs_main(in: VertexOutput)`
has to call its argument something else.

## Using it

Shaders are ordinary-looking source files compiled by a build script, so
nothing at runtime parses or transpiles:

```text
src/
  main.rs
  shaders/
    common.rs      <- helpers shared by the rest
    sprite.rs      <- one shader module
build.rs
```

```rust,ignore
// build.rs
fn main() {
    synaga::build::Shaders::new().prelude("common.rs").run();
}
```

```rust,ignore
// src/main.rs — the generated module is ordinary Rust
mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

let module: naga::Module = serde_json::from_slice(shaders::SPRITE).unwrap();
```

One `pub const` per module, named after the file: the JSON from Naga's
`serialize` feature. `ALL` is `(name, bytes)` for a host that hands every
shader to the same place. Cargo re-runs the build when any shader changes, and
a shader that does not compile fails the build the way a Rust error would:

```text
error: sprites@0.1.0: src/shaders/tonemap.rs:20:1: `fn tonemap`: operator `-` does not apply to these operand types
```

Entry point names in the module are the names from the source. A pipeline asks
for `atrous3x3` if that is what the function is called. [`entry_point_names`](src/build.rs)
lists them.

`examples/sprites` is this, working.

### The shader files are real Rust

Add [`synaga-shader`](crates/shader) as a dependency and a shader module is
an ordinary Rust module: `mod shaders;` like any other, `rustc` type-checks it,
`cargo fmt` formats it, and rust-analyzer understands it. The build script
reads the same files as text and lowers them, so each one is compiled
twice — once to be checked, once to become a Naga module.

```rust,ignore
use synaga_shader::*;

pub static camera: Uniform<Camera> = binding();

#[io]
pub struct VsOut {
    #[builtin(position)] pub clip: vec4,
    #[location(0)] pub uv: vec2,
}

#[vertex]
pub fn vs(#[location(0)] pos: vec3, #[location(1)] uv: vec2) -> VsOut {
    VsOut { clip: camera.view_proj * pos.extend(1.0), uv }
}
```

Nothing in `synaga-shader` computes anything — every body panics. The types
are there to be *checked*; the shader runs on a GPU. Real implementations can
be filled in later without a signature changing.

#### Three things Rust spells differently

Rust cannot do what WGSL does here, so a checkable shader says it another way.
Both spellings transpile identically; only one of them type-checks.

| WGSL | Checkable Rust | Why |
| --- | --- | --- |
| `v.xyz`, `v.rgb` | `v.xyz()`, `v.rgb()` | one piece of memory cannot carry a hundred overlapping names (`v.x` is still a field) |
| `vec3(x)`, `vec4(v, w)` | `vec3::splat(x)`, `v.extend(w)` | a function cannot be overloaded on arity |
| `a <= b` on vectors | `a.cmple(b)` | Rust's `<=` yields one `bool`, a shader's yields one per lane |
| `v as vec3<f32>` | `vec3::from(v)` | `as` only converts primitives |
| `T()` | `T::default()` | `T()` is a call, and a struct is not a function |
| `textureLoad(t, c)` on storage | `textureLoadStorage(t, c)` | same name, one argument fewer |

`usize` is `u32`. A GPU index is 32-bit and WGSL has no `usize`, but `[T; N]`,
`[T]` and the prelude's vectors index by it and `Index` offers nothing else, so
`arr[i as usize]` has to mean what it says.

The address space moves into the type, so a global needs no attribute:
`Uniform<T>`, `Storage<T>`, `StorageMut<T>`, `Workgroup<T>`, `Private<T>`, each
initialised `= binding()`.

Writable resources are `static mut`: assigning through a shared `static` is not
something Rust allows however the type is arranged. The stage attributes wrap
function bodies in `unsafe` so the shader source does not have to say it.

One line of boilerplate per shader tree, in `src/shaders/mod.rs`, turns off the
lints that would otherwise fire on every lowercase type and global, and on the
parameters WGSL is happy to leave unread:

```rust,ignore
#![allow(
    non_camel_case_types, non_snake_case, non_upper_case_globals,
    dead_code, unused_imports, unused_variables
)]
```

### A prelude in place of `#include`

Files named with `prelude` are declarations every module can use, and are not
compiled as shaders themselves. It may be called more than once, for a shader set
with several includes. What a module does not reach is pruned from its
output, so one prelude can hold everything the set needs between them without
every shader carrying all of it — which matters for a host that binds resources
by name and would otherwise have to find something to bind an unused uniform to.

### Or call it directly

```rust
use synaga::{parse_str, validate};

let module = parse_str("fn add(a: f32, b: f32) -> f32 { a + b }")?;
validate(&module)?;
// Printing WGSL is `synaga::to_wgsl`, and it needs `features = ["wgsl"]`.
```

## Status

Curiosity / scaffolding. The IR builder is the point — Naga already knows how
to go from there to WGSL, SPIR-V, MSL, and HLSL.
