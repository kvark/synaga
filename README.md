# synaga

Shaders written as Rust modules: `rustc` type-checks them as part of your
crate, and a build script compiles the same files to [Naga](https://docs.rs/naga)
IR. No `rustc_private`, no nightly — the build script parses with
[`syn`](https://docs.rs/syn) on stable and builds a `naga::Module` by hand.

## Using it

```text
src/
  main.rs
  shaders/
    mod.rs         <- `pub mod common; pub mod sprite;`
    common.rs      <- helpers
    sprite.rs      <- `use super::common::*;` and entry points
build.rs
```

```rust,ignore
// build.rs
fn main() {
    synaga::build::Shaders::new().run();
}
```

```rust,ignore
// src/main.rs
mod shaders; // the sources, which rustc checks

mod shader_ir {
    synaga_shader::include_ir!(); // what they compile to
}

let module: naga::Module = shader_ir::SPRITE.decode()?;
```

Every file in the directory with an `#[entry_point]` is a shader and becomes
one module. The other files are what those use: `use super::common::*` or
`common::luminance(..)` in a shader is what brings `common.rs` along — the same
line that tells `rustc` where to look. A file no shader reaches is left alone.
What an entry point does not reach is pruned from its module, which matters for
a host that binds resources by name and would otherwise have to find something
to bind an unused uniform to.

`#[cfg(...)]` and `cfg!(...)` hold or not as they do for `rustc`: the build
script reads what Cargo tells it, so `cfg!(debug_assertions)` is `true` in a
debug build.

`include_ir!` works the way `tonic::include_proto!` does: it brings in a
generated file of constants, one `Ir` per shader named after its file, and
`ALL` listing them. The bytes are the module in [bincode], behind an
eight-byte header naming the format and the Naga major version that wrote it.
`decode` produces whatever `naga::Module` you name, so your own copy of Naga
reads it; that copy has to be the same major version.

Cargo re-runs the build when a shader changes, and a shader that does not
compile fails the build the way a Rust error would:

```text
error: sprites@0.1.0: src/shaders/tonemap.rs:20:1: `fn tonemap`: operator `-` does not apply to these operand types
```

`examples/sprites` is this, working.

[bincode]: https://docs.rs/bincode

## Writing a shader

```rust,ignore
use synaga_shader::*;

pub struct Camera {
    pub view_proj: mat4,
}

pub static camera: Uniform<Camera> = binding();
pub static albedo: texture_2d<f32> = binding();
pub static linear: sampler = binding();

#[derive(Clone, Copy, Io)]
pub struct VsOut {
    #[builtin(position)]
    pub clip: vec4,
    #[location(0)]
    pub uv: vec2,
}

#[entry_point(vertex)]
pub fn vs(#[location(0)] pos: vec3, #[location(1)] uv: vec2) -> VsOut {
    VsOut { clip: camera.view_proj * pos.extend(1.0), uv }
}

#[entry_point(fragment)]
#[output(location(0))]
pub fn fs(input: VsOut) -> vec4 {
    albedo.sample(&linear, input.uv)
}
```

A shader module is an ordinary module of your crate, with
[`synaga-shader`](crates/shader) providing its types. `rustc` checks it,
`cargo fmt` formats it, rust-analyzer understands it, and nothing silences
their complaints: an unused variable, a needless `mut`, a helper nothing calls
are all reported. The one lint a shader tree usually wants off is
`non_upper_case_globals`, because resources keep the lowercase names a host
binds them by.

### Entry points and their interface

`#[entry_point(vertex)]`, `#[entry_point(fragment)]` or
`#[entry_point(compute, threads(8, 4))]`. Parameters carry `#[location(N)]` or
`#[builtin(name)]`; a fragment shader returning a bare value says where it goes
with `#[output(location(0))]`.

A struct whose fields are all bound derives `Io`, which is what lets `rustc`
accept the attributes on its fields. It counts the fields as read, since the
reader is often the rasterizer or another stage, which `rustc` cannot see.

### Resources

A resource is a `static` initialised with `= binding()`. The address space is
in the type: `Uniform<T>`, `Storage<T>`, `StorageMut<T>`, `Workgroup<T>`,
`Private<T>`; textures, samplers and acceleration structures are their own
types. Each derefs to what it holds, so `camera.view_proj` reads through it.

None of them is a `static mut`. Other invocations run at the same time and may
write the same memory, which Rust calls shared mutable state; edition 2024
refuses even a read through a `static mut`. So a write goes through
`get_mut`, which is `unsafe` because the shader, not the compiler, keeps the
invocations apart:

```rust,ignore
pub static particles: StorageMut<[Particle]> = binding();

let p = particles[i];                                  // a read needs nothing
unsafe { particles.get_mut()[i].life -= delta };       // a write says so
```

Atomics are `AtomicU32` and `AtomicI32`, with the standard methods but no
`Ordering`, since WGSL's atomics are relaxed and nothing stronger:
`count.fetch_add(1)`, `lock.compare_exchange_weak(0, 1)`, which hands back a
plain `{ old_value, exchanged }`. They take `&self`, as the standard ones do, so
a buffer changed only through its atomics needs no `unsafe` at all. On the CPU
they are real atomics.

### Textures and ray queries are methods

Each operation lives on the types it applies to, so `rustc` turns away what
WGSL would: sampling an integer texture, storing to a sampled one, a depth
comparison without a comparison sampler, loading from a write-only storage
texture.

| WGSL | Rust |
| --- | --- |
| `textureSample(t, s, uv)` | `t.sample(&s, uv)` |
| `textureSampleLevel(t, s, uv, l)` | `t.sample_level(&s, uv, l)` |
| `textureSampleBias`, `…Grad`, `…Compare`, `…CompareLevel` | `sample_bias`, `sample_grad`, `sample_compare`, `sample_compare_level` |
| `textureLoad(t, c, level)` | `t.load(c, level)` — a storage texture's takes no level |
| `textureStore(t, c, v)` | `t.store(c, v)` |
| `textureDimensions(t)`, `textureDimensions(t, l)` | `t.dimensions()`, `t.level_dimensions(l)` |
| `textureNumLevels(t)`, `…Layers`, `…Samples` | `t.num_levels()`, `num_layers`, `num_samples` |
| `rayQueryInitialize(&rq, acc, desc)` | `rq.initialize(&acc, desc)` |
| `rayQueryProceed(&rq)` | `rq.proceed()` |
| `rayQueryGetCommittedIntersection(&rq)` | `rq.committed_intersection()` |
| `arrayLength(&buf.items)` | `buf.items.len()` |

An array texture takes its layer after the coordinate, as WGSL does:
`layers.sample_level(&s, uv, layer, 0.0)`.

### Items, as Rust has them

Order does not matter: a function may call one defined below it, and a struct
may hold one declared later. Sibling files reach each other the Rust ways —
`use super::brdf::*`, `use super::light::{Sun as Light}`, or a path such as
`brdf::sample(..)` or `crate::shaders::brdf::sample(..)` — and two modules may
each have a helper of the same name. `type Color = vec4;` works. A function
cannot call itself, directly or not, since a shader cannot recurse.

### Things Rust spells differently

| WGSL | Rust | Why |
| --- | --- | --- |
| `v.xyz`, `v.rgb` | `v.xyz()`, `v.rgb()` | one piece of memory cannot carry a hundred overlapping names (`v.x` is still a field) |
| `vec3(x)`, `vec4(v, w)` | `vec3::splat(x)`, `v.extend(w)` | a function cannot be overloaded on arity |
| `a <= b` on vectors | `a.cmple(b)` | Rust's `<=` yields one `bool`, a shader's yields one per lane |
| `v as vec3<f32>` | `vec3::from(v)` | `as` only converts primitives |
| `T()` | `T::default()` | `T()` is a call, and a struct is not a function |

`usize` is `u32`. A GPU index is 32-bit and WGSL has no `usize`, but `[T; N]`,
`[T]` and the vectors index by it, so `arr[i as usize]` has to mean what it
says.

`discard()` never returns, so it may end a function whatever that function
returns. A call that returns nothing can end a block, as in Rust.

## The dialect in detail

- scalars `f32`, `u32`, `i32`, `bool`; vectors `vec2`/`vec3`/`vec4` (default
  `f32`), `vecN<T>`, `vec2f`/`vec3i`/`vec4u`; matrices `mat2`/`mat3`/`mat4`,
  `matCxR`, `mat4f`, `mat2x3<f32>`
- constructors `vec3(x, y, z)`; matrix constructors from column vectors or
  column-major scalars
- components `.x`/`.y`/`.z`/`.w`, swizzles, index `v[0]` / `v[i]` / `m[0]`
- literals, unary `-`/`!`, arithmetic / compare / bitwise / shift operators
- `let`, `let x: T;` assigned later, `if`/`else` as statement or value, `loop`,
  `while`, `for x in a..b` / `a..=b`, `break`, `continue`, `return`, tail
  expressions, `unsafe { .. }`
- `x = e` and compound assignment on any place: a local, a writable global, a
  field, a component, a matrix column
- out-parameters: `&mut T` is WGSL's `ptr<function, T>`; `&T` is the same
  pointer with writes refused
- `const NAME: T = …` (literals, vector/matrix constructors, other constants,
  `cfg!(..)`)
- structs, their literals and fields; arrays `[T; N]` and `[a, b, c]`; `[T]`
  for a runtime-sized storage buffer; `binding_array<T>` and
  `binding_array<T, N>`
- math builtins: `dot`, `cross`, `normalize`, `length`, `abs`, `min`, `max`,
  `clamp`, `mix`, `step`, `sin`, `cos`, `pow`, `transpose`, `determinant`, the
  bit-twiddling set and the packing set; `select(reject, accept, condition)` in
  WGSL's argument order; `all`, `any`, `isNan`, `isInf`; `bitcast::<T>(x)`
- `workgroupBarrier()`, `storageBarrier()`, `discard()`
- the predeclared `RAY_FLAG_*` and `RAY_QUERY_INTERSECTION_*` names

Not yet: labeled loops, `break` values, `match`, methods on your own types,
generics, cooperative matrices, `f16`, `const` arithmetic (Naga wants constants
already folded). Swizzles are values, so `v.xy = a` is rejected — as it is in
WGSL. Assignment to function arguments is rejected.

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
a host validating against a real device wants anyway. The build script takes
the same through `Shaders::bindings` and `Shaders::capabilities`.

The module is the product. `to_wgsl` is `feature = "wgsl"`, for a client that
wants to read the shader as text. It prints a ray query as the WGSL builtins
and adds `enable wgpu_ray_query`: Naga's own backend has no spelling for the
operation and panics, so the module is rewritten into calls first.

### Errors

`validate` and `validate_unbound` return a `ValidationError` that prints Naga's
whole source chain. Naga puts the useful part there: the top level says only
that a global is invalid, and the reason ("the array stride 4 is not a multiple
of the required alignment 16") is one `source()` down. A transpile error names
the file and line it is in, which need not be the shader's own file when the
problem is in something it uses.

### Places

`s.a`, `v.x`, `v[i]` and `m[0]` are lowered as pointers rather than as
components picked out of a loaded value. That is what makes them assignable,
and it means reading one field of a uniform buffer loads that field instead of
the whole struct. A function argument is a value, so its fields can be read but
not written.

### Host-assigned bindings

Some engines leave `@group`/`@binding` out of the shader and fill them in at
pipeline creation, matching globals up by name — [Blade][blade] does, and
asserts the module has none. `Shaders::bindings(Bindings::Host)` says so to the
build script, and `validate_unbound` is the same for a module built by hand.
Blade takes a `naga::Module` directly (`ShaderDesc::naga_module`), so a module
built here needs no WGSL round trip to reach it.

[blade]: https://github.com/kvark/blade

### Interpolation

A float `#[location]` binding gets the default every shading language shares —
perspective-correct, center-sampled — on entry-point arguments, results and
struct fields alike, as Naga's own WGSL frontend does. Integers cannot be
interpolated, so an integer `#[location]` that is — a vertex output or a
fragment input — has to say `#[flat]`:

```rust,ignore
#[derive(Clone, Copy, Io)]
struct VsOut {
    #[builtin(position)] pos: vec4,
    #[location(0)] uv: vec2,
    #[location(1)] #[flat] material: u32,
}
```

An entry point returning a bound struct does not take `#[output(...)]`, and a
struct argument whose fields carry no bindings is left for the host to fill
in — which is how Blade supplies vertex attributes.

### Blade

The dialect was built against [Blade][blade]'s shaders, which is why it covers
what it covers. Its whole set — the renderer's ray tracer, path tracer,
G-buffer fill, rasterizer, denoiser, skinning and post-processing, plus egui,
particles and bunnymark — is 14 modules from 30 files, and all of them
type-check with `rustc` (edition 2021 and 2024) and build with the build script
as described here. The port itself is
[kvark/blade#400](https://github.com/kvark/blade/pull/400).

Rust keywords are the one thing that forces a rename: Blade's
`fn fs_main(in: VertexOutput)` has to call its argument something else.

## Or call it directly

```rust
use synaga::{parse_str, validate};

let module = parse_str("fn add(a: f32, b: f32) -> f32 { a + b }")?;
validate(&module)?;
// Printing WGSL is `synaga::to_wgsl`, and it needs `features = ["wgsl"]`.
```

`synaga::parse` takes several named sources and a `Cfg`, which is what the
build script uses.

## Status

Young. The IR builder is the point — Naga already knows how to go from there to
WGSL, SPIR-V, MSL, and HLSL.
