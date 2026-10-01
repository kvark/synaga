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
reads it; that copy has to be the same major version. The generated file
asserts that version against `synaga_shader::ir::NAGA_MAJOR` as it compiles, so
a `synaga` build-dependency and a `synaga-shader` dependency that disagree is a
build error naming both numbers rather than a panic at the first `decode()`.

The generated code says `::synaga_shader::…`, so the host crate has to depend on
the package `synaga-shader` under that name — a `rename` is not enough, since
the path in the generated file is fixed.

Cargo re-runs the build when a shader changes, and a shader that does not
compile fails the build the way a Rust error would:

```text
error: sprites@0.1.0: src/shaders/tonemap.rs:20:14: `fn tonemap`: operator `-` does not apply to these operand types
```

`examples/sprites` is this, working.

### Seeing what was built

The IR is what a host runs, and it is not readable. Two things make a surprising
module diagnosable:

```rust,ignore
// build.rs — write each module beside its bytes, as WGSL.
synaga::build::Shaders::new().wgsl().run();
```

That writes `$OUT_DIR/wgsl/<name>.wgsl` next to each `<name>.naga`, and keeps
each module on the [`Shader`] the build returns, so a build script can report on
what it built — how many entry points, what survived pruning, the WGSL of the one
that looks wrong — without decoding the bytes it just wrote. `keep_modules()`
does the same without the text, and `Shader::module` is `None` unless one of
them asked.

A failure names the file, the line and the column:

```text
error: src/shaders/tonemap.rs:20:14: `fn tonemap`: operator `-` does not apply to these operand types
```

The position is the expression that failed rather than the item around it, and a
failure in something a shader `use`s names that file. Every expression,
statement, global and type in the module carries the byte range of the `syn` node
it came from, so a Naga validation failure names a line too. One build reports
every fault in the tree, not the first:

```text
error: src/shaders/a.rs:3:18: `fn fs`: unsupported expression: literal
src/shaders/b.rs:1:1: `fn fs`: unknown function `also_not_a_function`
```

The first is the expression that failed. The second is the entry point's own
line, since nothing inside the body had a position yet — a name that does not
resolve fails before any expression does.

[bincode]: https://docs.rs/bincode

## Writing a shader

```rust,ignore
use synaga_shader::*;

pub struct Camera {
    pub view_proj: Mat4,
}

pub static camera: Uniform<Camera> = group(0).binding(0);
pub static albedo: Texture2D<f32> = group(1).binding(0);
pub static linear: Sampler = group(1).binding(1);

#[derive(Clone, Copy, Io)]
pub struct VsOut {
    #[builtin(position)]
    pub clip: Vec4,
    #[location(0)]
    pub uv: Vec2,
}

#[entry_point(vertex)]
pub fn vs(#[location(0)] pos: Vec3, #[location(1)] uv: Vec2) -> VsOut {
    VsOut { clip: camera.view_proj * pos.extend(1.0), uv }
}

#[entry_point(fragment)]
pub fn fs(input: VsOut) -> Vec4 {
    albedo.sample(&linear, input.uv)
}
```

A shader module is an ordinary module of your crate, with
[`synaga-shader`](crates/shader) providing its types. `rustc` checks it,
`cargo fmt` formats it, rust-analyzer understands it, and nothing silences
their complaints: an unused variable, a needless `mut`, a helper nothing calls
are all reported. The one lint a shader tree usually wants off is
`non_upper_case_globals`, because resources keep lowercase names: WGSL's, and
the ones a host that binds by name looks for.

### Entry points and their interface

`#[entry_point(vertex)]`, `#[entry_point(fragment)]` or
`#[entry_point(compute, threads(8, 4))]`. A parameter carries `#[location(N)]`
or `#[builtin(name)]`, or is named after a builtin its stage takes and needs
neither:

```rust,ignore
#[entry_point(compute, threads(64))]
pub fn reset(global_invocation_id: Vec3<u32>, num_workgroups: Vec3<u32>) { .. }
```

A bare value an entry point returns is a vertex shader's position and a
fragment shader's first colour target, `location(0)`, and nothing else. Any
other result, a depth or a second colour target, is a struct whose fields say
where they go. The function never says it: Rust has no attributes on types,
and the result's binding belongs with the result.

A struct whose fields are all bound derives `Io`, which is what lets `rustc`
accept the attributes on its fields. It counts the fields as read, since the
reader is often the rasterizer or another stage, which `rustc` cannot see.

### Resources

A resource is a `static` whose type says what it is and whose initialiser says
where it binds. `= group(0).binding(1)` is WGSL's `@group(0) @binding(1)`, and
either number may be a `const`, which the host can build its layouts from too:

```rust,ignore
pub const MATERIAL: u32 = 1;

pub static albedo: Texture2D<f32> = group(MATERIAL).binding(0);
```

`= binding()` leaves the binding to the host (see
[below](#host-assigned-bindings)), and is also how workgroup and private
memory is written: it binds to nothing, and `rustc` refuses
`group(..).binding(..)` on it. The build script refuses a resource with no
binding, at the line it is on, unless the host assigns them.

The address space is in the type: `Uniform<T>`, `Storage<T>`, `StorageMut<T>`,
`Workgroup<T>`, `Private<T>`; textures, samplers and acceleration structures
are their own types. Each derefs to what it holds, so `camera.view_proj` reads
through it.

None of them is a `static mut`, since edition 2024 refuses even a read through
one. A write goes through `get_mut`:

```rust,ignore
pub static particles: StorageMut<[Particle]> = group(0).binding(0);

let p = particles[i];                       // a read derefs
particles.get_mut()[i].life -= delta;       // a write says so

let p = &mut particles.get_mut()[i];        // or name the element
p.pos += p.vel * delta;

emit_end.set(count);                        // or write all of it
```

`emit_end.set(count)` is `*emit_end.get_mut() = count`. `*emit_end = count`
would be nicer, but it borrows the `static` mutably, which Rust refuses for one
that is not `mut`; `set` takes `&self`, as `Cell::set` does.

Other invocations run at the same time and may write the same memory, and the
shader keeps them apart, as it has to in WGSL. `get_mut` hands out `&mut` from
`&self`, which would be unsound if it ever returned on the CPU; it does not,
so it is not `unsafe`. [Resources and aliasing](#resources-and-aliasing) says
what that means for a shader, and for running one on the CPU.

Atomics are `Atomic<u32>` and `Atomic<i32>`, with the standard methods but no
`Ordering`, since WGSL's atomics are relaxed and nothing stronger. They are
generic, as the vectors are, and not the standard `AtomicU32`, whose methods
take the `Ordering`:
`count.fetch_add(1)`, `lock.compare_exchange_weak(0, 1)`, which hands back a
plain `{ old_value, exchanged }`. They take `&self`, as the standard ones do, so
a buffer changed only through its atomics needs no `get_mut`. On the CPU they
are real atomics.

### Sharing a struct with the host

A struct the host fills in and uploads is the shader's own, rather than a copy
kept the same by hand. `#[repr(C)]` says the host shares it, and the build
checks, for each such struct a buffer holds, that the GPU reads every field
where `rustc` puts it. That is the one thing that can go silently wrong, and the
error names the line it is written on and says how to put it right:

```text
src/shaders/common.rs:3:1: `static g`: `G` is `#[repr(C)]`, which says the host
shares it, but `b` is at byte 4 in Rust and 16 on the GPU; 12 bytes of padding
before it line them up
```

```rust,ignore
#[repr(C)]
#[derive(Shared)]
pub struct Globals {
    pub mvp_transform: Mat4,
    pub sprite_size: Vec2,
    pub _pad: Vec2,
}
```

`#[derive(Shared)]` is what uploading one takes: `Clone`, `Copy`, a `Default`
of all zeroes, which is the GPU's default too, and `bytemuck`'s `Zeroable` and
`Pod`, checked as `bytemuck`'s own derive checks them.

The transpiler works out `rustc`'s layout from the types. To have `rustc`
confirm it, for the target being built, the module that lists the shader
modules says `synaga_shader::check_layout!()`. The build writes a `size_of`
assertion for each shared struct and an `offset_of!` for each field that module
can name, and a difference fails the build there:

```rust,ignore
// src/shaders/mod.rs
pub mod common;
pub mod sprite;

synaga_shader::check_layout!();
```

`Shared` needs `synaga-shader`'s `bytemuck` feature, which makes the vectors
and matrices `Pod`. They convert from arrays, a
matrix column by column, and with the `mint` feature from mint's types, which
most math crates convert to. A matrix column of three lanes takes four, in Rust
as on the GPU, so any matrix can be shared; an array of `Vec3`, whose elements
the GPU spaces 16 bytes apart, cannot. `examples/sprites` shares its uniforms
this way.

`check_layout!` is what makes this a build error rather than a runtime surprise,
so it is worth noticing if you have not added it. When a module has shared
structs and nothing has included the layout file, the build script says so:

```text
warning: synaga: 2 shared struct(s) in src/shaders have `rustc` layout checks in
…/shaders_layout.rs, which nothing has included — add
`synaga_shader::check_layout!("shaders_layout.rs");` to the module that lists
the shader modules, or the layout is checked only by synaga's own model of `rustc`
```

The checks cover `size_of`, `align_of` and each field's `offset_of!`.

A `#[repr(u32)]` enum and a `bitflags!` set around a `u32` are a `u32` on the
GPU, so a shared struct can hold them, and a shader compares and tests them as
Rust does: `debug.mode == Mode::Depth`, `debug.draw.contains(Draw::SPACE)`.
A set's `!` is its complement, which stays within the declared flags, and its
`-` is the difference; neither is the `u32` operator. A set that derives
`Pod` is declared on a newtype, as `bitflags` shows for any derive it lacks:

```rust,ignore
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Draw(u32);

bitflags::bitflags! {
    impl Draw: u32 {
        const SPACE = 1;
        const RESTIR = 1 << 1;
    }
}
```

An enum cannot be `Pod`, since not every `u32` is one of its variants, so a
struct that holds one derives `bytemuck::NoUninit`, which is all an upload
needs.

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
`layers.sample_level(&s, uv, layer, 0.0)`. A ray's flags are a set,
`RayFlag::TERMINATE_ON_FIRST_HIT | RayFlag::CULL_NO_OPAQUE`, and what a query
found is an enum, `hit.kind == RayQueryIntersection::None`: WGSL's
`RAY_FLAG_*` and `RAY_QUERY_INTERSECTION_*`, typed.

### Math, as `f32` and glam name it

`x.sqrt()`, `y.atan2(x)`, `a.max(b)`, `v.normalize()`, `v.dot(w)`, `a.lerp(b, t)`:
math is methods, named as `f32` names them and as glam names what only a vector
has. Each means what the Rust method means. Where the GPU builtin of that name
does something else, the difference is made up, as `x.fract()` is
`x - x.trunc()`, or refused: `x.round()` takes a half away from zero in Rust and
to the even neighbour on the GPU, so the shader says `x.round_ties_even()`.
An operator between a vector and a scalar applies the scalar to every lane,
from either side, as glam's do: `srgb / 12.92`, `1.0 - v`, `bits & 0xFF`.

The rest of `core` a shader reaches for works too: `core::f32::consts::PI`,
`u32::MAX`, `x.to_bits()` and `f32::from_bits(n)`, `n.rotate_left(k)`,
`n.unsigned_abs()`, `a.wrapping_mul(b)` and the other `wrapping_` operations,
`mask.all()` on a `Vec3<bool>`, `v.element_sum()`, and
`v.cast::<i32>()`, which converts every lane as `as` converts a scalar. A
struct literal may end in `..Default::default()` or `..other`. `T::default()`
is the zero value, so a struct has to derive `Default` for it, or implement it
in the shader: one written by hand in the host is one the transpiler cannot
see.

### Items, as Rust has them

Order does not matter: a function may call one defined below it, and a struct
may hold one declared later. Sibling files reach each other the Rust ways —
`use super::brdf::*`, `use super::light::{Sun as Light}`, or a path such as
`brdf::sample(..)` or `crate::shaders::brdf::sample(..)` — and two modules may
each have a helper of the same name. `type Color = Vec4;` works. A function
cannot call itself, directly or not, since a shader cannot recurse.

An `impl` gives a type associated functions and methods, and a trait gives
them to a type from elsewhere, a vector included:

```rust,ignore
impl Material {
    pub fn from_metallic_roughness(base: Vec3, metalness: f32, roughness: f32) -> Self {
        Self { diffuse: base * (1.0 - metalness), roughness }
    }
    pub fn alpha(self) -> f32 { self.roughness * self.roughness }
}

pub trait Quaternion { fn inv(self) -> Self; }
impl Quaternion for Vec4 {
    fn inv(self) -> Self { (-self.xyz()).extend(self.w) }
}
```

Each method is a function named after its type, `Material_alpha`. `self` and
`&self` take a copy, which is all a read needs, and `&mut self` a pointer, so
its receiver has to be a local: WGSL passes no pointer into a buffer. A
method is lowered when a shader calls it, so an `impl` only the host uses,
such as `Debug`, can sit beside the type. As in Rust, a type's own method
comes before a trait's: a trait's `dot` on a `Vec3` is never the one `v.dot(w)`
calls. A generic `impl` is left to the host, and `mut self` and a trait's
default methods are not supported.

### Things Rust spells differently

| WGSL | Rust | Why |
| --- | --- | --- |
| `vec3<i32>`, `mat4x4f`, `texture_2d<f32>` | `Vec3<i32>`, `Mat4`, `Texture2D<f32>` | Rust capitalizes types; a bare `Vec3` is `Vec3<f32>` |
| `vec3u(1, 2, 3)` | `vec3::<u32>(1, 2, 3)` | the scalar is a type argument; `vec3(1, 2, 3)` is `i32`, as in WGSL |
| `workgroupBarrier()`, `countOneBits(x)` | `workgroup_barrier()`, `count_one_bits(x)` | Rust's functions are snake_case |
| `v.xyz`, `v.rgb` | `v.xyz()`, `v.rgb()` | one piece of memory cannot carry a hundred overlapping names (`v.x` is still a field) |
| `vec3(x)`, `vec4(v, w)` | `Vec3::splat(x)`, `v.extend(w)` | a function cannot be overloaded on arity |
| `a <= b` on vectors | `a.cmple(b)` | Rust's `<=` yields one `bool`, a shader's yields one per lane |
| `all(a <= b)`, `any(a != b)` | `a <= b`, `a != b` | that one `bool` is every lane's, as `PartialOrd` and `PartialEq` are |
| `vec3<f32>(v)` | `Vec3::from(v)` | a type is not a function, and `as` only converts primitives |
| `T()` | `T::default()` | `T()` is a call, and a struct is not a function |

The types are WGSL's words in WGSL's order, capitalized: `texture_storage_2d_array`
is `TextureStorage2DArray`, `sampler_comparison` is `SamplerComparison`. A
float literal needs no suffix: `vec3(0.0, 1.0, 0.0)` is a `Vec3<f32>`, because
`f32` is the one float type a vector holds. The transpiler still reads WGSL's
own names, for a source `rustc` never sees.

`usize` is `u32`. A GPU index is 32-bit and WGSL has no `usize`, but `[T; N]`,
`[T]` and the vectors index by it, so `arr[i as usize]` has to mean what it
says. What the host shares cannot hold one, since the host's is as wide as a
pointer: a `#[repr(C)]` struct or a buffer with a `usize` or an `isize` in it,
through arrays and aliases, is refused with the fixed-width type to use.

Comparing two vectors with an operator is one `bool`, true when it is true in
every lane: `p >= Vec2::ZERO && p < extent` is a bounds check. `==` is every
lane too, and `!=` its negation, some lane differing. So `!(a < b)` is not
`a >= b`: it holds when some lane is not less, which is `a.cmpge(b).any()`, the
test for being out of bounds. `a >= b` holds only when every lane is. Clippy's
`neg_cmp_op_on_partial_ord` warns about `!(a < b)` for this reason, and
`a.cmpge(b).any()` says the same without the negation.

`discard()` never returns, so it may end a function whatever that function
returns. A call that returns nothing can end a block, as in Rust.

## What a shader means

A shader is checked as Rust, so it means what the Rust means, as far as a GPU
can do it. Where the GPU cannot, this says what happens instead.

### Evaluation

- A `match` takes the first arm that matches, as Rust does, though a
  `switch` takes the case that is the value: an arm or a value an earlier arm
  takes is dropped. A `break` in a `switch` would leave the `switch`, so a
  `match` with an arm that breaks out of a loop is an `if` chain.
- `a && b` and `a || b` evaluate `b` only when `a` does not decide. A `b` that
  calls a function, touches an atomic or indexes goes under an `if`; one that
  only reads and computes is evaluated early, since nothing can tell, and
  that saves a local.
- `for i in a..b` and `a..=b` evaluate `a` and `b` once, before the loop. An
  inclusive range runs for `b` and stops, even when `b` is the largest value
  its type has. `for mut i in ..` gives the body a copy to change; the
  iteration goes on regardless.
- Operands, arguments and struct fields are evaluated left to right. A
  function cannot call itself, directly or not.
- `T::default()` is the zero value, so a struct has to derive `Default` for it,
  and a `#[repr(u32)]` enum's `#[default]` variant has to be zero.

### Numbers

The GPU's integers wrap, and Rust's panic under overflow checks, which a debug
build has on. Code that means to wrap says so, `hash.wrapping_mul(k)`, and
wraps on both. The rest of what differs, operator by operator:

| | GPU | Rust on the CPU |
| --- | --- | --- |
| `+`, `-`, `*` past the type's range | wraps | panics under overflow checks, wraps without |
| `wrapping_add`, `_sub`, `_mul`, `_neg`, `_shl`, `_shr` | wraps | wraps |
| `x / 0`, `x % 0` | `x`, and `0` | panics |
| `i32::MIN / -1`, `i32::MIN % -1` | `i32::MIN`, and `0` | panics |
| `x << n`, `x >> n` with `n >= 32` | depends on the backend | panics under overflow checks |
| `-i32::MIN`, `i32::MIN.abs()` | `i32::MIN` | panics under overflow checks |
| a float `as` an integer, out of range | the nearest integer a float can hold: `1e10 as i32` is 2147483520 | the nearest integer: 2147483647 |
| a NaN `as` an integer | unspecified | 0 |
| `i32` `as` `u32` and back | the same bits | the same bits |
| an index out of bounds | what the host's Naga bounds-check policy says: kept in bounds, read as zero, or, by default, left to the driver | panics |
| `x.round()` | refused: the GPU's rounds a half to the even neighbour, which is `round_ties_even()` | |
| `0.0f32.signum()` | refused: the GPU's `sign(0.0)` is 0 | 1 |
| `min`, `max` with a NaN | either operand | the one that is a number |

`+`, `-` and `*` are correctly rounded on both, and so is a conversion that
fits. Beyond those, WGSL allows error: 2.5 ULP for `/`, a few for the square
roots, more for `exp`, `log` and `pow`, which is `exp2(y * log2(x))` and only
defined for `x >= 0`, and an absolute 2^-11 for `sin` and `cos` on `[-π, π]`.
`fma` may or may not round once. The CPU's are Rust's, which are closer. An
ill-conditioned function turns those few ULP into more: Blade's GGX sampling
at a roughness of 0.05 moves its density by percents for an ulp in the half
vector it draws.

The free functions keep WGSL's meanings on both sides, as they are named after
WGSL's: `fract(x)` is `x - floor(x)`, `round(x)` takes a half to the even
neighbour, `sign(0.0)` is 0, `abs(i32::MIN)` is `i32::MIN`, and
`clamp(x, low, high)` is `min(max(x, low), high)`. The methods of those names
are Rust's.

### On the CPU

`synaga-shader`'s vectors, matrices and builtins run on the CPU, each meaning
there what it means in the shader, so a shader's pure functions are ordinary
Rust functions: call them, test them, compare them with the GPU. Blade does,
for its BRDF, sampling, hashing, packing, quaternions and camera, integers
exactly and floats to within what the GPU's precision allows.

What needs the GPU panics on the CPU: resources, textures and samplers, ray
queries, barriers and `discard()`. Running those needs a runtime that runs a
workgroup's invocations with their memory, which there is not yet. So a
function that is to be tested on the CPU takes values, and the entry point
that calls it does the reading and writing.

### Resources and aliasing

A resource is a `static` whose type says its address space. Reading one
derefs to `&T`, and a shader reads through it: `camera.view` loads that field,
not the whole struct. Writing goes through `get_mut()`, which returns `&mut T`
from `&self`.

That is not Rust's aliasing. On the GPU every invocation runs at once, and each
holds what Rust would call an exclusive reference to the same buffer. Keeping
their writes apart is the shader's job, as it is in WGSL: an element per
invocation, a barrier between phases, or an atomic. Two things follow from the
memory model rather than from Rust:

- A store is as wide as its place. `particles.get_mut()[i].life -= dt` stores
  one field; `particles.get_mut()[i] = p` stores the whole element, and
  overwrites what another invocation wrote to a field of it meanwhile.
- Atomics are relaxed, the only ordering WGSL has, so their methods take no
  `Ordering` rather than one that would be ignored. On the CPU they are real
  atomics, relaxed.

`get_mut()` is sound in Rust only because it never returns on the CPU: that
is why a resource cannot be read or written there. Running a shader's entry
points on the CPU needs something else in its place, an access that carries
the buffer, the offset and the permission, `slot.load()` and `slot.store(v)`,
which a CPU runtime can check and schedule. That is not built. Until it is,
the CPU runs a shader's pure functions, and the GPU the rest.

## The dialect in detail

- scalars `f32`, `u32`, `i32`, `bool`; vectors `Vec2<T>`/`Vec3<T>`/`Vec4<T>`
  of any of them, `f32` when `T` is left out; matrices `Mat2`/`Mat3`/`Mat4` and
  `Mat2x3` through `Mat4x3`, columns first as in WGSL's `matCxR`
- constructors `vec3(x, y, z)`; matrix constructors from column vectors or
  column-major scalars
- components `.x`/`.y`/`.z`/`.w`, swizzles, index `v[0]` / `v[i]` / `m[0]`
- literals, unary `-`/`!`, arithmetic / compare / bitwise / shift operators
- `let`, `let x: T;` assigned later, `if`/`else` as statement or value, `loop`,
  `while`, `for x in a..b` / `a..=b`, `break`, `continue`, `return`, tail
  expressions, `unsafe { .. }`
- `match` on an integer, an enum, a flags set or a `bool`, as statement or
  value: literal, `const`, variant and flag patterns, `|`, `_`, and a name that
  binds the value
- `impl` blocks: associated functions and `self`, `&self` and `&mut self`
  methods, on a type of the shader's own or, through a trait, on a vector
- `x = e` and compound assignment on any place: a local, a writable global, a
  field, a component, a matrix column
- out-parameters: `&mut T` is WGSL's `ptr<function, T>`; `&T` is the same
  pointer with writes refused
- `let p = &mut place;` and `let p = &place;` name a place, which is then read
  and written through `p`
- `#[repr(u32)]` enums and `bitflags!` sets around a `u32`, as types and
  values: `==`, `|`, `&`, `^`, a set's `!` and `-`, `contains`, `intersects`,
  `is_empty`, `is_all`, `bits`, `Flags::empty()`, `Flags::all()`,
  `Flags::from_bits_truncate(b)`, an enum's `#[default]` variant
- `const NAME: T = …` (literals, vector/matrix constructors, other constants,
  `u32::MAX` and the rest of a primitive's own, `core::f32::consts`, `cfg!(..)`)
- structs, their literals, with `..Default::default()` or `..other` for the rest,
  and fields; arrays `[T; N]` and `[a, b, c]`; `[T]`
  for a runtime-sized storage buffer; `BindingArray<T>` and
  `BindingArray<T, N>`
- math builtins: `dot`, `cross`, `normalize`, `length`, `abs`, `min`, `max`,
  `clamp`, `mix`, `step`, `sin`, `cos`, `pow`, `transpose`, `determinant`, the
  bit-twiddling set and the packing set; `select(reject, accept, condition)` in
  WGSL's argument order; `all`, `any`, `isNan`, `isInf`; `bitcast::<T>(x)`
- `workgroup_barrier()`, `storage_barrier()`, `discard()`
- `RayFlag` and `RayQueryIntersection`, and WGSL's predeclared `RAY_FLAG_*` and
  `RAY_QUERY_INTERSECTION_*` names they stand for

Not yet: labeled loops, `break` values, `match` guards and range patterns,
generics, cooperative matrices, `f16`, `const` arithmetic (Naga wants constants
already folded). Swizzles are values, so `v.xy = a` is rejected — as it is in
WGSL. Assignment to function arguments is rejected.

### What the transpiler will not read

A shader is checked by `rustc` as ordinary Rust, so anything `rustc` accepts is
in the dialect as far as the type checker is concerned. The transpiler reads
less than that, and where it reads less it says so rather than quietly building a
smaller module — a shader missing from an output is the one failure this design
exists to prevent.

- **A file nothing reaches is an error.** `rustc` checks every file in the
  directory as a shader, and a file no `#[entry_point]` pulls in is in no module
  at all. Move it out of the directory, or reach it with `use super::…`.
- **`mod { .. }` is refused.** An `#[entry_point]` inside one would be a shader
  nothing ever compiles. Its items go in a file of their own.
- **`mod name;` is refused.** Every source is a module named after its file, so
  there is no separate file to pull in.
- **The directory is searched recursively**, so `shaders/brdf/ggx.rs` is a module
  named `ggx` reached as `use super::ggx::*`. Two files of one stem in different
  directories would be ambiguous, and the reachability scan treats them as one.

`#[cfg(...)]` and `cfg!(...)` hold or not as they do for `rustc`, including
`#[cfg(test)]`, which Cargo does not pass to a build script and which is
recovered from the profile it does.

### Typing

Operand rules follow Naga's, so a program the frontend accepts is a module its
validator accepts — `-x` needs a signed or float operand, `a & b` an integer or
bool one, a shift amount is always `u32`, and so on.

Untyped integer literals take their type from context the way Rust's inference
would: `n << 1`, `n * 2`, `clamp(n, 0, 10)`, `f(1)`, `let n: u32 = 1` and
`fn f() -> u32 { 1 }` all work whatever integer type is in play. A vector's
scalar comes from the same place: `let c: Vec3<u32> = vec3(1, 2, 3)` and
`(Vec4::splat(raw) >> vec4(0, 8, 16, 24)) & Vec4::splat(0xFF)` are `u32`
vectors because of what they meet, as `rustc` sees it. There is no `1` to `1.0`
conversion, again as in Rust.

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

`let p = &mut particles.get_mut()[i];` makes `p` that pointer. The element's
index is worked out once, where it is borrowed, and `p.x = 1.0` stores to the
buffer rather than to a copy.

### Host-assigned bindings

Some engines leave `@group`/`@binding` out of the shader and fill them in at
pipeline creation, matching globals up by name — [Blade][blade] does, and
asserts the module has none. `Shaders::bindings(Bindings::Host)` says so to the
build script, and `validate_unbound` is the same for a module built by hand.
Every resource is then written `= binding()`, and one that says where it binds
is refused at the line it is on, rather than by the host at pipeline creation.
A vertex struct argument whose fields have no `#[location]` is the same
arrangement for vertex attributes: Blade matches them up by name too, so it
builds only in this mode.
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
    #[builtin(position)] pos: Vec4,
    #[location(0)] uv: Vec2,
    #[location(1)] #[flat] material: u32,
}
```

When the default is not the one you want, say so in WGSL's own spelling rather
than through a synonym the reader has to learn:

```rust,ignore
#[derive(Clone, Copy, Io)]
struct VsOut {
    #[builtin(position)] #[invariant] clip: Vec4,
    #[location(0)] #[interpolate(linear, centroid)] weight: f32,
    #[location(1)] #[interpolate(flat, first)] id: u32,
}
```

`#[interpolate(mode)]` takes the mode alone, defaulting the sampling to
`center`. A position can also say `#[builtin(position_invariant)]`, which is
what WGSL calls it.

A struct argument whose fields carry no bindings is left for the host to fill
in — which is how Blade supplies vertex attributes, and needs
`Bindings::Host`.

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
