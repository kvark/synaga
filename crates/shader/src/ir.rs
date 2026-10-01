//! The compiled shaders, as the crate that uses them receives them.
//!
//! The build script writes each shader's Naga module to `OUT_DIR`, along with
//! a Rust file of constants that point at them.
//! [`include_ir!`](crate::include_ir) brings that file in, the way
//! `tonic::include_proto!` does for generated protobuf code, and
//! [`Ir::decode`] turns a constant back into a module:
//!
//! ```ignore
//! mod shader_ir {
//!     synaga_shader::include_ir!();
//! }
//!
//! let module: naga::Module = shader_ir::EGUI.decode().expect("egui shader IR");
//! ```
//!
//! The bytes are bincode, after an eight-byte header: `SYNAGA`, the format
//! version, and the major version of the Naga that wrote them. `decode`
//! produces whatever `naga::Module` the caller names, so the caller's own copy
//! of Naga does the reading. That copy has to agree with the writer's about
//! the format, which in practice means the same major version.

/// The first six bytes of every module the build step writes.
pub const MAGIC: &[u8; 6] = b"SYNAGA";

/// The major version of the Naga whose modules this crate reads.
///
/// Every module records the Naga that wrote it, and the generated file asserts
/// that the two agree — a build error naming both versions, rather than a panic
/// at the first `decode` where the bytes stop making sense. A crate that
/// depends on `synaga` as a build-dependency and on `synaga-shader` normally
/// gets the same version of each, and this is what proves it did.
pub const NAGA_MAJOR: u8 = 30;

/// The layout after the header. Bumped when it changes.
pub const FORMAT: u8 = 1;

const HEADER_LEN: usize = MAGIC.len() + 2;

/// One compiled shader module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ir {
    bytes: &'static [u8],
}

impl Ir {
    /// Wrap bytes the build step wrote. The generated constants do this.
    pub const fn new(bytes: &'static [u8]) -> Self {
        Self { bytes }
    }

    /// The bytes, header included.
    pub const fn bytes(&self) -> &'static [u8] {
        self.bytes
    }

    /// The major version of the Naga that wrote the module, if the header is
    /// there to say.
    pub fn naga_major(&self) -> Option<u8> {
        naga_major(self.bytes)
    }

    /// Decode into a Naga module: `let module: naga::Module = ir.decode()?`.
    /// See [`decode`].
    #[cfg(feature = "ir")]
    pub fn decode<M: serde::de::DeserializeOwned>(&self) -> Result<M, DecodeError> {
        decode(self.bytes)
    }
}

/// The format version and the writer's Naga major version.
fn header(bytes: &[u8]) -> Result<(u8, u8), DecodeError> {
    let header = bytes.get(..HEADER_LEN).ok_or(DecodeError::NotIr)?;
    if &header[..MAGIC.len()] != MAGIC {
        return Err(DecodeError::NotIr);
    }
    Ok((header[MAGIC.len()], header[MAGIC.len() + 1]))
}

/// The major version of the Naga that wrote `bytes`, if they carry the header
/// to say so.
pub fn naga_major(bytes: &[u8]) -> Option<u8> {
    header(bytes).ok().map(|(_, naga)| naga)
}

/// Decode a module the build step wrote, from wherever the bytes are now: an
/// [`Ir`] constant, or an asset cache they went through on the way.
///
/// Any type that deserializes will do, which is what lets the module be the
/// caller's own `naga::Module` rather than synaga's.
#[cfg(feature = "ir")]
pub fn decode<M: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<M, DecodeError> {
    let (format, naga_major) = header(bytes)?;
    if format != FORMAT {
        return Err(DecodeError::Format {
            found: format,
            expected: FORMAT,
        });
    }
    let payload = &bytes[HEADER_LEN..];
    let (module, read) =
        bincode::serde::decode_from_slice::<M, _>(payload, bincode::config::standard()).map_err(
            |err| DecodeError::Payload {
                naga_major,
                message: err.to_string(),
            },
        )?;
    // Bytes left over mean the reader's idea of the layout is not the
    // writer's, even though what it read happened to parse.
    if read != payload.len() {
        return Err(DecodeError::Payload {
            naga_major,
            message: format!("{} bytes left over", payload.len() - read),
        });
    }
    Ok(module)
}

/// Why [`decode`] failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The bytes were not written by synaga's build step.
    NotIr,
    /// Written in a layout this version of `synaga-shader` does not read.
    Format { found: u8, expected: u8 },
    /// The module did not decode. The usual reason is that the Naga reading
    /// it is not the version that wrote it.
    Payload { naga_major: u8, message: String },
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::NotIr => write!(f, "not a module written by synaga"),
            DecodeError::Format { found, expected } => write!(
                f,
                "synaga IR format {found}, but this synaga-shader reads format {expected}"
            ),
            DecodeError::Payload {
                naga_major,
                message,
            } => write!(
                f,
                "the module was written by Naga {naga_major} and did not decode ({message}); \
                 the Naga reading it has to be the same major version"
            ),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Bring in the constants the build step generated: one [`Ir`] per shader
/// module, and `ALL` listing them. Place it inside a module of its own.
///
/// ```ignore
/// mod shader_ir {
///     synaga_shader::include_ir!();
/// }
/// ```
///
/// Name the file if the build script chose a different one with
/// `Shaders::module_name`: `include_ir!("bunnymark_shaders.rs")`.
#[macro_export]
macro_rules! include_ir {
    () => {
        include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
    };
    ($file:literal) => {
        include!(concat!(env!("OUT_DIR"), "/", $file));
    };
}

/// Checks `rustc`'s layout of each struct the shaders share with the host
/// against the GPU's, as `size_of` and `offset_of!` assertions the build wrote.
///
/// The transpiler already lays each such struct out as `rustc` would and
/// refuses it where the GPU disagrees; this asks `rustc` itself, for the target
/// being built. It goes in the module that lists the shader modules, since
/// the assertions name each struct through its module: `camera::Camera` is
/// `self::camera::Camera` there.
///
/// ```ignore
/// // src/shaders/mod.rs
/// pub mod camera;
/// synaga_shader::check_layout!();
/// ```
///
/// The file is named after the generated module: `shaders_layout.rs` for the
/// default `shaders.rs`, or name it for another.
#[macro_export]
macro_rules! check_layout {
    () => {
        include!(concat!(env!("OUT_DIR"), "/shaders_layout.rs"));
    };
    ($file:literal) => {
        include!(concat!(env!("OUT_DIR"), "/", $file));
    };
}
