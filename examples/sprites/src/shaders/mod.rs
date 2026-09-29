//! Shader modules.
//!
//! These are compiled twice: `rustc` checks them as ordinary Rust, and the
//! build script reads the same files and serializes a Naga module.
//!
//! Resources keep lowercase names, as WGSL globals have and as a host that
//! binds by name needs, which is the one Rust convention a shader breaks on
//! purpose.
#![allow(non_upper_case_globals)]

pub mod common;
pub mod sprite;
pub mod tonemap;

// `rustc`'s layout of `Globals` and `Locals`, asserted to be the GPU's.
synaga_shader::check_layout!();
