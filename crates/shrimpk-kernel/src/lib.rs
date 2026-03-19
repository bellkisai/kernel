//! # shrimpk-kernel
//!
//! The ShrimPK AI Operating System kernel.
//! This is the facade crate — it re-exports all subsystem crates
//! for convenient single-import usage.
//!
//! ```rust
//! use shrimpk_kernel::*; // Gets everything
//! ```

pub use shrimpk_context;
pub use shrimpk_core;
pub use shrimpk_memory;
pub use shrimpk_router;
pub use shrimpk_security;
