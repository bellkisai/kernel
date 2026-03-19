//! # bellkis-kernel
//!
//! The Bellkis AI Operating System kernel.
//! This is the facade crate — it re-exports all subsystem crates
//! for convenient single-import usage.
//!
//! ```rust
//! use bellkis_kernel::*; // Gets everything
//! ```

pub use bellkis_core;
pub use bellkis_memory;
pub use bellkis_router;
pub use bellkis_context;
pub use bellkis_security;
