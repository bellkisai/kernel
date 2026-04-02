//! `shrimpk-ros2` — ROS2 bridge for ShrimPK Echo Memory.
//!
//! Subscribes to ROS2 topics (text, image, audio, pose) and forwards each
//! received message to the ShrimPK daemon at `http://localhost:11435` as a
//! push-based memory.
//!
//! # Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | *(none)* | Compiles as a replay tool: reads JSON log files and replays them to the daemon |
//! | `ros2` | Enables the live ROS2 node via `r2r` — requires a ROS2 installation |
//!
//! # Architecture
//!
//! ```text
//! ROS2 topics  ──►  node.rs (r2r, #[cfg(ros2)])
//!                       │
//!                       ▼
//!                  bridge.rs  ──►  client.rs  ──►  ShrimPK daemon :11435
//!                  (conversion)     (HTTP POST)
//! ```

pub mod bridge;
pub mod client;
pub mod config;

#[cfg(feature = "ros2")]
pub mod node;
