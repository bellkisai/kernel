//! Bridge configuration — which ROS2 topics to subscribe and where the daemon lives.

use serde::{Deserialize, Serialize};

/// Top-level configuration for the ShrimPK ROS2 bridge.
///
/// Loaded from a TOML file supplied via `--config <path>`.
///
/// # Example TOML
/// ```toml
/// daemon_url = "http://localhost:11435"
///
/// [[topics]]
/// name = "/rosout"
/// msg_type = "String"
///
/// [[topics]]
/// name = "/camera/image_raw"
/// msg_type = "Image"
/// label = "robot-camera"
///
/// [[topics]]
/// name = "/robot_description"
/// msg_type = "String"
/// label = "urdf-desc"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// ShrimPK daemon base URL (default: `http://localhost:11435`).
    pub daemon_url: String,

    /// ROS2 topics to subscribe, each mapped to a message type.
    pub topics: Vec<TopicConfig>,

    /// Optional bearer token for daemon authentication (future use).
    pub auth_token: Option<String>,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            daemon_url: "http://localhost:11435".to_string(),
            topics: vec![],
            auth_token: None,
        }
    }
}

/// Configuration for a single subscribed ROS2 topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicConfig {
    /// Topic name, e.g. `"/camera/image_raw"`.
    pub name: String,

    /// Determines how the raw message is deserialized and which memory
    /// modality is used.
    pub msg_type: MsgType,

    /// Optional hint stored alongside the memory (e.g. `"robot-camera"`).
    pub label: Option<String>,
}

/// ROS2 message type selector — controls both deserialization and the
/// ShrimPK memory modality used.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MsgType {
    /// `std_msgs/String` → text memory.
    String,
    /// `sensor_msgs/Image` → vision memory (RGB bytes converted to PNG).
    Image,
    /// `audio_common_msgs/AudioData` → speech memory (PCM f32 samples).
    Audio,
    /// `geometry_msgs/PoseStamped` → text memory formatted as
    /// `"position: x=1.2 y=3.4 z=0.0"`.
    Pose,
}

/// A single recorded message entry in a JSON replay log.
///
/// Replay log format (JSON array):
/// ```json
/// [
///   { "topic": "/rosout", "msg_type": "String", "label": null,
///     "payload": { "text": "hello world" } },
///   { "topic": "/camera/image_raw", "msg_type": "Image", "label": "cam",
///     "payload": { "width": 4, "height": 4,
///                  "rgb_bytes_base64": "<base64>" } }
/// ]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEntry {
    /// Original ROS2 topic name (informational).
    pub topic: String,
    /// Message type used for deserialization.
    pub msg_type: MsgType,
    /// Optional label hint.
    pub label: Option<String>,
    /// Payload — shape depends on `msg_type`.
    pub payload: serde_json::Value,
}
