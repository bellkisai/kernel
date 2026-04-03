//! Live ROS2 node — only compiled when the `ros2` feature is enabled.
//!
//! Requires a working ROS2 installation (Humble or later) and the `r2r` crate.
//! To build: `cargo build --features ros2`
//!
//! # Design
//!
//! One `r2r::Node` is created per bridge process.  For each configured topic,
//! an async task is spawned that:
//! 1. Creates a typed subscriber via `r2r`.
//! 2. Receives messages in a loop.
//! 3. Calls the corresponding [`MessageBridge`] handler.
//!
//! The node is driven by a dedicated `r2r::SpinOnce` loop on the main thread
//! while the subscriber tasks run on the Tokio runtime.

use anyhow::Result;
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::bridge::MessageBridge;
use crate::config::{BridgeConfig, MsgType};

/// Initialise a ROS2 context, create a node, and subscribe to all configured
/// topics.  Blocks until the process is interrupted.
pub async fn run(cfg: BridgeConfig, bridge: Arc<MessageBridge>) -> Result<()> {
    use r2r::QosProfile;

    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "shrimpk_ros2_bridge", "")?;
    info!("ROS2 node 'shrimpk_ros2_bridge' created");

    for topic_cfg in cfg.topics {
        let bridge = Arc::clone(&bridge);
        let label = topic_cfg.label.clone();
        let name = topic_cfg.name.clone();

        match topic_cfg.msg_type {
            MsgType::String => {
                let mut sub =
                    node.subscribe::<r2r::std_msgs::msg::String>(&name, QosProfile::default())?;
                info!(topic = %name, "subscribed (String)");
                tokio::spawn(async move {
                    while let Some(msg) = futures_util::StreamExt::next(&mut sub).await {
                        if let Err(e) = bridge.handle_string(&msg.data, label.as_deref()).await {
                            error!(topic = %name, err = %e, "handle_string failed");
                        }
                    }
                    warn!(topic = %name, "String subscriber stream ended");
                });
            }
            MsgType::Image => {
                let mut sub =
                    node.subscribe::<r2r::sensor_msgs::msg::Image>(&name, QosProfile::default())?;
                info!(topic = %name, "subscribed (Image)");
                tokio::spawn(async move {
                    while let Some(msg) = futures_util::StreamExt::next(&mut sub).await {
                        let w = msg.width;
                        let h = msg.height;
                        // sensor_msgs/Image data is raw; assume RGB8 encoding.
                        if let Err(e) = bridge
                            .handle_image_rgb(&msg.data, w, h, label.as_deref())
                            .await
                        {
                            error!(topic = %name, err = %e, "handle_image_rgb failed");
                        }
                    }
                    warn!(topic = %name, "Image subscriber stream ended");
                });
            }
            MsgType::Audio => {
                // audio_common_msgs/AudioData carries raw bytes; assume f32 LE PCM at 16 kHz.
                let mut sub = node.subscribe::<r2r::audio_common_msgs::msg::AudioData>(
                    &name,
                    QosProfile::default(),
                )?;
                info!(topic = %name, "subscribed (Audio)");
                tokio::spawn(async move {
                    while let Some(msg) = futures_util::StreamExt::next(&mut sub).await {
                        // Convert raw bytes → f32 samples (little-endian).
                        let bytes = &msg.data;
                        if bytes.len() % 4 != 0 {
                            warn!(topic = %name, "audio payload not aligned to 4 bytes — skipping");
                            continue;
                        }
                        let pcm: Vec<f32> = bytes
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect();
                        if let Err(e) = bridge
                            .handle_audio_pcm(&pcm, 16_000, label.as_deref())
                            .await
                        {
                            error!(topic = %name, err = %e, "handle_audio_pcm failed");
                        }
                    }
                    warn!(topic = %name, "Audio subscriber stream ended");
                });
            }
            MsgType::Pose => {
                let mut sub = node.subscribe::<r2r::geometry_msgs::msg::PoseStamped>(
                    &name,
                    QosProfile::default(),
                )?;
                info!(topic = %name, "subscribed (Pose)");
                tokio::spawn(async move {
                    while let Some(msg) = futures_util::StreamExt::next(&mut sub).await {
                        let p = &msg.pose.position;
                        let frame = &msg.header.frame_id;
                        let fid = if frame.is_empty() {
                            None
                        } else {
                            Some(frame.as_str())
                        };
                        if let Err(e) = bridge
                            .handle_pose(p.x, p.y, p.z, fid, label.as_deref())
                            .await
                        {
                            error!(topic = %name, err = %e, "handle_pose failed");
                        }
                    }
                    warn!(topic = %name, "Pose subscriber stream ended");
                });
            }
        }
    }

    // Spin the ROS2 node at 100 Hz until interrupted.
    loop {
        node.spin_once(std::time::Duration::from_millis(10));
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}
