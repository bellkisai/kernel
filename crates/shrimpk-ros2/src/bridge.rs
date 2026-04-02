//! Message conversion layer — transforms raw robot sensor data into ShrimPK
//! memory store calls.
//!
//! `MessageBridge` is intentionally decoupled from ROS2: it operates on
//! already-deserialized primitives (strings, raw bytes, floats) so that the
//! same logic works in both the live ROS2 node and the JSON replay tool.

use anyhow::Result;
use image::{ImageFormat, RgbImage};
use std::io::Cursor;
use tracing::{debug, instrument};

use crate::client::DaemonClient;

/// Maximum dimension (width or height in pixels) to which images are resized
/// before encoding to PNG for the vision channel.
const MAX_IMAGE_DIM: u32 = 512;

/// Converts raw robot sensor messages into ShrimPK daemon store calls.
pub struct MessageBridge {
    client: DaemonClient,
}

impl MessageBridge {
    /// Wrap an existing [`DaemonClient`].
    pub fn new(client: DaemonClient) -> Self {
        Self { client }
    }

    /// Forward a text string to the text memory channel.
    #[instrument(skip(self, text), fields(len = text.len()))]
    pub async fn handle_string(&self, text: &str, label: Option<&str>) -> Result<()> {
        debug!("handle_string");
        self.client.store_text(text, label).await
    }

    /// Convert raw RGB bytes (from `sensor_msgs/Image`) to PNG and store as a
    /// vision memory.
    ///
    /// The image is resized to at most `512×512` before encoding to keep the
    /// daemon payload small.  Aspect ratio is preserved.
    #[instrument(skip(self, rgb_bytes), fields(w = width, h = height))]
    pub async fn handle_image_rgb(
        &self,
        rgb_bytes: &[u8],
        width: u32,
        height: u32,
        label: Option<&str>,
    ) -> Result<()> {
        debug!("handle_image_rgb: encoding to PNG");

        let img = RgbImage::from_raw(width, height, rgb_bytes.to_vec())
            .ok_or_else(|| anyhow::anyhow!("RGB buffer too small for {width}×{height}"))?;

        // Resize if either dimension exceeds the cap.
        let img = if width > MAX_IMAGE_DIM || height > MAX_IMAGE_DIM {
            let resized =
                image::imageops::resize(&img, MAX_IMAGE_DIM, MAX_IMAGE_DIM, image::imageops::FilterType::Lanczos3);
            image::DynamicImage::ImageRgb8(resized)
        } else {
            image::DynamicImage::ImageRgb8(img)
        };

        let mut png_buf = Cursor::new(Vec::new());
        img.write_to(&mut png_buf, ImageFormat::Png)
            .map_err(|e| anyhow::anyhow!("PNG encode failed: {e}"))?;

        self.client
            .store_image(png_buf.get_ref(), label)
            .await
    }

    /// Forward PCM f32 audio samples to the speech memory channel.
    #[instrument(skip(self, pcm), fields(samples = pcm.len(), rate = sample_rate))]
    pub async fn handle_audio_pcm(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        label: Option<&str>,
    ) -> Result<()> {
        debug!("handle_audio_pcm");
        self.client.store_audio(pcm, sample_rate, label).await
    }

    /// Serialize a 3D pose as a human-readable text memory and store it.
    ///
    /// Output format: `"position: x=1.20 y=3.40 z=0.00"` — concise enough to
    /// fit in a single embedding and easy to search with natural language.
    ///
    /// If `frame_id` is provided the output becomes:
    /// `"position: x=1.20 y=3.40 z=0.00 (frame: map)"`.
    #[instrument(skip(self))]
    pub async fn handle_pose(
        &self,
        x: f64,
        y: f64,
        z: f64,
        frame_id: Option<&str>,
        label: Option<&str>,
    ) -> Result<()> {
        let text = match frame_id {
            Some(frame) => format!("position: x={x:.2} y={y:.2} z={z:.2} (frame: {frame})"),
            None => format!("position: x={x:.2} y={y:.2} z={z:.2}"),
        };
        debug!(pose_text = %text, "handle_pose");
        self.client.store_text(&text, label).await
    }
}

/// Serialize a pose triple to the canonical text representation.
///
/// Extracted as a pure function so it can be unit-tested without any async
/// or HTTP machinery.
pub fn pose_to_text(x: f64, y: f64, z: f64, frame_id: Option<&str>) -> String {
    match frame_id {
        Some(frame) => format!("position: x={x:.2} y={y:.2} z={z:.2} (frame: {frame})"),
        None => format!("position: x={x:.2} y={y:.2} z={z:.2}"),
    }
}
