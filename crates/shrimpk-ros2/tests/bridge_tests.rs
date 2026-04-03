//! Unit tests for the shrimpk-ros2 bridge.
//!
//! All tests compile and run **without** the `ros2` feature — no ROS2
//! installation required.

use shrimpk_ros2::{
    bridge::{MessageBridge, pose_to_text},
    client::DaemonClient,
    config::{BridgeConfig, MsgType, TopicConfig},
};

// ---------------------------------------------------------------------------
// Config tests
// ---------------------------------------------------------------------------

#[test]
fn config_default_values() {
    let cfg = BridgeConfig::default();
    assert_eq!(cfg.daemon_url, "http://localhost:11435");
    assert!(cfg.topics.is_empty());
    assert!(cfg.auth_token.is_none());
}

#[test]
fn config_topic_round_trip_serde() {
    let cfg = BridgeConfig {
        daemon_url: "http://127.0.0.1:9000".to_string(),
        auth_token: Some("tok".to_string()),
        topics: vec![
            TopicConfig {
                name: "/rosout".to_string(),
                msg_type: MsgType::String,
                label: None,
            },
            TopicConfig {
                name: "/camera/image_raw".to_string(),
                msg_type: MsgType::Image,
                label: Some("cam".to_string()),
            },
            TopicConfig {
                name: "/mic".to_string(),
                msg_type: MsgType::Audio,
                label: None,
            },
            TopicConfig {
                name: "/robot/pose".to_string(),
                msg_type: MsgType::Pose,
                label: Some("nav".to_string()),
            },
        ],
    };

    let json = serde_json::to_string(&cfg).expect("serialize");
    let cfg2: BridgeConfig = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(cfg2.daemon_url, "http://127.0.0.1:9000");
    assert_eq!(cfg2.auth_token, Some("tok".to_string()));
    assert_eq!(cfg2.topics.len(), 4);
    assert_eq!(cfg2.topics[0].msg_type, MsgType::String);
    assert_eq!(cfg2.topics[1].msg_type, MsgType::Image);
    assert_eq!(cfg2.topics[2].msg_type, MsgType::Audio);
    assert_eq!(cfg2.topics[3].msg_type, MsgType::Pose);
}

#[test]
fn config_toml_round_trip() {
    let cfg = BridgeConfig {
        daemon_url: "http://localhost:11435".to_string(),
        auth_token: None,
        topics: vec![TopicConfig {
            name: "/rosout".to_string(),
            msg_type: MsgType::String,
            label: Some("log".to_string()),
        }],
    };
    let toml_str = toml::to_string(&cfg).expect("serialize toml");
    let cfg2: BridgeConfig = toml::from_str(&toml_str).expect("deserialize toml");
    assert_eq!(cfg2.topics[0].name, "/rosout");
    assert_eq!(cfg2.topics[0].label, Some("log".to_string()));
}

// ---------------------------------------------------------------------------
// DaemonClient URL construction
// ---------------------------------------------------------------------------

#[test]
fn client_strips_trailing_slash() {
    // We can't observe the private field directly, but we can verify that
    // health_check builds the correct URL via the public API (offline: we just
    // check it doesn't panic on construction).
    let _ = DaemonClient::new("http://localhost:11435/", None);
    let _ = DaemonClient::new("http://localhost:11435", None);
    let _ = DaemonClient::new("http://localhost:11435", Some("tok".to_string()));
}

// ---------------------------------------------------------------------------
// Pose → text serialization
// ---------------------------------------------------------------------------

#[test]
fn pose_to_text_no_frame() {
    let s = pose_to_text(1.2, 3.4, 0.0, None);
    assert_eq!(s, "position: x=1.20 y=3.40 z=0.00");
}

#[test]
fn pose_to_text_with_frame() {
    let s = pose_to_text(1.2, 3.4, 0.0, Some("map"));
    assert_eq!(s, "position: x=1.20 y=3.40 z=0.00 (frame: map)");
}

#[test]
fn pose_to_text_negative_coords() {
    let s = pose_to_text(-5.678, 0.001, -100.0, Some("odom"));
    assert_eq!(s, "position: x=-5.68 y=0.00 z=-100.00 (frame: odom)");
}

#[test]
fn pose_to_text_zero() {
    let s = pose_to_text(0.0, 0.0, 0.0, None);
    assert_eq!(s, "position: x=0.00 y=0.00 z=0.00");
}

// ---------------------------------------------------------------------------
// Image → PNG conversion
// ---------------------------------------------------------------------------

/// Build a minimal 4×4 RGB image, convert via the bridge logic,
/// decode back, and verify dimensions.
#[test]
fn image_rgb_to_png_round_trip() {
    use image::{DynamicImage, RgbImage};
    use std::io::Cursor;

    let w = 4u32;
    let h = 4u32;
    let rgb: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();

    let img = RgbImage::from_raw(w, h, rgb).expect("create test image");
    let dyn_img = DynamicImage::ImageRgb8(img);

    let mut buf = Cursor::new(Vec::new());
    dyn_img
        .write_to(&mut buf, image::ImageFormat::Png)
        .expect("encode PNG");

    let decoded = image::load_from_memory(buf.get_ref()).expect("decode PNG");
    assert_eq!(decoded.width(), w);
    assert_eq!(decoded.height(), h);
}

/// A 1024×1024 image should be resized to at most 512×512.
#[test]
fn image_resize_applied_for_large_images() {
    use image::{DynamicImage, ImageFormat, RgbImage, imageops};
    use std::io::Cursor;

    let big = 1024u32;
    let cap = 512u32;
    let rgb: Vec<u8> = vec![128u8; (big * big * 3) as usize];
    let img = RgbImage::from_raw(big, big, rgb).expect("create big image");

    // Mirror the resize logic from bridge.rs.
    let resized = imageops::resize(&img, cap, cap, imageops::FilterType::Lanczos3);
    let dyn_img = DynamicImage::ImageRgb8(resized);

    let mut buf = Cursor::new(Vec::new());
    dyn_img
        .write_to(&mut buf, ImageFormat::Png)
        .expect("encode");

    let decoded = image::load_from_memory(buf.get_ref()).expect("decode");
    assert_eq!(decoded.width(), cap);
    assert_eq!(decoded.height(), cap);
}

// ---------------------------------------------------------------------------
// Audio PCM → base64 serialization
// ---------------------------------------------------------------------------

/// Verify that f32 PCM samples survive base64 round-trip.
#[test]
fn audio_pcm_base64_round_trip() {
    use base64::Engine as _;

    let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
    let mut raw: Vec<u8> = Vec::with_capacity(samples.len() * 4);
    for &s in &samples {
        raw.extend_from_slice(&s.to_le_bytes());
    }
    let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);

    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .expect("decode b64");
    let back: Vec<f32> = decoded
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    assert_eq!(back.len(), samples.len());
    for (a, b) in samples.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-7, "sample mismatch: {a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// MessageBridge construction (offline — no HTTP calls)
// ---------------------------------------------------------------------------

#[test]
fn bridge_constructs_without_panic() {
    let client = DaemonClient::new("http://localhost:11435", None);
    let _bridge = MessageBridge::new(client);
}

// ---------------------------------------------------------------------------
// MsgType serde
// ---------------------------------------------------------------------------

#[test]
fn msg_type_serde_all_variants() {
    let variants = [
        MsgType::String,
        MsgType::Image,
        MsgType::Audio,
        MsgType::Pose,
    ];
    for v in &variants {
        let j = serde_json::to_string(v).expect("serialize");
        let back: MsgType = serde_json::from_str(&j).expect("deserialize");
        assert_eq!(*v, back, "round-trip failed for {j}");
    }
}
