//! `shrimpk-ros2` binary entry point.
//!
//! ## Without `ros2` feature (default)
//! Runs as a **replay tool**: reads a JSON log file produced by a previous
//! ROS2 session and replays each recorded message to the ShrimPK daemon.
//! Useful for testing and offline analysis without a live robot.
//!
//! ## With `ros2` feature
//! Connects to a live ROS2 network (via `r2r`), subscribes to the configured
//! topics, and streams incoming sensor data to the daemon in real-time.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info};

use shrimpk_ros2::{
    bridge::MessageBridge,
    client::DaemonClient,
    config::{BridgeConfig, MsgType, ReplayEntry},
};

/// ROS2 bridge for ShrimPK Echo Memory.
///
/// Without --features ros2: replay a JSON log to the daemon.
/// With    --features ros2: connect to a live ROS2 network.
#[derive(Parser)]
#[command(
    name = "shrimpk-ros2",
    version,
    about = "ROS2 bridge for ShrimPK Echo Memory"
)]
struct Args {
    /// ShrimPK daemon URL.
    #[arg(short, long, default_value = "http://localhost:11435")]
    daemon: String,

    /// TOML config file specifying topics to subscribe.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Replay a JSON log file (no-ros2 mode only).
    #[arg(long)]
    replay: Option<PathBuf>,

    /// Check daemon connectivity and exit.
    #[arg(long)]
    health_check: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Minimal tracing setup — INFO by default, override with RUST_LOG.
    // Use EnvFilter with a default directive to avoid unsafe set_var.
    use tracing_subscriber::EnvFilter;
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("shrimpk_ros2=info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = Args::parse();

    // Load config (TOML file or default).
    let mut cfg = if let Some(path) = &args.config {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read config file: {}", path.display()))?;
        toml::from_str::<BridgeConfig>(&raw)
            .with_context(|| format!("Invalid TOML in: {}", path.display()))?
    } else {
        BridgeConfig::default()
    };

    // CLI --daemon overrides config file value.
    cfg.daemon_url = args.daemon.clone();

    let client = DaemonClient::new(&cfg.daemon_url, cfg.auth_token.clone());
    let bridge = Arc::new(MessageBridge::new(client));

    // -- Health check mode --
    if args.health_check {
        let ok = DaemonClient::new(&cfg.daemon_url, cfg.auth_token.clone())
            .health_check()
            .await
            .context("Health check request failed")?;
        if ok {
            println!("ShrimPK daemon at {} is reachable.", cfg.daemon_url);
        } else {
            eprintln!("ShrimPK daemon at {} returned non-200.", cfg.daemon_url);
            std::process::exit(1);
        }
        return Ok(());
    }

    // -- Feature dispatch --
    #[cfg(feature = "ros2")]
    {
        info!("Starting live ROS2 bridge (r2r)");
        if cfg.topics.is_empty() {
            warn!("No topics configured — specify --config <file>");
        }
        shrimpk_ros2::node::run(cfg, bridge).await?;
    }

    #[cfg(not(feature = "ros2"))]
    {
        run_no_ros2(args, cfg, bridge).await?;
    }

    Ok(())
}

/// Entry point when the crate is compiled **without** the `ros2` feature.
#[cfg(not(feature = "ros2"))]
async fn run_no_ros2(args: Args, _cfg: BridgeConfig, bridge: Arc<MessageBridge>) -> Result<()> {
    if let Some(replay_path) = args.replay {
        info!(path = %replay_path.display(), "Replaying JSON log");
        replay_log(replay_path, bridge).await
    } else {
        eprintln!(
            "shrimpk-ros2: ROS2 bridge not compiled.\n\
             \n\
             To connect to a live ROS2 network, rebuild with:\n\
             \n  cargo build -p shrimpk-ros2 --features ros2\n\
             \n\
             Without ROS2, use --replay <log.json> to replay a recorded session,\n\
             or --health-check to verify daemon connectivity."
        );
        std::process::exit(1);
    }
}

/// Replay a JSON log file — feed each entry through the bridge.
#[cfg(not(feature = "ros2"))]
async fn replay_log(path: PathBuf, bridge: Arc<MessageBridge>) -> Result<()> {
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("Cannot read replay log: {}", path.display()))?;

    let entries: Vec<ReplayEntry> = serde_json::from_str(&raw)
        .context("Replay log must be a JSON array of ReplayEntry objects")?;

    info!(count = entries.len(), "Starting replay");
    let mut ok = 0usize;
    let mut err = 0usize;

    for entry in &entries {
        let result = dispatch_replay_entry(&bridge, entry).await;
        match result {
            Ok(()) => ok += 1,
            Err(e) => {
                error!(topic = %entry.topic, err = %e, "Replay entry failed");
                err += 1;
            }
        }
    }

    info!(ok, err, "Replay complete");
    if err > 0 {
        anyhow::bail!("{err} replay entries failed");
    }
    Ok(())
}

/// Dispatch a single replay entry to the correct bridge handler.
#[cfg(not(feature = "ros2"))]
async fn dispatch_replay_entry(bridge: &MessageBridge, entry: &ReplayEntry) -> Result<()> {
    let label = entry.label.as_deref();
    match entry.msg_type {
        MsgType::String => {
            let text = entry.payload["text"]
                .as_str()
                .context("ReplayEntry.payload.text must be a string")?;
            bridge.handle_string(text, label).await
        }
        MsgType::Image => {
            let b64 = entry.payload["rgb_bytes_base64"]
                .as_str()
                .context("ReplayEntry.payload.rgb_bytes_base64 must be a string")?;
            let width = entry.payload["width"]
                .as_u64()
                .context("ReplayEntry.payload.width must be an integer")?
                as u32;
            let height = entry.payload["height"]
                .as_u64()
                .context("ReplayEntry.payload.height must be an integer")?
                as u32;
            use base64::Engine as _;
            let rgb_bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .context("Invalid base64 in rgb_bytes_base64")?;
            bridge
                .handle_image_rgb(&rgb_bytes, width, height, label)
                .await
        }
        MsgType::Audio => {
            let samples = entry.payload["samples"]
                .as_array()
                .context("ReplayEntry.payload.samples must be an array")?;
            let sample_rate = entry.payload["sample_rate"].as_u64().unwrap_or(16_000) as u32;
            let pcm: Vec<f32> = samples
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            bridge.handle_audio_pcm(&pcm, sample_rate, label).await
        }
        MsgType::Pose => {
            let x = entry.payload["x"].as_f64().unwrap_or(0.0);
            let y = entry.payload["y"].as_f64().unwrap_or(0.0);
            let z = entry.payload["z"].as_f64().unwrap_or(0.0);
            let frame = entry.payload["frame_id"].as_str();
            bridge.handle_pose(x, y, z, frame, label).await
        }
    }
}
