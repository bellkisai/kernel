//! Integration tests for the ShrimPK daemon.
//!
//! These tests spawn a daemon process, make HTTP requests, and verify responses.
//! Marked #[ignore] because they require the binary to be built and a free port.

use std::process::{Child, Command};
use std::time::Duration;

const TEST_PORT: u16 = 11436; // Use different port from default to avoid conflicts

fn spawn_daemon() -> Child {
    let binary = if cfg!(target_os = "windows") {
        "target/debug/shrimpk-daemon.exe"
    } else {
        "target/debug/shrimpk-daemon"
    };

    Command::new(binary)
        .args(["--port", &TEST_PORT.to_string()])
        .env(
            "SHRIMPK_DATA_DIR",
            std::env::temp_dir().join("shrimpk-test"),
        )
        .spawn()
        .expect("Failed to spawn daemon")
}

fn wait_for_daemon() -> bool {
    for _ in 0..30 {
        if std::net::TcpStream::connect_timeout(
            &format!("127.0.0.1:{TEST_PORT}").parse().unwrap(),
            Duration::from_millis(100),
        )
        .is_ok()
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    false
}

fn kill_daemon(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn base_url() -> String {
    format!("http://127.0.0.1:{TEST_PORT}")
}

#[test]
#[ignore = "requires built daemon binary"]
fn health_endpoint_returns_ok() {
    let child = spawn_daemon();
    assert!(wait_for_daemon(), "Daemon failed to start");

    let resp = reqwest::blocking::get(format!("{}/health", base_url())).unwrap();
    assert!(resp.status().is_success());

    let json: serde_json::Value = resp.json().unwrap();
    assert_eq!(json["status"], "ok");
    assert!(json["version"].is_string());

    kill_daemon(child);
}

#[test]
#[ignore = "requires built daemon binary"]
fn store_and_echo_roundtrip() {
    let child = spawn_daemon();
    assert!(wait_for_daemon(), "Daemon failed to start");

    let client = reqwest::blocking::Client::new();

    // Store
    let store_resp = client
        .post(format!("{}/api/store", base_url()))
        .json(&serde_json::json!({"text": "Integration test memory", "source": "test"}))
        .send()
        .unwrap();
    assert!(store_resp.status().is_success());
    let store_json: serde_json::Value = store_resp.json().unwrap();
    assert!(store_json["memory_id"].is_string());

    // Echo
    let echo_resp = client
        .post(format!("{}/api/echo", base_url()))
        .json(&serde_json::json!({"query": "integration test", "max_results": 5}))
        .send()
        .unwrap();
    assert!(echo_resp.status().is_success());
    let echo_json: serde_json::Value = echo_resp.json().unwrap();
    assert!(echo_json["count"].as_u64().unwrap() > 0);

    kill_daemon(child);
}

#[test]
#[ignore = "requires built daemon binary"]
fn stats_returns_memory_count() {
    let child = spawn_daemon();
    assert!(wait_for_daemon(), "Daemon failed to start");

    let resp = reqwest::blocking::get(format!("{}/api/stats", base_url())).unwrap();
    assert!(resp.status().is_success());

    let json: serde_json::Value = resp.json().unwrap();
    assert!(json["total_memories"].is_number());
    assert!(json["max_capacity"].is_number());

    kill_daemon(child);
}

#[test]
#[ignore = "requires built daemon binary"]
fn store_and_forget_cycle() {
    let child = spawn_daemon();
    assert!(wait_for_daemon(), "Daemon failed to start");

    let client = reqwest::blocking::Client::new();

    // Store
    let store_resp = client
        .post(format!("{}/api/store", base_url()))
        .json(&serde_json::json!({"text": "Memory to forget", "source": "test"}))
        .send()
        .unwrap();
    let store_json: serde_json::Value = store_resp.json().unwrap();
    let memory_id = store_json["memory_id"].as_str().unwrap();

    // Forget
    let forget_resp = client
        .delete(format!("{}/api/memories/{}", base_url(), memory_id))
        .send()
        .unwrap();
    assert!(forget_resp.status().is_success());

    kill_daemon(child);
}

#[test]
#[ignore = "requires built daemon binary"]
fn config_show_returns_json() {
    let child = spawn_daemon();
    assert!(wait_for_daemon(), "Daemon failed to start");

    let resp = reqwest::blocking::get(format!("{}/api/config", base_url())).unwrap();
    assert!(resp.status().is_success());

    let json: serde_json::Value = resp.json().unwrap();
    assert!(json["max_memories"].is_number());
    assert!(json["similarity_threshold"].is_number());

    kill_daemon(child);
}

#[test]
#[ignore = "requires built daemon binary"]
fn list_memories_returns_array() {
    let child = spawn_daemon();
    assert!(wait_for_daemon(), "Daemon failed to start");

    let resp = reqwest::blocking::get(format!("{}/api/memories?limit=10", base_url())).unwrap();
    assert!(resp.status().is_success());

    let json: serde_json::Value = resp.json().unwrap();
    assert!(json["memories"].is_array());
    assert!(json["total"].is_number());

    kill_daemon(child);
}

#[test]
#[ignore = "requires built daemon binary"]
fn invalid_uuid_returns_400() {
    let child = spawn_daemon();
    assert!(wait_for_daemon(), "Daemon failed to start");

    let client = reqwest::blocking::Client::new();
    let resp = client
        .delete(format!("{}/api/memories/not-a-uuid", base_url()))
        .send()
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);

    kill_daemon(child);
}

#[test]
#[ignore = "requires built daemon binary"]
fn persist_endpoint_works() {
    let child = spawn_daemon();
    assert!(wait_for_daemon(), "Daemon failed to start");

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{}/api/persist", base_url()))
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    let json: serde_json::Value = resp.json().unwrap();
    assert_eq!(json["persisted"], true);

    kill_daemon(child);
}

#[test]
#[ignore = "requires built daemon binary"]
fn consolidate_endpoint_works() {
    let child = spawn_daemon();
    assert!(wait_for_daemon(), "Daemon failed to start");

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{}/api/consolidate", base_url()))
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    let json: serde_json::Value = resp.json().unwrap();
    assert!(json["duration_ms"].is_number());

    kill_daemon(child);
}

#[test]
#[ignore = "requires built daemon binary"]
fn auth_token_required_when_set() {
    // Start daemon with auth token
    let binary = if cfg!(target_os = "windows") {
        "target/debug/shrimpk-daemon.exe"
    } else {
        "target/debug/shrimpk-daemon"
    };

    let child = Command::new(binary)
        .args(["--port", "11437"])
        .env(
            "SHRIMPK_DATA_DIR",
            std::env::temp_dir().join("shrimpk-auth-test"),
        )
        .env("SHRIMPK_AUTH_TOKEN", "test-secret-123")
        .spawn()
        .expect("Failed to spawn daemon");

    // Wait for auth daemon
    for _ in 0..30 {
        if std::net::TcpStream::connect_timeout(
            &"127.0.0.1:11437".parse().unwrap(),
            Duration::from_millis(100),
        )
        .is_ok()
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    let client = reqwest::blocking::Client::new();

    // Without token → 401
    let resp = client.get("http://127.0.0.1:11437/health").send().unwrap();
    assert_eq!(resp.status().as_u16(), 401);

    // With token → 200
    let resp = client
        .get("http://127.0.0.1:11437/health")
        .header("Authorization", "Bearer test-secret-123")
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    kill_daemon(child);
}
