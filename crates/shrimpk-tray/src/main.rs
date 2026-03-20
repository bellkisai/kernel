//! ShrimPK System Tray — manage the daemon from your taskbar.
//!
//! Like Ollama's tray icon: shows status, lets you stop the daemon,
//! view stats, copy port, and open the data directory.

use shrimpk_core::config;
use std::process::Command;
use tray_icon::TrayIconBuilder;
use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};

const DAEMON_PORT: u16 = 11435;

fn load_icon() -> tray_icon::Icon {
    let icon_bytes = include_bytes!("../assets/icon.png");
    let img = image::load_from_memory(icon_bytes).expect("Failed to load icon");
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    tray_icon::Icon::from_rgba(rgba.into_raw(), w, h).expect("Failed to create icon")
}

fn daemon_running() -> bool {
    std::net::TcpStream::connect_timeout(
        &format!("127.0.0.1:{DAEMON_PORT}").parse().unwrap(),
        std::time::Duration::from_millis(200),
    )
    .is_ok()
}

fn get_stats() -> String {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        match reqwest::get(format!("http://127.0.0.1:{DAEMON_PORT}/api/stats")).await {
            Ok(resp) => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let memories = json["total_memories"].as_u64().unwrap_or(0);
                    let queries = json["total_echo_queries"].as_u64().unwrap_or(0);
                    format!("Memories: {memories} | Queries: {queries}")
                } else {
                    "Stats unavailable".into()
                }
            }
            Err(_) => "Daemon not responding".into(),
        }
    })
}

fn stop_daemon() {
    let pid_path = config::config_dir().join("daemon.pid");
    if let Ok(content) = std::fs::read_to_string(&pid_path) {
        let pid: Option<u32> = content
            .lines()
            .find(|l| l.starts_with("pid="))
            .and_then(|l| l.strip_prefix("pid="))
            .and_then(|s| s.parse().ok());
        if let Some(pid) = pid {
            #[cfg(target_os = "windows")]
            {
                let _ = Command::new("taskkill")
                    .args(["/F", "/PID", &pid.to_string()])
                    .output();
            }
            #[cfg(not(target_os = "windows"))]
            {
                let _ = Command::new("kill").arg(pid.to_string()).output();
            }
        }
    }
}

fn open_data_dir() {
    let dir = config::config_dir();
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("explorer").arg(dir).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(dir).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open").arg(dir).spawn();
    }
}

fn main() {
    // Menu items
    let status_item = MenuItem::new("ShrimPK — Checking...", false, None);
    let stats_item = MenuItem::new("Show Stats", true, None);
    let copy_port_item = MenuItem::new(format!("Copy Port ({DAEMON_PORT})"), true, None);
    let open_dir_item = MenuItem::new("Open Data Directory", true, None);
    let stop_item = MenuItem::new("Stop Daemon", true, None);
    let quit_item = MenuItem::new("Quit Tray", true, None);

    // Build menu
    let menu = Menu::new();
    let _ = menu.append(&status_item);
    let _ = menu.append(&PredefinedMenuItem::separator());
    let _ = menu.append(&stats_item);
    let _ = menu.append(&copy_port_item);
    let _ = menu.append(&open_dir_item);
    let _ = menu.append(&PredefinedMenuItem::separator());
    let _ = menu.append(&stop_item);
    let _ = menu.append(&quit_item);

    // Create tray icon
    let icon = load_icon();
    let _tray = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("ShrimPK Echo Memory")
        .with_icon(icon)
        .build()
        .expect("Failed to create tray icon");

    // Update status
    let running = daemon_running();
    let status_text = if running {
        "ShrimPK — Running"
    } else {
        "ShrimPK — Daemon Not Running"
    };
    status_item.set_text(status_text);

    // Save menu item IDs for event matching
    let stats_id = stats_item.id().clone();
    let copy_port_id = copy_port_item.id().clone();
    let open_dir_id = open_dir_item.id().clone();
    let stop_id = stop_item.id().clone();
    let quit_id = quit_item.id().clone();

    // Native event loop
    let event_loop = MenuEvent::receiver();

    println!("[shrimpk-tray] Tray icon active. Right-click to manage daemon.");

    loop {
        // Check for menu events (non-blocking with timeout)
        if let Ok(event) = event_loop.recv_timeout(std::time::Duration::from_secs(5)) {
            if event.id == stats_id {
                let stats = get_stats();
                // Update the status item with live stats
                status_item.set_text(format!("ShrimPK — {stats}"));
            } else if event.id == copy_port_id {
                // Copy port to clipboard
                #[cfg(target_os = "windows")]
                {
                    let _ = Command::new("cmd")
                        .args(["/C", &format!("echo {DAEMON_PORT}| clip")])
                        .output();
                }
                println!("[shrimpk-tray] Port {DAEMON_PORT} copied to clipboard");
            } else if event.id == open_dir_id {
                open_data_dir();
            } else if event.id == stop_id {
                stop_daemon();
                status_item.set_text("ShrimPK — Stopped");
                println!("[shrimpk-tray] Daemon stopped.");
            } else if event.id == quit_id {
                println!("[shrimpk-tray] Quitting.");
                break;
            }
        }

        // Periodic status refresh (every 5s via the recv timeout)
        let running = daemon_running();
        let text = if running {
            "ShrimPK — Running"
        } else {
            "ShrimPK — Not Running"
        };
        status_item.set_text(text);
    }
}
