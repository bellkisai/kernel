#![windows_subsystem = "windows"]
//! ShrimPK System Tray — manage the daemon from your taskbar.
//!
//! Like Ollama's tray icon: shows status, lets you stop the daemon,
//! view stats, copy port, and open the data directory.
//!
//! ## Windows message-pump safety
//! All HTTP calls use `reqwest::blocking` on a spawned thread that sends
//! results back through an `mpsc` channel.  The main thread drains that
//! channel alongside `MenuEvent::receiver()` after every Win32 message,
//! so the message pump is never blocked by network I/O.
//!
//! `MenuItem` uses `Rc` internally (not `Send`), so it never leaves the
//! main thread — only plain `String` values cross the thread boundary.

use shrimpk_core::config;
use std::process::Command;
use std::sync::mpsc;
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

/// Find shrimpk-daemon.exe next to the current binary and launch it hidden.
fn ensure_daemon_running() {
    if daemon_running() {
        return;
    }

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return,
    };

    let daemon_path = exe
        .parent()
        .map(|dir| dir.join("shrimpk-daemon.exe"))
        .unwrap_or_default();

    if !daemon_path.exists() {
        eprintln!(
            "[shrimpk-tray] daemon binary not found at {:?}",
            daemon_path
        );
        return;
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW = 0x08000000
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        match Command::new(&daemon_path)
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
        {
            Ok(_) => eprintln!("[shrimpk-tray] launched daemon from {:?}", daemon_path),
            Err(e) => eprintln!("[shrimpk-tray] failed to launch daemon: {e}"),
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        match Command::new(&daemon_path).spawn() {
            Ok(_) => eprintln!("[shrimpk-tray] launched daemon from {:?}", daemon_path),
            Err(e) => eprintln!("[shrimpk-tray] failed to launch daemon: {e}"),
        }
    }

    // Give the daemon a moment to start
    std::thread::sleep(std::time::Duration::from_millis(500));
}

/// Fetch stats from the daemon using `reqwest::blocking` (no tokio runtime needed).
/// Safe to call from any thread -- uses synchronous I/O with a 3-second timeout.
fn get_stats_blocking() -> String {
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(_) => return "Stats unavailable (client error)".into(),
    };

    match client
        .get(format!("http://127.0.0.1:{DAEMON_PORT}/api/stats"))
        .send()
    {
        Ok(resp) => match resp.json::<serde_json::Value>() {
            Ok(json) => {
                let memories = json["total_memories"].as_u64().unwrap_or(0);
                let queries = json["total_echo_queries"].as_u64().unwrap_or(0);
                format!("Memories: {memories} | Queries: {queries}")
            }
            Err(_) => "Stats unavailable".into(),
        },
        Err(_) => "Daemon not responding".into(),
    }
}

/// Spawn a background thread that fetches stats and sends the result string
/// back through `tx`.  The main thread reads from the corresponding `rx`
/// and updates the `MenuItem` (which must stay on the main thread).
fn fetch_stats_in_background(tx: &mpsc::Sender<String>) {
    let tx = tx.clone();
    std::thread::spawn(move || {
        let stats = get_stats_blocking();
        let _ = tx.send(stats);
    });
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

fn copy_port_to_clipboard() {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        // Pipe the port number into clip.exe without a trailing newline
        let _ = Command::new("cmd")
            .args(["/C", &format!("echo|set /p={DAEMON_PORT}| clip")])
            .creation_flags(CREATE_NO_WINDOW)
            .output();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("sh")
            .args(["-c", &format!("printf '{DAEMON_PORT}' | pbcopy")])
            .output();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("sh")
            .args([
                "-c",
                &format!("printf '{DAEMON_PORT}' | xclip -selection clipboard"),
            ])
            .output();
    }
}

fn main() {
    // Ensure daemon is running before we show the tray
    ensure_daemon_running();

    // Channel for background threads to send results back to the main thread.
    // The main thread owns all MenuItems (not Send) and updates them here.
    let (stats_tx, stats_rx) = mpsc::channel::<String>();

    // Menu items
    let status_item = MenuItem::new("ShrimPK \u{2014} Checking...", false, None);
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
        "ShrimPK \u{2014} Running"
    } else {
        "ShrimPK \u{2014} Daemon Not Running"
    };
    status_item.set_text(status_text);

    // Save menu item IDs for event matching
    let stats_id = stats_item.id().clone();
    let copy_port_id = copy_port_item.id().clone();
    let open_dir_id = open_dir_item.id().clone();
    let stop_id = stop_item.id().clone();
    let quit_id = quit_item.id().clone();

    // Platform-specific event loop
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::UI::WindowsAndMessaging::*;
        unsafe {
            // Timer fires WM_TIMER every 5 seconds for periodic status checks
            SetTimer(std::ptr::null_mut() as _, 1, 5000, None);

            let mut msg = std::mem::zeroed();
            loop {
                let ret = GetMessageW(&mut msg, std::ptr::null_mut() as _, 0, 0);
                if ret <= 0 {
                    break;
                }
                TranslateMessage(&msg);
                DispatchMessageW(&msg);

                // Drain stats results from background threads
                while let Ok(stats) = stats_rx.try_recv() {
                    status_item.set_text(format!("ShrimPK \u{2014} {stats}"));
                }

                // Drain all pending menu events after each Win32 message
                let mut should_quit = false;
                while let Ok(event) = MenuEvent::receiver().try_recv() {
                    if event.id == stats_id {
                        // Show immediate feedback, then fetch on a background thread
                        status_item.set_text("ShrimPK \u{2014} Fetching stats...");
                        fetch_stats_in_background(&stats_tx);
                    } else if event.id == copy_port_id {
                        copy_port_to_clipboard();
                    } else if event.id == open_dir_id {
                        open_data_dir();
                    } else if event.id == stop_id {
                        stop_daemon();
                        status_item.set_text("ShrimPK \u{2014} Stopped");
                    } else if event.id == quit_id {
                        should_quit = true;
                    }
                }

                if should_quit {
                    break;
                }

                // On WM_TIMER (msg.message == 0x0113), refresh daemon status
                if msg.message == 0x0113 {
                    let running = daemon_running();
                    let text = if running {
                        "ShrimPK \u{2014} Running"
                    } else {
                        "ShrimPK \u{2014} Not Running"
                    };
                    status_item.set_text(text);
                }
            }
        }
    }

    // Fallback for non-Windows platforms: simple recv loop
    #[cfg(not(target_os = "windows"))]
    {
        let event_loop = MenuEvent::receiver();
        loop {
            // Drain stats results from background threads
            while let Ok(stats) = stats_rx.try_recv() {
                status_item.set_text(format!("ShrimPK \u{2014} {stats}"));
            }

            if let Ok(event) = event_loop.recv_timeout(std::time::Duration::from_secs(5)) {
                if event.id == stats_id {
                    status_item.set_text("ShrimPK \u{2014} Fetching stats...");
                    fetch_stats_in_background(&stats_tx);
                } else if event.id == copy_port_id {
                    copy_port_to_clipboard();
                } else if event.id == open_dir_id {
                    open_data_dir();
                } else if event.id == stop_id {
                    stop_daemon();
                    status_item.set_text("ShrimPK \u{2014} Stopped");
                } else if event.id == quit_id {
                    break;
                }
            }

            // Periodic status refresh
            let running = daemon_running();
            let text = if running {
                "ShrimPK \u{2014} Running"
            } else {
                "ShrimPK \u{2014} Not Running"
            };
            status_item.set_text(text);
        }
    }
}
