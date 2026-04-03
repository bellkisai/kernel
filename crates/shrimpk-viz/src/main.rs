//! ShrimPK Viz — GraphRAG knowledge graph visualization.
//!
//! Standalone Tauri v2 desktop app that connects to the ShrimPK daemon
//! HTTP API (localhost:11435) to visualize the memory graph.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running ShrimPK Viz");
}
