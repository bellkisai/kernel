//! Interactive TUI for exploring the ShrimPK memory vault.
//!
//! Launch: `shrimpk explore`
//!
//! Two views:
//! - **List**: scrollable memory list sorted by importance/echo count/novelty
//! - **Explore**: single memory card + label-grouped connections, drill into neighbors

use std::collections::HashMap;
use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::prelude::*;
use ratatui::widgets::{Block, List, ListItem, ListState, Paragraph};

use shrimpk_core::{MemoryEntrySummary, MemoryId};
use shrimpk_memory::EchoEngine;

// ─── Types ────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum View {
    List,
    Explore,
}

#[derive(Clone, Copy)]
enum SortMode {
    Importance,
    EchoCount,
    Novelty,
}

impl SortMode {
    fn next(self) -> Self {
        match self {
            Self::Importance => Self::EchoCount,
            Self::EchoCount => Self::Novelty,
            Self::Novelty => Self::Importance,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Importance => "importance",
            Self::EchoCount => "echo count",
            Self::Novelty => "novelty",
        }
    }
}

/// Flattened connection list entry — either a label header or a memory item.
enum FlatConn {
    Header { label: String, count: usize },
    Memory { id: MemoryId, content: String, importance: f32 },
}

/// Saved explore state for back-navigation.
struct ExploreSnapshot {
    idx: usize,
    labels: Vec<String>,
    conns: Vec<FlatConn>,
    selected: Option<usize>,
}

// ─── App State ────────────────────────────────────────────────

struct App {
    view: View,
    memories: Vec<MemoryEntrySummary>,
    list_state: ListState,
    sort_mode: SortMode,
    search: String,
    searching: bool,
    filtered: Vec<usize>,

    // Explore view
    explore_idx: usize,
    explore_labels: Vec<String>,
    flat_conns: Vec<FlatConn>,
    conn_state: ListState,
    history: Vec<ExploreSnapshot>,

    id_map: HashMap<MemoryId, usize>,
    quit: bool,
}

impl App {
    fn new(mut memories: Vec<MemoryEntrySummary>) -> Self {
        // Default sort: importance descending
        memories.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let id_map: HashMap<MemoryId, usize> =
            memories.iter().enumerate().map(|(i, m)| (m.id.clone(), i)).collect();

        let filtered: Vec<usize> = (0..memories.len()).collect();

        let mut list_state = ListState::default();
        if !memories.is_empty() {
            list_state.select(Some(0));
        }

        App {
            view: View::List,
            memories,
            list_state,
            sort_mode: SortMode::Importance,
            search: String::new(),
            searching: false,
            filtered,
            explore_idx: 0,
            explore_labels: Vec::new(),
            flat_conns: Vec::new(),
            conn_state: ListState::default(),
            history: Vec::new(),
            id_map,
            quit: false,
        }
    }

    fn apply_sort(&mut self) {
        self.filtered.sort_by(|&a, &b| {
            let ma = &self.memories[a];
            let mb = &self.memories[b];
            match self.sort_mode {
                SortMode::Importance => mb
                    .importance
                    .partial_cmp(&ma.importance)
                    .unwrap_or(std::cmp::Ordering::Equal),
                SortMode::EchoCount => mb.echo_count.cmp(&ma.echo_count),
                SortMode::Novelty => mb
                    .novelty_score
                    .partial_cmp(&ma.novelty_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });
    }

    fn apply_filter(&mut self) {
        let q = self.search.to_lowercase();
        self.filtered = if q.is_empty() {
            (0..self.memories.len()).collect()
        } else {
            (0..self.memories.len())
                .filter(|&i| self.memories[i].content.to_lowercase().contains(&q))
                .collect()
        };
        self.apply_sort();
        if self.filtered.is_empty() {
            self.list_state.select(None);
        } else {
            self.list_state.select(Some(0));
        }
    }

    fn selected_memory_idx(&self) -> Option<usize> {
        self.list_state
            .selected()
            .and_then(|sel| self.filtered.get(sel).copied())
    }

    /// Selected connection memory index (skips headers).
    fn selected_conn_memory_id(&self) -> Option<MemoryId> {
        self.conn_state.selected().and_then(|sel| {
            if let FlatConn::Memory { id, .. } = &self.flat_conns[sel] {
                Some(id.clone())
            } else {
                None
            }
        })
    }

    async fn enter_explore(&mut self, idx: usize, engine: &EchoEngine) {
        self.explore_idx = idx;
        let id = self.memories[idx].id.clone();

        self.explore_labels.clear();
        self.flat_conns.clear();

        if let Ok(graph) = engine.memory_graph(&id, 5).await {
            self.explore_labels = graph.labels;

            for conn in &graph.connections {
                self.flat_conns.push(FlatConn::Header {
                    label: conn.label.clone(),
                    count: conn.count,
                });
                for top_id in &conn.top_ids {
                    if *top_id == id {
                        continue; // skip self
                    }
                    if let Some(&mem_idx) = self.id_map.get(top_id) {
                        let m = &self.memories[mem_idx];
                        self.flat_conns.push(FlatConn::Memory {
                            id: top_id.clone(),
                            content: trunc(&m.content, 70),
                            importance: m.importance,
                        });
                    }
                }
            }
        }

        self.conn_state = ListState::default();
        // Select first memory item (skip headers)
        let first = self
            .flat_conns
            .iter()
            .position(|c| matches!(c, FlatConn::Memory { .. }));
        self.conn_state.select(first);

        self.view = View::Explore;
    }

    async fn drill_into(&mut self, target_id: MemoryId, engine: &EchoEngine) {
        // Save current state
        let snapshot = ExploreSnapshot {
            idx: self.explore_idx,
            labels: std::mem::take(&mut self.explore_labels),
            conns: std::mem::take(&mut self.flat_conns),
            selected: self.conn_state.selected(),
        };
        self.history.push(snapshot);

        if let Some(&target_idx) = self.id_map.get(&target_id) {
            self.enter_explore(target_idx, engine).await;
        }
    }

    fn go_back(&mut self) {
        if let Some(prev) = self.history.pop() {
            self.explore_idx = prev.idx;
            self.explore_labels = prev.labels;
            self.flat_conns = prev.conns;
            self.conn_state = ListState::default();
            self.conn_state.select(prev.selected);
        } else {
            self.view = View::List;
        }
    }

    /// Move connection cursor to next memory item (skip headers).
    fn conn_next(&mut self) {
        let len = self.flat_conns.len();
        if len == 0 {
            return;
        }
        let start = self.conn_state.selected().map_or(0, |i| i + 1);
        for i in start..len {
            if matches!(self.flat_conns[i], FlatConn::Memory { .. }) {
                self.conn_state.select(Some(i));
                return;
            }
        }
    }

    /// Move connection cursor to previous memory item (skip headers).
    fn conn_prev(&mut self) {
        let cur = match self.conn_state.selected() {
            Some(i) if i > 0 => i,
            _ => return,
        };
        for i in (0..cur).rev() {
            if matches!(self.flat_conns[i], FlatConn::Memory { .. }) {
                self.conn_state.select(Some(i));
                return;
            }
        }
    }
}

// ─── Entry Point ──────────────────────────────────────────────

pub async fn run_explore(engine: &EchoEngine) -> anyhow::Result<()> {
    let summaries = engine.all_entry_summaries().await;

    if summaries.is_empty() {
        println!("Your vault is empty.");
        println!("Store a memory to get started:");
        println!("  shrimpk store \"I prefer Rust for systems programming\"");
        return Ok(());
    }

    let mut app = App::new(summaries);

    // Terminal setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Event loop
    loop {
        terminal.draw(|frame| render(&mut app, frame))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                handle_key(&mut app, key.code, engine).await;
            }
        }

        if app.quit {
            break;
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

    Ok(())
}

// ─── Rendering ────────────────────────────────────────────────

fn render(app: &mut App, frame: &mut Frame) {
    let chunks = Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).split(frame.area());

    match app.view {
        View::List => render_list(app, frame, chunks[0]),
        View::Explore => render_explore(app, frame, chunks[0]),
    }

    render_status(app, frame, chunks[1]);
}

fn render_list(app: &mut App, frame: &mut Frame, area: Rect) {
    let items: Vec<ListItem> = app
        .filtered
        .iter()
        .map(|&idx| {
            let m = &app.memories[idx];
            let bar = importance_bar(m.importance, 8);
            let content = trunc(&m.content, area.width.saturating_sub(22) as usize);

            ListItem::new(Line::from(vec![
                Span::styled(bar.0, Style::new().fg(Color::Green)),
                Span::styled(bar.1, Style::new().fg(Color::DarkGray)),
                Span::raw(" "),
                Span::styled(
                    format!("{:.2}", m.importance),
                    Style::new().fg(Color::Yellow),
                ),
                Span::raw("  "),
                Span::raw(content),
            ]))
        })
        .collect();

    let title = if app.searching {
        format!(
            " Reef [{}/{}] search: {}_ ",
            app.filtered.len(),
            app.memories.len(),
            app.search
        )
    } else {
        format!(
            " Reef [{} memories] sorted: {} ",
            app.filtered.len(),
            app.sort_mode.label()
        )
    };

    let list = List::new(items)
        .block(Block::bordered().title(title).title_style(Style::new().fg(Color::Cyan).bold()))
        .highlight_style(Style::new().reversed())
        .highlight_symbol("> ");

    frame.render_stateful_widget(list, area, &mut app.list_state);
}

fn render_explore(app: &mut App, frame: &mut Frame, area: Rect) {
    let chunks = Layout::vertical([Constraint::Length(7), Constraint::Min(0)]).split(area);

    // ─ Top: Memory Card ─
    let m = &app.memories[app.explore_idx];
    let labels_str = if app.explore_labels.is_empty() {
        "(no labels)".to_string()
    } else {
        app.explore_labels.join("  ")
    };

    let depth = app.history.len();
    let depth_indicator = if depth > 0 {
        format!(" [depth {}]", depth + 1)
    } else {
        String::new()
    };

    let card = Paragraph::new(vec![
        Line::from(Span::styled(
            format!("\"{}\"", trunc(&m.content, area.width.saturating_sub(6) as usize)),
            Style::new().bold(),
        )),
        Line::raw(""),
        Line::from(vec![
            Span::raw("source: "),
            Span::styled(&m.source, Style::new().fg(Color::Cyan)),
            Span::raw("  echoed: "),
            Span::styled(format!("{}x", m.echo_count), Style::new().fg(Color::Yellow)),
            Span::raw("  importance: "),
            Span::styled(format!("{:.2}", m.importance), Style::new().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("category: "),
            Span::styled(format!("{:?}", m.category), Style::new().fg(Color::Magenta)),
            Span::raw("  novelty: "),
            Span::styled(format!("{:.2}", m.novelty_score), Style::new().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("labels: "),
            Span::styled(labels_str, Style::new().fg(Color::Magenta)),
        ]),
    ])
    .block(
        Block::bordered()
            .title(format!(" Exploring{depth_indicator} "))
            .title_style(Style::new().fg(Color::Cyan).bold()),
    );
    frame.render_widget(card, chunks[0]);

    // ─ Bottom: Connections ─
    let conn_items: Vec<ListItem> = app
        .flat_conns
        .iter()
        .map(|c| match c {
            FlatConn::Header { label, count } => ListItem::new(Line::from(vec![
                Span::styled(
                    format!("  {} ({count})", label),
                    Style::new().fg(Color::Cyan).bold(),
                ),
            ])),
            FlatConn::Memory {
                content,
                importance,
                ..
            } => ListItem::new(Line::from(vec![
                Span::raw("    "),
                Span::styled(format!("{importance:.2}"), Style::new().fg(Color::Yellow)),
                Span::raw("  "),
                Span::raw(content.as_str()),
            ])),
        })
        .collect();

    let total_conns: usize = app
        .flat_conns
        .iter()
        .filter(|c| matches!(c, FlatConn::Memory { .. }))
        .count();
    let total_labels = app
        .flat_conns
        .iter()
        .filter(|c| matches!(c, FlatConn::Header { .. }))
        .count();

    let conn_title = format!(" Connections ({total_conns} via {total_labels} labels) ");
    let conn_list = List::new(conn_items)
        .block(
            Block::bordered()
                .title(conn_title)
                .title_style(Style::new().fg(Color::Cyan).bold()),
        )
        .highlight_style(Style::new().reversed())
        .highlight_symbol("> ");

    frame.render_stateful_widget(conn_list, chunks[1], &mut app.conn_state);
}

fn render_status(app: &App, frame: &mut Frame, area: Rect) {
    let text = match app.view {
        View::List if app.searching => {
            " [Esc] cancel  [Enter] confirm  [Backspace] delete".to_string()
        }
        View::List => " [Up/Down] navigate  [Enter] explore  [/] search  [s] sort  [q] quit"
            .to_string(),
        View::Explore => {
            " [Up/Down] select  [Enter] drill in  [Esc] back  [q] quit".to_string()
        }
    };
    let bar = Paragraph::new(text).style(Style::new().fg(Color::White).bg(Color::DarkGray));
    frame.render_widget(bar, area);
}

// ─── Key Handling ─────────────────────────────────────────────

async fn handle_key(app: &mut App, code: KeyCode, engine: &EchoEngine) {
    // Global quit
    if matches!(code, KeyCode::Char('q')) && !app.searching {
        app.quit = true;
        return;
    }

    match app.view {
        View::List if app.searching => match code {
            KeyCode::Esc => {
                app.searching = false;
                app.search.clear();
                app.apply_filter();
            }
            KeyCode::Enter => {
                app.searching = false;
            }
            KeyCode::Backspace => {
                app.search.pop();
                app.apply_filter();
            }
            KeyCode::Char(c) => {
                app.search.push(c);
                app.apply_filter();
            }
            _ => {}
        },
        View::List => match code {
            KeyCode::Up | KeyCode::Char('k') => {
                select_prev(&mut app.list_state);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next(&mut app.list_state, app.filtered.len());
            }
            KeyCode::Enter => {
                if let Some(idx) = app.selected_memory_idx() {
                    app.history.clear();
                    app.enter_explore(idx, engine).await;
                }
            }
            KeyCode::Char('/') => {
                app.searching = true;
            }
            KeyCode::Char('s') => {
                app.sort_mode = app.sort_mode.next();
                app.apply_sort();
            }
            KeyCode::Home => {
                if !app.filtered.is_empty() {
                    app.list_state.select(Some(0));
                }
            }
            KeyCode::End => {
                if !app.filtered.is_empty() {
                    app.list_state.select(Some(app.filtered.len() - 1));
                }
            }
            _ => {}
        },
        View::Explore => match code {
            KeyCode::Up | KeyCode::Char('k') => {
                app.conn_prev();
            }
            KeyCode::Down | KeyCode::Char('j') => {
                app.conn_next();
            }
            KeyCode::Enter => {
                if let Some(target_id) = app.selected_conn_memory_id() {
                    app.drill_into(target_id, engine).await;
                }
            }
            KeyCode::Esc | KeyCode::Backspace => {
                app.go_back();
            }
            _ => {}
        },
    }
}

// ─── Helpers ──────────────────────────────────────────────────

/// Render an importance bar as (filled, empty) span content.
fn importance_bar(value: f32, width: usize) -> (String, String) {
    let filled = ((value * width as f32).round() as usize).min(width);
    let empty = width - filled;
    ("\u{2588}".repeat(filled), "\u{2591}".repeat(empty))
}

fn select_next(state: &mut ListState, len: usize) {
    if len == 0 {
        return;
    }
    let i = state.selected().map_or(0, |i| (i + 1).min(len - 1));
    state.select(Some(i));
}

fn select_prev(state: &mut ListState) {
    let i = state.selected().map_or(0, |i| i.saturating_sub(1));
    state.select(Some(i));
}

fn trunc(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else if max > 3 {
        format!("{}...", &s[..max - 3])
    } else {
        s[..max].to_string()
    }
}
