//! Local issue tracker used by `guga-codex issues`.
//!
//! Storage format is JSONL in `.issues/issues.jsonl` so it stays compatible
//! with existing issue data.

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};

pub const ID_PREFIX: &str = "guga-codex-";
const ID_HASH_LEN: usize = 5;

pub const STATUS_OPEN: &str = "open";
pub const STATUS_IN_PROGRESS: &str = "in_progress";
pub const STATUS_BLOCKED: &str = "blocked";
pub const STATUS_CLOSED: &str = "closed";

const ONBOARD_SECTION: &str = "\
## Guga-Codex Issues

This project uses `guga-codex issues` for issue tracking.
Run `guga-codex issues prime` for workflow context.

**Important**: Do NOT use your built-in TodoWrite or task management tools.
All task tracking must go through `guga-codex issues` commands.
";

const ONBOARD_FILE: &str = "\
# AGENTS.md

This project uses `guga-codex issues` for issue tracking.
Run `guga-codex issues prime` for workflow context.

**Important**: Do NOT use your built-in TodoWrite or task management tools.
All task tracking must go through `guga-codex issues` commands.
";

const PRIME_PROMPT: &str = "\
# guga-codex issues prime

guga-codex issues is a local issue tracker for AI-assisted work.
IDs: guga-codex-xxxxx

Workflow:
1. Pick work: `guga-codex issues ready --json`
2. Create issues for new work: `guga-codex issues create \"...\" --description \"...\"`
3. Mark in progress: `guga-codex issues status <id> in_progress`
4. Update details: `guga-codex issues update <id> --description \"...\" --notes \"...\" --priority 1`
5. Link dependencies: `guga-codex issues dep add <child> <parent>`
6. Close when done: `guga-codex issues close <id>`

Useful:
- `guga-codex issues list [--status <status>] [--ready] [--all]`
- `guga-codex issues show <id>`
- Use `--json` for machine parsing

Notes:
- `list` hides closed by default; add `--all` to include them.
- If lock is stale, retry with `--force`.
- Do not read or edit issue storage directly; use `guga-codex issues` commands.
- Do NOT use built-in TodoWrite or task management; use `guga-codex issues` exclusively.
";

fn default_status() -> String {
    STATUS_OPEN.to_string()
}

fn deserialize_timestamp<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TimestampValue {
        String(String),
        U64(u64),
        I64(i64),
        F64(f64),
    }

    match TimestampValue::deserialize(deserializer)? {
        TimestampValue::String(v) => Ok(v),
        TimestampValue::U64(v) => Ok(v.to_string()),
        TimestampValue::I64(v) => Ok(v.to_string()),
        TimestampValue::F64(v) => Ok((v as i64).to_string()),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Issue {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_status")]
    pub status: String,
    pub priority: u8,
    #[serde(deserialize_with = "deserialize_timestamp")]
    pub created_at: String,
    #[serde(deserialize_with = "deserialize_timestamp")]
    pub updated_at: String,
    #[serde(default)]
    pub deps: Vec<String>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Debug, Clone)]
pub struct CreateIssueInput {
    pub description: String,
    pub priority: u8,
    pub deps: Vec<String>,
    pub notes: String,
}

impl Default for CreateIssueInput {
    fn default() -> Self {
        Self {
            description: String::new(),
            priority: 2,
            deps: Vec::new(),
            notes: String::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct UpdateIssueInput {
    pub title: Option<String>,
    pub description: Option<String>,
    pub notes: Option<String>,
    pub append_notes: Option<String>,
    pub priority: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ListSort {
    #[default]
    CreatedDesc,
    Priority,
}

#[derive(Debug, Clone, Default)]
pub struct ListIssuesOptions {
    pub status: Option<String>,
    pub priority: Option<u8>,
    pub ready_only: bool,
    pub include_closed: bool,
    pub search: Option<String>,
    pub sort: ListSort,
}

#[derive(Debug, Clone)]
pub struct IssueStore {
    root: PathBuf,
}

impl IssueStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn issues_dir(&self) -> PathBuf {
        self.root.join(".issues")
    }

    pub fn issues_path(&self) -> PathBuf {
        self.issues_dir().join("issues.jsonl")
    }

    pub fn backup_path(&self) -> PathBuf {
        self.issues_dir().join("issues.jsonl.bak")
    }

    pub fn tmp_path(&self) -> PathBuf {
        self.issues_dir().join("issues.jsonl.tmp")
    }

    pub fn lock_path(&self) -> PathBuf {
        self.issues_dir().join("lock")
    }

    pub fn init(&self, force: bool) -> Result<(PathBuf, Option<String>)> {
        self.with_lock(force, |store| {
            store.ensure_workspace()?;
            let doc_change = ensure_agents_doc(&store.root)?;
            Ok((store.issues_path(), doc_change))
        })
    }

    pub fn ensure_workspace(&self) -> Result<()> {
        let dir = self.issues_dir();
        fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;

        let path = self.issues_path();
        if !path.exists() {
            fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        }
        Ok(())
    }

    pub fn with_lock<T, F>(&self, force: bool, operation: F) -> Result<T>
    where
        F: FnOnce(&IssueStore) -> Result<T>,
    {
        self.ensure_workspace()?;
        let lock_path = self.lock_path();

        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(mut file) => {
                let _ = writeln!(
                    file,
                    "pid={} time={}",
                    std::process::id(),
                    Utc::now().to_rfc3339()
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if !force {
                    return Err(anyhow!("workspace locked"));
                }
                fs::remove_file(&lock_path)
                    .with_context(|| format!("remove stale lock {}", lock_path.display()))?;
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&lock_path)
                    .with_context(|| format!("create lock {}", lock_path.display()))?;
                let _ = writeln!(
                    file,
                    "pid={} time={}",
                    std::process::id(),
                    Utc::now().to_rfc3339()
                );
            }
            Err(e) => {
                return Err(e).with_context(|| format!("create lock {}", lock_path.display()));
            }
        }

        let guard = LockGuard {
            path: lock_path.clone(),
        };
        let result = operation(self);
        drop(guard);
        result
    }

    pub fn load_all(&self) -> Result<Vec<Issue>> {
        self.ensure_workspace()?;
        let primary = self.issues_path();
        let backup = self.backup_path();

        match load_issues_from_path(&primary) {
            Ok(issues) => Ok(issues),
            Err(primary_err) => {
                if backup.exists() {
                    load_issues_from_path(&backup).with_context(|| {
                        format!(
                            "failed loading primary {} and backup {}",
                            primary.display(),
                            backup.display()
                        )
                    })
                } else {
                    Err(primary_err)
                }
            }
        }
    }

    pub fn save_all(&self, issues: &[Issue]) -> Result<()> {
        self.ensure_workspace()?;

        let primary = self.issues_path();
        let backup = self.backup_path();
        let tmp = self.tmp_path();

        if primary.exists() {
            fs::copy(&primary, &backup)
                .with_context(|| format!("backup {} -> {}", primary.display(), backup.display()))?;
        }

        let mut file =
            fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        for issue in issues {
            serde_json::to_writer(&mut file, issue)
                .with_context(|| format!("serialize issue {}", issue.id))?;
            file.write_all(b"\n")
                .with_context(|| format!("write {}", tmp.display()))?;
        }
        file.flush()
            .with_context(|| format!("flush {}", tmp.display()))?;
        let _ = file.sync_all();

        fs::rename(&tmp, &primary).with_context(|| format!("replace {}", primary.display()))?;
        Ok(())
    }

    pub fn create_issue(&self, title: &str, input: CreateIssueInput) -> Result<Issue> {
        let title = title.trim();
        if title.is_empty() {
            return Err(anyhow!("title cannot be empty"));
        }

        let mut issues = self.load_all()?;
        for dep in &input.deps {
            if !issues.iter().any(|issue| &issue.id == dep) {
                return Err(anyhow!("dependency not found: {dep}"));
            }
        }

        let now = now_millis_string();
        let id = next_issue_id(&issues, title, &input.description, &now);

        let issue = Issue {
            id,
            title: title.to_string(),
            description: input.description,
            status: STATUS_OPEN.to_string(),
            priority: input.priority,
            created_at: now.clone(),
            updated_at: now,
            deps: dedup_preserve_order(input.deps),
            notes: input.notes,
        };

        issues.push(issue.clone());
        self.save_all(&issues)?;
        Ok(issue)
    }

    pub fn list_issues(&self, options: &ListIssuesOptions) -> Result<Vec<Issue>> {
        let all_issues = self.load_all()?;
        let mut filtered = all_issues.clone();

        if let Some(status_input) = &options.status {
            let status = normalize_status(status_input)
                .ok_or_else(|| anyhow!("invalid status: {status_input}"))?;
            filtered.retain(|issue| issue.status == status);
        } else if !options.include_closed {
            filtered.retain(|issue| issue.status != STATUS_CLOSED);
        }

        if let Some(priority) = options.priority {
            filtered.retain(|issue| issue.priority == priority);
        }

        if let Some(needle) = &options.search {
            let needle = needle.to_ascii_lowercase();
            filtered.retain(|issue| {
                issue.title.to_ascii_lowercase().contains(&needle)
                    || issue.description.to_ascii_lowercase().contains(&needle)
                    || issue.notes.to_ascii_lowercase().contains(&needle)
            });
        }

        if options.ready_only {
            let status_map = build_status_map(&all_issues);
            filtered.retain(|issue| is_ready(issue, &status_map));
        }

        match options.sort {
            ListSort::Priority => {
                filtered.sort_by(|a, b| a.priority.cmp(&b.priority).then_with(|| a.id.cmp(&b.id)));
            }
            ListSort::CreatedDesc => {
                filtered.sort_by(|a, b| {
                    parse_timestamp_millis(&b.created_at)
                        .cmp(&parse_timestamp_millis(&a.created_at))
                        .then_with(|| b.id.cmp(&a.id))
                });
            }
        }

        Ok(filtered)
    }

    pub fn ready_issues(&self) -> Result<Vec<Issue>> {
        self.list_issues(&ListIssuesOptions {
            status: None,
            priority: None,
            ready_only: true,
            include_closed: true,
            search: None,
            sort: ListSort::CreatedDesc,
        })
    }

    pub fn get_issue(&self, id: &str) -> Result<Option<Issue>> {
        let issues = self.load_all()?;
        Ok(issues.into_iter().find(|issue| issue.id == id))
    }

    pub fn set_status(&self, id: &str, status_input: &str) -> Result<Issue> {
        let status = normalize_status(status_input)
            .ok_or_else(|| anyhow!("invalid status: {status_input}"))?;

        let mut issues = self.load_all()?;
        let idx = find_issue_index(&issues, id).ok_or_else(|| anyhow!("issue not found: {id}"))?;
        issues[idx].status = status.to_string();
        issues[idx].updated_at = now_millis_string();
        let updated = issues[idx].clone();
        self.save_all(&issues)?;
        Ok(updated)
    }

    pub fn close_issue(&self, id: &str) -> Result<Issue> {
        self.set_status(id, STATUS_CLOSED)
    }

    pub fn update_issue(&self, id: &str, input: UpdateIssueInput) -> Result<Issue> {
        if input.notes.is_some() && input.append_notes.is_some() {
            return Err(anyhow!("use --notes or --append-notes, not both"));
        }

        let mut issues = self.load_all()?;
        let idx = find_issue_index(&issues, id).ok_or_else(|| anyhow!("issue not found: {id}"))?;
        let mut issue = issues[idx].clone();

        if let Some(title) = input.title {
            let title = title.trim();
            if title.is_empty() {
                return Err(anyhow!("title cannot be empty"));
            }
            issue.title = title.to_string();
        }

        if let Some(description) = input.description {
            issue.description = description;
        }

        if let Some(notes) = input.notes {
            issue.notes = notes;
        }

        if let Some(extra) = input.append_notes {
            let extra = extra.trim();
            if !extra.is_empty() {
                if issue.notes.trim().is_empty() {
                    issue.notes = extra.to_string();
                } else {
                    issue.notes.push('\n');
                    issue.notes.push_str(extra);
                }
            }
        }

        if let Some(priority) = input.priority {
            issue.priority = priority;
        }

        issue.updated_at = now_millis_string();
        issues[idx] = issue.clone();
        self.save_all(&issues)?;
        Ok(issue)
    }

    pub fn delete_issue(&self, id: &str) -> Result<()> {
        let mut issues = self.load_all()?;
        let idx = find_issue_index(&issues, id).ok_or_else(|| anyhow!("issue not found: {id}"))?;
        issues.remove(idx);

        let now = now_millis_string();
        for issue in &mut issues {
            let old_len = issue.deps.len();
            issue.deps.retain(|dep| dep != id);
            if issue.deps.len() != old_len {
                issue.updated_at = now.clone();
            }
        }

        self.save_all(&issues)?;
        Ok(())
    }

    pub fn add_dependency(&self, child: &str, parent: &str) -> Result<Issue> {
        if child == parent {
            return Err(anyhow!("dependency cannot be self"));
        }

        let mut issues = self.load_all()?;
        let child_idx = find_issue_index(&issues, child)
            .ok_or_else(|| anyhow!("child issue not found: {child}"))?;
        if find_issue_index(&issues, parent).is_none() {
            return Err(anyhow!("parent issue not found: {parent}"));
        }

        if issues[child_idx].deps.iter().any(|dep| dep == parent) {
            return Err(anyhow!("dependency already present"));
        }

        if would_create_cycle(&issues, child, parent) {
            return Err(anyhow!("dependency would create cycle"));
        }

        issues[child_idx].deps.push(parent.to_string());
        issues[child_idx].updated_at = now_millis_string();
        let updated = issues[child_idx].clone();
        self.save_all(&issues)?;
        Ok(updated)
    }

    pub fn remove_dependency(&self, child: &str, parent: &str) -> Result<Issue> {
        let mut issues = self.load_all()?;
        let child_idx = find_issue_index(&issues, child)
            .ok_or_else(|| anyhow!("child issue not found: {child}"))?;

        let old_len = issues[child_idx].deps.len();
        issues[child_idx].deps.retain(|dep| dep != parent);
        if issues[child_idx].deps.len() == old_len {
            return Err(anyhow!("dependency not found"));
        }

        issues[child_idx].updated_at = now_millis_string();
        let updated = issues[child_idx].clone();
        self.save_all(&issues)?;
        Ok(updated)
    }
}

struct LockGuard {
    path: PathBuf,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub fn onboard_text() -> &'static str {
    ONBOARD_SECTION
}

pub fn prime_prompt() -> &'static str {
    PRIME_PROMPT
}

pub fn ensure_agents_doc(root: &Path) -> Result<Option<String>> {
    let path = root.join("AGENTS.md");

    if path.exists() {
        let content =
            fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;

        if content.contains("guga-codex issues") {
            return Ok(None);
        }

        let updated = if content.trim().is_empty() {
            ONBOARD_FILE.to_string()
        } else {
            format!("{}\n\n{}", content.trim_end(), ONBOARD_SECTION)
        };
        fs::write(&path, updated).with_context(|| format!("write {}", path.display()))?;
        return Ok(Some("updated AGENTS.md".to_string()));
    }

    fs::write(&path, ONBOARD_FILE).with_context(|| format!("write {}", path.display()))?;
    Ok(Some("created AGENTS.md".to_string()))
}

pub fn normalize_status(input: &str) -> Option<&'static str> {
    let normalized = input.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "open" => Some(STATUS_OPEN),
        "in_progress" | "inprogress" => Some(STATUS_IN_PROGRESS),
        "blocked" => Some(STATUS_BLOCKED),
        "closed" | "done" => Some(STATUS_CLOSED),
        _ => None,
    }
}

pub fn is_ready(issue: &Issue, status_by_id: &HashMap<String, String>) -> bool {
    if issue.status != STATUS_OPEN && issue.status != STATUS_IN_PROGRESS {
        return false;
    }
    if issue.deps.is_empty() {
        return true;
    }

    issue
        .deps
        .iter()
        .all(|dep| matches!(status_by_id.get(dep), Some(status) if status == STATUS_CLOSED))
}

pub fn build_status_map(issues: &[Issue]) -> HashMap<String, String> {
    issues
        .iter()
        .map(|issue| (issue.id.clone(), issue.status.clone()))
        .collect()
}

pub fn format_issue_line(issue: &Issue, color: bool) -> String {
    format!(
        "{} [{}] (p{}) {}",
        colorize(issue.id.as_str(), Ansi::Cyan, color),
        colorize_status(&issue.status, color),
        colorize(&issue.priority.to_string(), Ansi::Magenta, color),
        issue.title,
    )
}

pub fn format_issue_details(issue: &Issue, color: bool) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("id: {}", colorize(&issue.id, Ansi::Cyan, color)));
    lines.push(format!("title: {}", issue.title));
    if !issue.description.trim().is_empty() {
        lines.push(format!("description: {}", issue.description));
    }
    lines.push(format!(
        "status: {}",
        colorize_status(&issue.status, color)
    ));
    lines.push(format!(
        "priority: {}",
        colorize(&issue.priority.to_string(), Ansi::Magenta, color)
    ));
    lines.push(format!("created_at: {}", issue.created_at));
    lines.push(format!("updated_at: {}", issue.updated_at));
    if issue.deps.is_empty() {
        lines.push("deps: -".to_string());
    } else {
        lines.push(format!("deps: {}", issue.deps.join(", ")));
    }
    if !issue.notes.trim().is_empty() {
        lines.push(format!("notes: {}", issue.notes));
    }
    lines
}

#[derive(Debug, Clone, Copy)]
enum Ansi {
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    Gray,
}

fn colorize(text: &str, ansi: Ansi, color: bool) -> String {
    if !color {
        return text.to_string();
    }
    let code = match ansi {
        Ansi::Red => "31",
        Ansi::Green => "32",
        Ansi::Yellow => "33",
        Ansi::Blue => "34",
        Ansi::Magenta => "35",
        Ansi::Cyan => "36",
        Ansi::Gray => "90",
    };
    format!("\x1b[{code}m{text}\x1b[0m")
}

fn colorize_status(status: &str, color: bool) -> String {
    match status {
        STATUS_OPEN => colorize(status, Ansi::Green, color),
        STATUS_IN_PROGRESS => colorize(status, Ansi::Yellow, color),
        STATUS_BLOCKED => colorize(status, Ansi::Red, color),
        STATUS_CLOSED => colorize(status, Ansi::Gray, color),
        _ => colorize(status, Ansi::Blue, color),
    }
}

pub fn serve(store: &IssueStore, host: &str, port: u16) -> Result<()> {
    store.ensure_workspace()?;
    let listener =
        TcpListener::bind((host, port)).with_context(|| format!("bind {}:{}", host, port))?;

    for stream_result in listener.incoming() {
        match stream_result {
            Ok(stream) => {
                let _ = handle_http_connection(stream, store);
            }
            Err(e) => {
                return Err(anyhow!("accept error: {e}"));
            }
        }
    }
    Ok(())
}

fn handle_http_connection(mut stream: TcpStream, store: &IssueStore) -> Result<()> {
    let mut first_line = String::new();
    {
        let mut reader = BufReader::new(&mut stream);
        reader
            .read_line(&mut first_line)
            .context("read HTTP request line")?;

        let mut line = String::new();
        loop {
            line.clear();
            let bytes = reader.read_line(&mut line).context("read HTTP headers")?;
            if bytes == 0 || line == "\r\n" {
                break;
            }
        }
    }

    let path = parse_http_path(&first_line);

    if path == "/" || path == "/index.html" {
        let issues = store.load_all()?;
        let body = render_issues_page(&issues);
        write_http_response(
            &mut stream,
            200,
            "OK",
            "text/html; charset=utf-8",
            body.as_bytes(),
        )?;
    } else {
        write_http_response(
            &mut stream,
            404,
            "Not Found",
            "text/plain; charset=utf-8",
            b"Not Found",
        )?;
    }

    Ok(())
}

fn parse_http_path(request_line: &str) -> &str {
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");
    if method.eq_ignore_ascii_case("GET") {
        path
    } else {
        "/404"
    }
}

fn write_http_response(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_code,
        status_text,
        content_type,
        body.len()
    );

    stream
        .write_all(header.as_bytes())
        .context("write HTTP headers")?;
    stream.write_all(body).context("write HTTP body")?;
    stream.flush().ok();
    Ok(())
}

fn render_issues_page(issues: &[Issue]) -> String {
    let mut sorted = issues.to_vec();
    sorted.sort_by(|a, b| {
        parse_timestamp_millis(&b.created_at)
            .cmp(&parse_timestamp_millis(&a.created_at))
            .then_with(|| b.id.cmp(&a.id))
    });

    let mut html = String::new();
    html.push_str("<!doctype html><html><head>");
    html.push_str("<meta charset=\"utf-8\">");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
    html.push_str("<title>guga-codex issues</title>");
    html.push_str("<style>");
    html.push_str("body{font-family:-apple-system,system-ui,Segoe UI,Roboto,sans-serif;margin:24px;background:#0b0e14;color:#e6edf3;}");
    html.push_str("h1{font-size:20px;margin:0 0 16px;}table{width:100%;border-collapse:collapse;background:#111827;border:1px solid #1f2937;border-radius:8px;overflow:hidden;}");
    html.push_str("th,td{padding:10px 12px;border-bottom:1px solid #1f2937;text-align:left;vertical-align:top;}th{font-size:12px;color:#9ca3af;text-transform:uppercase;letter-spacing:.04em;background:#0f172a;}tr:last-child td{border-bottom:none;}");
    html.push_str("code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}span.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;}span.open{background:#064e3b;color:#a7f3d0;}span.inprogress{background:#78350f;color:#fde68a;}span.blocked{background:#7f1d1d;color:#fecaca;}span.closed{background:#374151;color:#e5e7eb;}span.unknown{background:#0f172a;color:#cbd5f5;}div.desc{white-space:pre-wrap;color:#d1d5db;margin-top:4px;}");
    html.push_str("</style></head><body>");
    html.push_str("<h1>guga-codex issues</h1>");

    if sorted.is_empty() {
        html.push_str("<p>No issues.</p></body></html>");
        return html;
    }

    html.push_str("<table><thead><tr><th>ID</th><th>Status</th><th>Priority</th><th>Title</th></tr></thead><tbody>");
    for issue in &sorted {
        let status_class = match issue.status.as_str() {
            STATUS_OPEN => "open",
            STATUS_IN_PROGRESS => "inprogress",
            STATUS_BLOCKED => "blocked",
            STATUS_CLOSED => "closed",
            _ => "unknown",
        };

        html.push_str("<tr>");
        html.push_str(&format!(
            "<td><code>{}</code></td>",
            escape_html(&issue.id)
        ));
        html.push_str(&format!(
            "<td><span class=\"badge {}\">{}</span></td>",
            status_class,
            escape_html(&issue.status)
        ));
        html.push_str(&format!("<td>{}</td>", issue.priority));
        html.push_str("<td>");
        html.push_str(&escape_html(&issue.title));
        if !issue.description.trim().is_empty() {
            html.push_str(&format!(
                "<div class=\"desc\">{}</div>",
                escape_html(&issue.description)
            ));
        }
        if !issue.notes.trim().is_empty() {
            html.push_str(&format!(
                "<div class=\"desc\">{}</div>",
                escape_html(&issue.notes)
            ));
        }
        html.push_str("</td></tr>");
    }
    html.push_str("</tbody></table></body></html>");
    html
}

fn escape_html(input: &str) -> String {
    let mut out = String::new();
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

fn load_issues_from_path(path: &Path) -> Result<Vec<Issue>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut text = String::new();
    fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .read_to_string(&mut text)
        .with_context(|| format!("read {}", path.display()))?;

    let mut issues = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let issue: Issue = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "parse issue JSON at {}:{}",
                path.display(),
                line_no.saturating_add(1)
            )
        })?;
        issues.push(issue);
    }
    Ok(issues)
}

fn find_issue_index(issues: &[Issue], id: &str) -> Option<usize> {
    issues.iter().position(|issue| issue.id == id)
}

fn now_millis_string() -> String {
    Utc::now().timestamp_millis().to_string()
}

fn parse_timestamp_millis(value: &str) -> i64 {
    value.parse::<i64>().unwrap_or(0)
}

fn dedup_preserve_order(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            out.push(value);
        }
    }
    out
}

fn next_issue_id(existing: &[Issue], title: &str, description: &str, created_at: &str) -> String {
    use std::collections::hash_map::DefaultHasher;

    let modulus = pow_u64(36, ID_HASH_LEN as u32);
    let mut nonce = 0_u64;
    loop {
        let mut hasher = DefaultHasher::new();
        title.hash(&mut hasher);
        description.hash(&mut hasher);
        created_at.hash(&mut hasher);
        nonce.hash(&mut hasher);

        let value = if modulus == 0 {
            hasher.finish()
        } else {
            hasher.finish() % modulus
        };
        let suffix = to_base36_padded(value, ID_HASH_LEN);
        let candidate = format!("{ID_PREFIX}{suffix}");
        if !existing.iter().any(|issue| issue.id == candidate) {
            return candidate;
        }
        nonce = nonce.saturating_add(1);
    }
}

fn pow_u64(base: u64, exp: u32) -> u64 {
    let mut out = 1_u64;
    for _ in 0..exp {
        out = out.saturating_mul(base);
    }
    out
}

fn to_base36_padded(mut value: u64, width: usize) -> String {
    const ALPHABET: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if width == 0 {
        return String::new();
    }

    let mut chars = vec!['0'; width];
    for i in (0..width).rev() {
        chars[i] = ALPHABET[(value % 36) as usize] as char;
        value /= 36;
    }
    chars.into_iter().collect()
}

fn would_create_cycle(issues: &[Issue], child: &str, parent: &str) -> bool {
    let mut deps_by_id: HashMap<&str, Vec<&str>> = HashMap::new();
    for issue in issues {
        deps_by_id.insert(
            issue.id.as_str(),
            issue.deps.iter().map(|s| s.as_str()).collect(),
        );
    }

    let mut stack = vec![parent];
    let mut visited: HashSet<&str> = HashSet::new();

    while let Some(current) = stack.pop() {
        if current == child {
            return true;
        }
        if !visited.insert(current) {
            continue;
        }
        if let Some(deps) = deps_by_id.get(current) {
            for dep in deps {
                stack.push(dep);
            }
        }
    }
    false
}

pub fn workspace_root(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return path
            .canonicalize()
            .with_context(|| format!("canonicalize {}", path.display()));
    }
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_store() -> (TempDir, IssueStore) {
        let tmp = TempDir::new().expect("create temp dir");
        let store = IssueStore::new(tmp.path());
        (tmp, store)
    }

    #[test]
    fn create_and_read_issue_roundtrip() {
        let (_tmp, store) = setup_store();
        store.init(false).expect("init");

        let created = store
            .with_lock(false, |s| {
                s.create_issue(
                    "add issue tracker",
                    CreateIssueInput {
                        description: "rewrite issues in Rust".to_string(),
                        priority: 1,
                        deps: vec![],
                        notes: "ship mvp first".to_string(),
                    },
                )
            })
            .expect("create");

        let loaded = store.get_issue(&created.id).expect("get").expect("present");
        assert_eq!(loaded.title, "add issue tracker");
        assert_eq!(loaded.status, STATUS_OPEN);
        assert_eq!(loaded.priority, 1);
        assert!(loaded.id.starts_with(ID_PREFIX));
    }

    #[test]
    fn ready_requires_dependencies_closed() {
        let (_tmp, store) = setup_store();
        store.init(false).expect("init");

        let parent = store
            .with_lock(false, |s| {
                s.create_issue("parent", CreateIssueInput::default())
            })
            .expect("parent");
        let child = store
            .with_lock(false, |s| {
                s.create_issue(
                    "child",
                    CreateIssueInput {
                        deps: vec![parent.id.clone()],
                        ..CreateIssueInput::default()
                    },
                )
            })
            .expect("child");

        let ready_before = store.ready_issues().expect("ready");
        assert!(ready_before.iter().all(|issue| issue.id != child.id));

        store
            .with_lock(false, |s| s.close_issue(&parent.id))
            .expect("close parent");
        let ready_after = store.ready_issues().expect("ready");
        assert!(ready_after.iter().any(|issue| issue.id == child.id));
    }

    #[test]
    fn dependency_cycle_is_rejected() {
        let (_tmp, store) = setup_store();
        store.init(false).expect("init");

        let a = store
            .with_lock(false, |s| s.create_issue("a", CreateIssueInput::default()))
            .expect("a");
        let b = store
            .with_lock(false, |s| s.create_issue("b", CreateIssueInput::default()))
            .expect("b");
        let c = store
            .with_lock(false, |s| s.create_issue("c", CreateIssueInput::default()))
            .expect("c");

        store
            .with_lock(false, |s| s.add_dependency(&a.id, &b.id))
            .expect("a->b");
        store
            .with_lock(false, |s| s.add_dependency(&b.id, &c.id))
            .expect("b->c");

        let err = store
            .with_lock(false, |s| s.add_dependency(&c.id, &a.id))
            .expect_err("c->a should fail");
        assert!(err.to_string().contains("cycle"));
    }

    #[test]
    fn lock_requires_force_when_present() {
        let (_tmp, store) = setup_store();
        store.ensure_workspace().expect("workspace");
        fs::write(store.lock_path(), "stale").expect("write lock");

        let err = store
            .with_lock(false, |_| Ok::<_, anyhow::Error>(()))
            .expect_err("should fail without force");
        assert!(err.to_string().contains("locked"));

        store
            .with_lock(true, |_| Ok::<_, anyhow::Error>(()))
            .expect("force lock");
    }

    #[test]
    fn sort_priority_orders_low_number_first() {
        let (_tmp, store) = setup_store();
        store.init(false).expect("init");

        store
            .with_lock(false, |s| {
                s.create_issue(
                    "p3",
                    CreateIssueInput {
                        priority: 3,
                        ..CreateIssueInput::default()
                    },
                )
            })
            .expect("p3");
        store
            .with_lock(false, |s| {
                s.create_issue(
                    "p1",
                    CreateIssueInput {
                        priority: 1,
                        ..CreateIssueInput::default()
                    },
                )
            })
            .expect("p1");

        let list = store
            .list_issues(&ListIssuesOptions {
                sort: ListSort::Priority,
                include_closed: true,
                ..ListIssuesOptions::default()
            })
            .expect("list");

        assert!(list.len() >= 2);
        assert!(list[0].priority <= list[1].priority);
    }
}
