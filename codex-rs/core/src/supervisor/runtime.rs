use std::path::Path;
use std::path::PathBuf;

use codex_protocol::ThreadId;
use codex_protocol::request_user_input::RequestUserInputResponse;
use tokio::sync::Mutex;
use tracing::warn;

use crate::supervisor::notebook::SupervisorNotebook;

const SUPERVISOR_ENV_VAR: &str = "GUGUGAGA_SUPERVISOR_ENABLED";

pub(crate) struct SupervisorRuntime {
    enabled: bool,
    notebook_path: Option<PathBuf>,
    notebook: Mutex<SupervisorNotebook>,
    last_update_key: Mutex<Option<String>>,
}

impl SupervisorRuntime {
    pub(crate) async fn from_env(codex_home: &Path, conversation_id: ThreadId) -> Self {
        let enabled = supervisor_enabled_from_env();
        Self::new(codex_home, conversation_id, enabled).await
    }

    async fn new(codex_home: &Path, conversation_id: ThreadId, enabled: bool) -> Self {
        if !enabled {
            return Self {
                enabled,
                notebook_path: None,
                notebook: Mutex::new(SupervisorNotebook::default()),
                last_update_key: Mutex::new(None),
            };
        }

        let notebook_path = codex_home
            .join("gugugaga")
            .join("notebooks")
            .join(format!("{conversation_id}.json"));
        let notebook = load_notebook(&notebook_path).await.unwrap_or_default();

        Self {
            enabled,
            notebook_path: Some(notebook_path),
            notebook: Mutex::new(notebook),
            last_update_key: Mutex::new(None),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) async fn after_agent(
        &self,
        turn_id: &str,
        input_messages: &[String],
        last_assistant_message: Option<&str>,
    ) -> Option<String> {
        if !self.enabled {
            return None;
        }

        let user_summary = summarize_user_message(input_messages.last());
        let assistant_summary = summarize_assistant_message(last_assistant_message);
        let update_key = format!("{turn_id}|{user_summary}|{assistant_summary}");
        let mut last_update_key = self.last_update_key.lock().await;
        if last_update_key.as_deref() == Some(update_key.as_str()) {
            return None;
        }
        *last_update_key = Some(update_key);
        drop(last_update_key);

        let mut notebook = self.notebook.lock().await;
        notebook.apply_after_agent_update(
            format!("Reviewed gugugaga turn {turn_id}"),
            assistant_summary,
            "Awaiting next gugugaga review".to_string(),
        );
        let summary = notebook.summary_line();
        let snapshot = notebook.clone();
        drop(notebook);
        self.persist_notebook(&snapshot).await;

        Some(format!(
            "gugugaga supervisor updated notebook after turn review ({summary})."
        ))
    }

    pub(crate) async fn request_user_input_response(
        &self,
        turn_id: &str,
        response: &RequestUserInputResponse,
    ) -> Option<String> {
        if !self.enabled {
            return None;
        }

        let answer_groups = response.answers.len();
        let selected_answers = response
            .answers
            .values()
            .map(|answer| answer.answers.len())
            .sum::<usize>();

        let mut notebook = self.notebook.lock().await;
        notebook.apply_request_user_input_update(turn_id, answer_groups, selected_answers);
        let summary = notebook.summary_line();
        let snapshot = notebook.clone();
        drop(notebook);
        self.persist_notebook(&snapshot).await;

        Some(format!(
            "gugugaga supervisor recorded request_user_input response ({summary})."
        ))
    }

    async fn persist_notebook(&self, notebook: &SupervisorNotebook) {
        let Some(path) = self.notebook_path.as_ref() else {
            return;
        };

        if let Some(parent) = path.parent()
            && let Err(err) = tokio::fs::create_dir_all(parent).await
        {
            warn!(
                path = %path.display(),
                error = %err,
                "failed to create supervisor notebook parent directory"
            );
            return;
        }

        match serde_json::to_string_pretty(notebook) {
            Ok(serialized) => {
                if let Err(err) = tokio::fs::write(path, serialized).await {
                    warn!(
                        path = %path.display(),
                        error = %err,
                        "failed to persist supervisor notebook"
                    );
                }
            }
            Err(err) => {
                warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to serialize supervisor notebook"
                );
            }
        }
    }
}

async fn load_notebook(path: &Path) -> Option<SupervisorNotebook> {
    let raw = match tokio::fs::read_to_string(path).await {
        Ok(raw) => raw,
        Err(err) => {
            if err.kind() == std::io::ErrorKind::NotFound {
                return None;
            }
            warn!(
                path = %path.display(),
                error = %err,
                "failed to read supervisor notebook; starting empty"
            );
            return None;
        }
    };

    match serde_json::from_str::<SupervisorNotebook>(&raw) {
        Ok(notebook) => Some(notebook),
        Err(err) => {
            warn!(
                path = %path.display(),
                error = %err,
                "failed to parse supervisor notebook; starting empty"
            );
            None
        }
    }
}

fn summarize_user_message(user_message: Option<&String>) -> String {
    truncate_text(
        user_message.map_or("", String::as_str).trim(),
        180,
        "No user message was captured for this turn.",
    )
}

fn summarize_assistant_message(last_assistant_message: Option<&str>) -> String {
    truncate_text(
        last_assistant_message.unwrap_or("").trim(),
        220,
        "Assistant produced no final text.",
    )
}

fn truncate_text(input: &str, limit: usize, fallback: &str) -> String {
    if input.is_empty() {
        return fallback.to_string();
    }

    let mut output = String::with_capacity(limit.min(input.len()));
    for (count, ch) in input.chars().enumerate() {
        if count == limit {
            output.push_str("...");
            return output;
        }
        output.push(ch);
    }
    output
}

fn supervisor_enabled_from_env() -> bool {
    std::env::var(SUPERVISOR_ENV_VAR)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn disabled_runtime_skips_updates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = SupervisorRuntime::new(dir.path(), ThreadId::default(), false).await;

        let event = runtime
            .after_agent("turn-1", &[String::from("hello")], Some("world"))
            .await;
        assert_eq!(event, None);
    }

    #[tokio::test]
    async fn enabled_runtime_persists_updates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let conversation_id = ThreadId::default();
        let runtime = SupervisorRuntime::new(dir.path(), conversation_id, true).await;

        let event = runtime
            .after_agent("turn-1", &[String::from("hello")], Some("world"))
            .await;
        assert!(event.is_some());

        let notebook_path = dir
            .path()
            .join("gugugaga")
            .join("notebooks")
            .join(format!("{conversation_id}.json"));
        assert!(notebook_path.exists());

        let stored = tokio::fs::read_to_string(notebook_path)
            .await
            .expect("read notebook");
        assert!(stored.contains("Reviewed gugugaga turn turn-1"));
    }

    #[tokio::test]
    async fn deduplicates_equivalent_after_agent_updates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = SupervisorRuntime::new(dir.path(), ThreadId::default(), true).await;

        let first = runtime
            .after_agent("turn-1", &[String::from("hello")], Some("world"))
            .await;
        let second = runtime
            .after_agent("turn-1", &[String::from("hello")], Some("world"))
            .await;

        assert!(first.is_some());
        assert_eq!(second, None);
    }

    #[test]
    fn truncate_text_returns_fallback_for_empty_input() {
        assert_eq!(truncate_text("", 20, "fallback"), "fallback");
    }
}
