use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use chrono::DateTime;
use chrono::Utc;
use codex_protocol::ThreadId;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_protocol::parse_command::ParsedCommand;
use codex_protocol::protocol::BackgroundEventEvent;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ExecCommandBeginEvent;
use codex_protocol::protocol::ExecCommandEndEvent;
use codex_protocol::protocol::ExecCommandSource;
use codex_protocol::protocol::ExecCommandStatus;
use codex_protocol::protocol::FileChange;
use codex_protocol::protocol::PatchApplyBeginEvent;
use codex_protocol::protocol::PatchApplyEndEvent;
use codex_protocol::protocol::PatchApplyStatus;
use codex_protocol::protocol::TurnDiffEvent;
use codex_protocol::request_user_input::RequestUserInputResponse;
use futures::StreamExt;
use serde::Deserialize;
use serde::Serialize;
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::warn;
use wildmatch::WildMatch;

use crate::client::ModelClient;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::tools::ResponsesApiTool;
use crate::client_common::tools::ToolSpec;
use crate::codex::TurnContext;
use crate::supervisor::notebook::AttentionItem;
use crate::supervisor::notebook::AttentionSource;
use crate::supervisor::notebook::CompletedItem;
use crate::supervisor::notebook::MistakeEntry;
use crate::supervisor::notebook::Priority;
use crate::supervisor::notebook::SupervisorNotebook;
use crate::tools::spec::AdditionalProperties;
use crate::tools::spec::JsonSchema;

const SUPERVISOR_ENV_VAR: &str = "GUGA_CODEX_SUPERVISOR_ENABLED";
const MAX_SUPERVISOR_FOLLOW_UP_ROUNDS: u32 = 8;
const MAX_SUPERVISOR_TOOL_CALLS: usize = 24;
const MAX_GLOB_VISITS: usize = 20_000;
const TOOL_OUTPUT_LIMIT: usize = 4_000;
const SUPERVISOR_STREAM_TIMEOUT: Duration = Duration::from_secs(45);
const SUPERVISOR_BASE_INSTRUCTIONS: &str = "You are GugaCodex, the supervision agent for Codex. Prefer conservative decisions and JSON-only outputs.";
const SUPERVISOR_CHAT_BASE_INSTRUCTIONS: &str = "You are GugaCodex, the supervision agent for Codex. Reply to the user in natural language and do not wrap your reply in JSON.";
const NOTEBOOK_VALIDATION_EXAMPLE: &str = r#"Example (minimal business-field format):
{
  "current_activity": "Reviewing current task",
  "completed": [
    { "what": "Implemented X", "significance": "Unblocks Y" }
  ],
  "attention": [
    { "content": "Need to verify edge case" }
  ],
  "mistakes": [
    {
      "what_happened": "Introduced a regression",
      "how_corrected": "Reverted and added coverage",
      "lesson": "Run targeted tests before final output"
    }
  ]
}"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistoryTurn {
    timestamp: DateTime<Utc>,
    role: String,
    content: String,
}

#[derive(Debug, Clone)]
struct StructuredToolCall {
    call_id: String,
    tool_name: String,
    arguments: String,
    item: ResponseItem,
}

#[derive(Debug, Clone)]
struct SupervisorDecision {
    result: String,
    summary: String,
    violation_type: Option<String>,
    description: Option<String>,
    correction: Option<String>,
}

impl SupervisorDecision {
    fn ok(summary: String) -> Self {
        Self {
            result: "ok".to_string(),
            summary,
            violation_type: None,
            description: None,
            correction: None,
        }
    }

    fn warning_message(&self) -> Option<String> {
        if !self.result.eq_ignore_ascii_case("violation") {
            return None;
        }

        let violation_type = self
            .violation_type
            .as_deref()
            .unwrap_or("UNSPECIFIED_VIOLATION");
        let description = self
            .description
            .as_deref()
            .unwrap_or(self.summary.as_str())
            .trim();
        let correction = self.correction.as_deref().unwrap_or("(none)").trim();

        Some(format!(
            "guga-codex supervisor violation {violation_type}: {description}. correction: {correction}"
        ))
    }

    fn correction_for_codex(&self) -> Option<String> {
        if !self.result.eq_ignore_ascii_case("violation") {
            return None;
        }
        self.correction
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    }

    fn correction_user_message(&self) -> Option<String> {
        self.correction_for_codex()
            .map(|correction| format!("🛡️ Corrected: {correction}"))
    }
}

#[derive(Debug, Clone)]
struct SupervisorPassResult {
    decision: SupervisorDecision,
    notebook_changed: bool,
    ui_events: Vec<EventMsg>,
}

#[derive(Debug, Clone, Default)]
struct NotebookSnapshot {
    raw: String,
}

#[derive(Debug, Clone)]
struct NotebookPatchHunk {
    old_lines: Vec<String>,
    new_lines: Vec<String>,
}

#[derive(Debug, Clone)]
struct NotebookPatchRenderPayload {
    summary: String,
    changes: HashMap<PathBuf, FileChange>,
    turn_diff: String,
}

#[derive(Debug)]
struct NotebookFieldError {
    path: String,
    reason: String,
    expected: String,
}

#[derive(Debug, Default)]
struct NotebookValidationReport {
    errors: Vec<NotebookFieldError>,
}

impl NotebookValidationReport {
    fn push(
        &mut self,
        path: impl Into<String>,
        reason: impl Into<String>,
        expected: impl Into<String>,
    ) {
        self.errors.push(NotebookFieldError {
            path: path.into(),
            reason: reason.into(),
            expected: expected.into(),
        });
    }

    fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    fn into_message(self) -> String {
        const MAX_ERRORS: usize = 24;
        let mut lines = vec!["apply_patch_notebook: Validation failed".to_string()];
        for error in self.errors.into_iter().take(MAX_ERRORS) {
            lines.push(format!(
                "- {}: {} (expected: {})",
                error.path, error.reason, error.expected
            ));
        }
        lines.push(NOTEBOOK_VALIDATION_EXAMPLE.to_string());
        lines.join("\n")
    }
}

pub(crate) struct SupervisorAfterAgentOutcome {
    pub(crate) warning_message: Option<String>,
    pub(crate) user_ack_message: Option<String>,
    pub(crate) correction_user_message: Option<String>,
    pub(crate) correction_to_codex: Option<String>,
    pub(crate) events: Vec<EventMsg>,
}

pub(crate) struct SupervisorChatOutcome {
    pub(crate) reply: String,
    pub(crate) events: Vec<EventMsg>,
    pub(crate) notebook_changed: bool,
}

pub(crate) struct SupervisorRuntime {
    enabled: bool,
    notebook_path: Option<PathBuf>,
    history_archive_path: Option<PathBuf>,
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
                history_archive_path: None,
                notebook: Mutex::new(SupervisorNotebook::default()),
                last_update_key: Mutex::new(None),
            };
        }

        let notebook_path = codex_home
            .join("guga-codex")
            .join("notebooks")
            .join(format!("{conversation_id}.json"));
        let history_archive_path = codex_home
            .join("guga-codex")
            .join("history")
            .join(format!("{conversation_id}.jsonl"));
        let notebook = load_notebook(&notebook_path).await.unwrap_or_default();

        Self {
            enabled,
            notebook_path: Some(notebook_path),
            history_archive_path: Some(history_archive_path),
            notebook: Mutex::new(notebook),
            last_update_key: Mutex::new(None),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    #[allow(dead_code)]
    pub(crate) async fn after_agent(
        &self,
        turn_id: &str,
        input_messages: &[String],
        last_assistant_message: Option<&str>,
    ) -> Option<String> {
        self.after_agent_with_turn(turn_id, input_messages, last_assistant_message, None, None)
            .await
            .and_then(|outcome| outcome.warning_message)
    }

    pub(crate) async fn after_agent_with_turn(
        &self,
        turn_id: &str,
        input_messages: &[String],
        last_assistant_message: Option<&str>,
        turn_context: Option<&TurnContext>,
        model_client: Option<&ModelClient>,
    ) -> Option<SupervisorAfterAgentOutcome> {
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

        if let Some(user_message) = input_messages.last() {
            self.append_history_turn("user", user_message).await;
        }
        if let Some(assistant_message) = last_assistant_message {
            self.append_history_turn("codex", assistant_message).await;
        }

        let mut review_summary = assistant_summary;
        let mut notebook_changed_by_tools = false;
        let mut violation_note = None;
        let mut correction_to_codex = None;
        let mut correction_user_message = None;
        let mut ui_events = Vec::new();

        if let (Some(tc), Some(client), Some(assistant_message)) =
            (turn_context, model_client, last_assistant_message)
        {
            match self
                .run_supervisor_pass(tc, client, input_messages, assistant_message)
                .await
            {
                Ok(pass) => {
                    review_summary = pass.decision.summary.clone();
                    notebook_changed_by_tools = pass.notebook_changed;
                    violation_note = pass.decision.warning_message();
                    correction_to_codex = pass.decision.correction_for_codex();
                    correction_user_message = pass.decision.correction_user_message();
                    ui_events = pass.ui_events;
                }
                Err(err) => {
                    warn!(
                        turn_id = turn_id,
                        error = %err,
                        "supervisor structured follow-up failed; using fallback notebook update"
                    );
                }
            }
        }

        let user_ack_message = if correction_to_codex.is_some() {
            None
        } else {
            Some(format_turn_ack_message(review_summary.as_str()))
        };

        if !notebook_changed_by_tools {
            let mut notebook = self.notebook.lock().await;
            notebook.apply_after_agent_update(
                format!("Reviewed guga-codex turn {turn_id}"),
                review_summary,
                "Awaiting next guga-codex review".to_string(),
            );
            let snapshot = notebook.clone();
            drop(notebook);
            self.persist_notebook(&snapshot).await;
        }

        // Keep warning output as a fallback when no actionable correction is available.
        let warning_message = if correction_to_codex.is_some() {
            None
        } else {
            violation_note
        };

        Some(SupervisorAfterAgentOutcome {
            warning_message,
            user_ack_message,
            correction_user_message,
            correction_to_codex,
            events: ui_events,
        })
    }

    #[allow(dead_code)]
    pub(crate) async fn search_history(&self, query: &str) -> Option<String> {
        let query = query.trim();
        if query.is_empty() {
            return Some("search_history(\"\"): query is empty".to_string());
        }
        let turns = self.load_history_turns().await?;
        let query_lower = query.to_ascii_lowercase();
        let mut matches = Vec::new();
        for turn in turns {
            if turn.content.to_ascii_lowercase().contains(&query_lower) {
                matches.push(turn);
            }
        }
        if matches.is_empty() {
            return Some(format!("search_history(\"{query}\"): No results"));
        }
        let mut lines = vec![format!(
            "search_history(\"{query}\"): {} results",
            matches.len()
        )];
        for (idx, turn) in matches.into_iter().take(10).enumerate() {
            lines.push(format!(
                "{}. [{} @ {}] {}",
                idx + 1,
                turn.role,
                turn.timestamp.to_rfc3339(),
                truncate_text(&turn.content, 240, "")
            ));
        }
        Some(lines.join("\n"))
    }

    #[allow(dead_code)]
    pub(crate) async fn read_recent(&self, count: usize) -> Option<String> {
        let turns = self.load_history_turns().await?;
        if turns.is_empty() {
            return Some("read_recent: No history available".to_string());
        }
        let count = count.clamp(1, 20);
        let start = turns.len().saturating_sub(count);
        let mut lines = vec![format!(
            "read_recent({count}): {} turns",
            turns.len().saturating_sub(start)
        )];
        for turn in &turns[start..] {
            lines.push(format!(
                "[{} @ {}] {}",
                turn.role,
                turn.timestamp.to_rfc3339(),
                turn.content
            ));
        }
        Some(lines.join("\n"))
    }

    #[allow(dead_code)]
    pub(crate) async fn read_turn(&self, index: usize) -> Option<String> {
        let turns = self.load_history_turns().await?;
        let Some(turn) = turns.get(index) else {
            return Some(format!("read_turn({index}): Turn not found"));
        };
        Some(format!(
            "read_turn({index}): [{} @ {}]\n{}",
            turn.role,
            turn.timestamp.to_rfc3339(),
            turn.content
        ))
    }

    #[allow(dead_code)]
    pub(crate) async fn history_stats(&self) -> Option<String> {
        let turns = self.load_history_turns().await?;
        let approx_tokens = turns
            .iter()
            .map(|turn| turn.content.len() / 4)
            .sum::<usize>();
        Some(format!(
            "history_stats: {} total turns in archive ({} approx tokens)",
            turns.len(),
            approx_tokens
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
        let snapshot = notebook.clone();
        drop(notebook);
        self.persist_notebook(&snapshot).await;
        None
    }

    pub(crate) async fn chat_with_user(
        &self,
        user_message: &str,
        turn_context: &TurnContext,
        model_client: &ModelClient,
    ) -> Result<SupervisorChatOutcome, String> {
        if !self.enabled {
            return Err("supervisor is disabled".to_string());
        }

        let user_message = user_message.trim();
        if user_message.is_empty() {
            return Err("user message is empty".to_string());
        }

        let mut client_session = model_client.new_session();
        let mut turn_items: Vec<ResponseItem> = Vec::new();
        let mut executed_tool_signatures: HashSet<String> = HashSet::new();
        let mut executed_tool_calls: usize = 0;
        let mut notebook_changed = false;
        let mut ui_events: Vec<EventMsg> = Vec::new();
        let turn_metadata_header = turn_context.turn_metadata_state.current_header_value();

        let reply = loop {
            let round = (ui_events
                .iter()
                .filter(|event| matches!(event, EventMsg::BackgroundEvent(_)))
                .count() as u32)
                + 1;
            if round > MAX_SUPERVISOR_FOLLOW_UP_ROUNDS {
                break "I reached the tool follow-up guard limit for this chat turn. Please restate the key point and I will continue concisely.".to_string();
            }

            let round_message = if round == 1 {
                "Supervising: Thinking...".to_string()
            } else {
                format!("Supervising: Thinking (tool follow-up #{})...", round - 1)
            };
            ui_events.push(EventMsg::BackgroundEvent(BackgroundEventEvent {
                message: round_message,
            }));

            let notebook_text = self.read_notebook_raw().await;
            let history_excerpt = self.build_recent_history_excerpt(16).await;
            let prompt_text = build_supervisor_chat_prompt(
                user_message,
                notebook_text.as_str(),
                history_excerpt.as_str(),
            );

            let mut input = vec![ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText { text: prompt_text }],
                end_turn: None,
                phase: None,
            }];
            input.extend(turn_items.clone());

            let prompt = Prompt {
                input,
                tools: supervisor_tools_schema(),
                parallel_tool_calls: false,
                base_instructions: BaseInstructions {
                    text: SUPERVISOR_CHAT_BASE_INSTRUCTIONS.to_string(),
                },
                personality: None,
                output_schema: None,
            };

            let mut stream = tokio::time::timeout(
                SUPERVISOR_STREAM_TIMEOUT,
                client_session.stream(
                    &prompt,
                    &turn_context.model_info,
                    &turn_context.otel_manager,
                    Some(ReasoningEffortConfig::Low),
                    turn_context.reasoning_summary,
                    turn_metadata_header.as_deref(),
                ),
            )
            .await
            .map_err(|_| {
                format!("supervisor chat stream start timed out in follow-up round {round}")
            })?
            .map_err(|err| format!("supervisor chat stream setup failed: {err}"))?;

            let mut response_text = String::new();
            let mut output_items = Vec::new();
            let mut completed = false;

            loop {
                let maybe_event = tokio::time::timeout(SUPERVISOR_STREAM_TIMEOUT, stream.next())
                    .await
                    .map_err(|_| {
                        format!("supervisor chat stream timed out in follow-up round {round}")
                    })?;
                let Some(event) = maybe_event else {
                    break;
                };

                match event.map_err(|err| format!("supervisor chat stream error: {err}"))? {
                    ResponseEvent::OutputTextDelta(delta) => response_text.push_str(&delta),
                    ResponseEvent::OutputItemDone(item) => {
                        if response_text.is_empty()
                            && let Some(text) = extract_response_item_text(&item)
                        {
                            response_text.push_str(&text);
                        }
                        output_items.push(item);
                    }
                    ResponseEvent::Completed { .. } => {
                        completed = true;
                        break;
                    }
                    _ => {}
                }
            }

            if !completed {
                return Err(format!(
                    "supervisor chat stream closed before completion in follow-up round {round}"
                ));
            }

            let tool_calls = extract_structured_tool_calls(&output_items);
            if tool_calls.is_empty() {
                break normalize_supervisor_chat_reply(response_text.as_str());
            }

            self.execute_structured_tool_calls_for_round(
                turn_context,
                round,
                &tool_calls,
                &mut turn_items,
                &mut executed_tool_signatures,
                &mut executed_tool_calls,
                &mut notebook_changed,
                &mut ui_events,
            )
            .await;
        };
        self.append_history_turn("user_to_guga_codex", user_message)
            .await;
        self.append_history_turn("guga-codex", reply.as_str()).await;
        Ok(SupervisorChatOutcome {
            reply,
            events: ui_events,
            notebook_changed,
        })
    }

    async fn run_supervisor_pass(
        &self,
        turn_context: &TurnContext,
        model_client: &ModelClient,
        input_messages: &[String],
        last_assistant_message: &str,
    ) -> Result<SupervisorPassResult, String> {
        if last_assistant_message.trim().is_empty() {
            return Ok(SupervisorPassResult {
                decision: SupervisorDecision::ok(
                    "Codex produced no final text; keeping conservative OK.".to_string(),
                ),
                notebook_changed: false,
                ui_events: Vec::new(),
            });
        }

        let mut client_session = model_client.new_session();
        let mut turn_items: Vec<ResponseItem> = Vec::new();
        let mut executed_tool_signatures: HashSet<String> = HashSet::new();
        let mut executed_tool_calls: usize = 0;
        let mut notebook_changed = false;
        let mut ui_events: Vec<EventMsg> = Vec::new();
        let turn_metadata_header = turn_context.turn_metadata_state.current_header_value();

        for round in 1..=MAX_SUPERVISOR_FOLLOW_UP_ROUNDS {
            let round_message = if round == 1 {
                "Supervising: Analyzing completed turn...".to_string()
            } else {
                format!("Supervising: Analyzing (tool follow-up #{})...", round - 1)
            };
            ui_events.push(EventMsg::BackgroundEvent(BackgroundEventEvent {
                message: round_message,
            }));

            let notebook_text = self.read_notebook_raw().await;
            let history_excerpt = self.build_recent_history_excerpt(12).await;
            let prompt_text = build_supervisor_prompt(
                input_messages,
                last_assistant_message,
                notebook_text.as_str(),
                history_excerpt.as_str(),
            );

            let mut input = vec![ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText { text: prompt_text }],
                end_turn: None,
                phase: None,
            }];
            input.extend(turn_items.clone());

            let prompt = Prompt {
                input,
                tools: supervisor_tools_schema(),
                parallel_tool_calls: false,
                base_instructions: BaseInstructions {
                    text: SUPERVISOR_BASE_INSTRUCTIONS.to_string(),
                },
                personality: None,
                output_schema: None,
            };

            let mut stream = tokio::time::timeout(
                SUPERVISOR_STREAM_TIMEOUT,
                client_session.stream(
                    &prompt,
                    &turn_context.model_info,
                    &turn_context.otel_manager,
                    Some(ReasoningEffortConfig::Low),
                    turn_context.reasoning_summary,
                    turn_metadata_header.as_deref(),
                ),
            )
            .await
            .map_err(|_| format!("supervisor stream start timed out in follow-up round {round}"))?
            .map_err(|err| format!("supervisor stream setup failed: {err}"))?;

            let mut response_text = String::new();
            let mut output_items = Vec::new();
            let mut completed = false;

            loop {
                let maybe_event = tokio::time::timeout(SUPERVISOR_STREAM_TIMEOUT, stream.next())
                    .await
                    .map_err(|_| {
                        format!("supervisor stream timed out in follow-up round {round}")
                    })?;

                let Some(event) = maybe_event else {
                    break;
                };

                match event.map_err(|err| format!("supervisor stream error: {err}"))? {
                    ResponseEvent::OutputTextDelta(delta) => response_text.push_str(&delta),
                    ResponseEvent::OutputItemDone(item) => {
                        if response_text.is_empty()
                            && let Some(text) = extract_response_item_text(&item)
                        {
                            response_text.push_str(&text);
                        }
                        output_items.push(item);
                    }
                    ResponseEvent::Completed { .. } => {
                        completed = true;
                        break;
                    }
                    _ => {}
                }
            }

            if !completed {
                return Err(format!(
                    "supervisor stream closed before completion in follow-up round {round}"
                ));
            }

            let tool_calls = extract_structured_tool_calls(&output_items);
            if tool_calls.is_empty() {
                let decision = parse_supervisor_decision(&response_text);
                return Ok(SupervisorPassResult {
                    decision,
                    notebook_changed,
                    ui_events,
                });
            }

            self.execute_structured_tool_calls_for_round(
                turn_context,
                round,
                &tool_calls,
                &mut turn_items,
                &mut executed_tool_signatures,
                &mut executed_tool_calls,
                &mut notebook_changed,
                &mut ui_events,
            )
            .await;
        }

        Ok(SupervisorPassResult {
            decision: SupervisorDecision::ok(
                "Supervisor reached follow-up guard limit and returned conservative OK."
                    .to_string(),
            ),
            notebook_changed,
            ui_events,
        })
    }

    async fn execute_structured_tool_calls_for_round(
        &self,
        turn_context: &TurnContext,
        round: u32,
        tool_calls: &[StructuredToolCall],
        turn_items: &mut Vec<ResponseItem>,
        executed_tool_signatures: &mut HashSet<String>,
        executed_tool_calls: &mut usize,
        notebook_changed: &mut bool,
        ui_events: &mut Vec<EventMsg>,
    ) {
        for tool_call in tool_calls {
            turn_items.push(tool_call.item.clone());
            let signature = format!("{}::{}", tool_call.tool_name, tool_call.arguments);
            let mut command = supervisor_tool_display_prefix(tool_call.tool_name.as_str());
            let tool_call_id = format!("guga-codex-supervisor-{round}-{}", tool_call.call_id);
            let turn_id = turn_context.sub_id.clone();
            let cwd = turn_context.cwd.clone();

            if !executed_tool_signatures.insert(signature) {
                let output = "duplicate tool call skipped in same turn".to_string();
                let args_preview = tool_args_preview(tool_call.arguments.as_str());
                if !args_preview.is_empty() {
                    command.push(args_preview);
                }
                let parsed_cmd = vec![ParsedCommand::Unknown {
                    cmd: command.join(" "),
                }];
                ui_events.push(EventMsg::ExecCommandBegin(ExecCommandBeginEvent {
                    call_id: tool_call_id.clone(),
                    process_id: None,
                    turn_id: turn_id.clone(),
                    command: command.clone(),
                    cwd: cwd.clone(),
                    parsed_cmd: parsed_cmd.clone(),
                    source: ExecCommandSource::Agent,
                    interaction_input: None,
                }));
                ui_events.push(EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                    call_id: tool_call_id,
                    process_id: None,
                    turn_id,
                    command,
                    cwd,
                    parsed_cmd,
                    source: ExecCommandSource::Agent,
                    interaction_input: None,
                    stdout: String::new(),
                    stderr: output.clone(),
                    aggregated_output: output.clone(),
                    exit_code: 1,
                    duration: Duration::from_millis(0),
                    formatted_output: output.clone(),
                    status: ExecCommandStatus::Failed,
                }));
                turn_items.push(ResponseItem::FunctionCallOutput {
                    call_id: tool_call.call_id.clone(),
                    output: FunctionCallOutputPayload::from_text(output),
                });
                continue;
            }

            if *executed_tool_calls >= MAX_SUPERVISOR_TOOL_CALLS {
                let output = "tool-call limit reached for this turn".to_string();
                let args_preview = tool_args_preview(tool_call.arguments.as_str());
                if !args_preview.is_empty() {
                    command.push(args_preview);
                }
                let parsed_cmd = vec![ParsedCommand::Unknown {
                    cmd: command.join(" "),
                }];
                ui_events.push(EventMsg::ExecCommandBegin(ExecCommandBeginEvent {
                    call_id: tool_call_id.clone(),
                    process_id: None,
                    turn_id: turn_id.clone(),
                    command: command.clone(),
                    cwd: cwd.clone(),
                    parsed_cmd: parsed_cmd.clone(),
                    source: ExecCommandSource::Agent,
                    interaction_input: None,
                }));
                ui_events.push(EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                    call_id: tool_call_id,
                    process_id: None,
                    turn_id,
                    command,
                    cwd,
                    parsed_cmd,
                    source: ExecCommandSource::Agent,
                    interaction_input: None,
                    stdout: String::new(),
                    stderr: output.clone(),
                    aggregated_output: output.clone(),
                    exit_code: 1,
                    duration: Duration::from_millis(0),
                    formatted_output: output.clone(),
                    status: ExecCommandStatus::Failed,
                }));
                turn_items.push(ResponseItem::FunctionCallOutput {
                    call_id: tool_call.call_id.clone(),
                    output: FunctionCallOutputPayload::from_text(output),
                });
                continue;
            }

            *executed_tool_calls += 1;
            let started = std::time::Instant::now();
            let (output, args_for_display, success, exit_code, status) =
                match Self::normalize_tool_arguments(
                    tool_call.tool_name.as_str(),
                    tool_call.arguments.as_str(),
                ) {
                    Ok(normalized_args) => {
                        let output = self
                            .execute_tool_call(
                                tool_call.tool_name.as_str(),
                                normalized_args.as_str(),
                                turn_context.cwd.as_path(),
                            )
                            .await;
                        (
                            output,
                            normalized_args,
                            true,
                            0,
                            ExecCommandStatus::Completed,
                        )
                    }
                    Err(err) => (
                        format!("{}: Invalid arguments: {err}", tool_call.tool_name),
                        tool_call.arguments.clone(),
                        false,
                        1,
                        ExecCommandStatus::Failed,
                    ),
                };
            let duration = started.elapsed();

            if tool_call.tool_name == "apply_patch_notebook"
                && let Some(payload) = self.build_notebook_patch_render_payload(output.as_str())
            {
                *notebook_changed = true;
                ui_events.push(EventMsg::PatchApplyBegin(PatchApplyBeginEvent {
                    call_id: tool_call_id.clone(),
                    turn_id: turn_id.clone(),
                    auto_approved: true,
                    changes: payload.changes.clone(),
                }));
                ui_events.push(EventMsg::PatchApplyEnd(PatchApplyEndEvent {
                    call_id: tool_call_id,
                    turn_id,
                    stdout: payload.summary,
                    stderr: String::new(),
                    success: true,
                    changes: payload.changes,
                    status: PatchApplyStatus::Completed,
                }));
                ui_events.push(EventMsg::TurnDiff(TurnDiffEvent {
                    unified_diff: payload.turn_diff,
                }));
                turn_items.push(ResponseItem::FunctionCallOutput {
                    call_id: tool_call.call_id.clone(),
                    output: FunctionCallOutputPayload::from_text(output),
                });
                continue;
            }

            if tool_call.tool_name == "apply_patch_notebook"
                && let Some((summary, patch_success, patch_status)) =
                    notebook_patch_status_from_output(output.as_str())
            {
                if patch_success && summary.contains("Applied") {
                    *notebook_changed = true;
                }
                ui_events.push(EventMsg::PatchApplyEnd(PatchApplyEndEvent {
                    call_id: tool_call_id,
                    turn_id,
                    stdout: summary.clone(),
                    stderr: if patch_success {
                        String::new()
                    } else {
                        summary
                    },
                    success: patch_success,
                    changes: HashMap::new(),
                    status: patch_status,
                }));
                turn_items.push(ResponseItem::FunctionCallOutput {
                    call_id: tool_call.call_id.clone(),
                    output: FunctionCallOutputPayload::from_text(output),
                });
                continue;
            }

            let args_preview = tool_args_preview(args_for_display.as_str());
            if !args_preview.is_empty() {
                command.push(args_preview);
            }
            let parsed_cmd = supervisor_tool_parsed_command(
                tool_call.tool_name.as_str(),
                args_for_display.as_str(),
                cwd.as_path(),
                self.notebook_path.as_ref(),
                success,
                command.as_slice(),
            );
            ui_events.push(EventMsg::ExecCommandBegin(ExecCommandBeginEvent {
                call_id: tool_call_id.clone(),
                process_id: None,
                turn_id: turn_id.clone(),
                command: command.clone(),
                cwd: cwd.clone(),
                parsed_cmd: parsed_cmd.clone(),
                source: ExecCommandSource::Agent,
                interaction_input: None,
            }));
            ui_events.push(EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                call_id: tool_call_id,
                process_id: None,
                turn_id,
                command,
                cwd,
                parsed_cmd,
                source: ExecCommandSource::Agent,
                interaction_input: None,
                stdout: if success {
                    truncate_text(output.as_str(), TOOL_OUTPUT_LIMIT, "(no output)")
                } else {
                    String::new()
                },
                stderr: if success {
                    String::new()
                } else {
                    truncate_text(output.as_str(), TOOL_OUTPUT_LIMIT, "")
                },
                aggregated_output: output.clone(),
                exit_code,
                duration,
                formatted_output: output.clone(),
                status,
            }));

            if tool_call.tool_name == "apply_patch_notebook"
                && output.contains("apply_patch_notebook: Applied")
            {
                *notebook_changed = true;
            }

            turn_items.push(ResponseItem::FunctionCallOutput {
                call_id: tool_call.call_id.clone(),
                output: FunctionCallOutputPayload::from_text(output),
            });
        }
    }

    fn build_notebook_patch_render_payload(
        &self,
        output: &str,
    ) -> Option<NotebookPatchRenderPayload> {
        let trimmed = output.trim();
        let (summary, diff_body) = trimmed.split_once('\n')?;
        if !summary.starts_with("apply_patch_notebook: Applied") {
            return None;
        }

        let diff_body = diff_body.trim();
        if !diff_body.starts_with("@@") {
            return None;
        }

        let notebook_path = self
            .notebook_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("guga-codex/notebook.json"));
        let notebook_path_display = notebook_path.to_string_lossy().replace('\\', "/");
        let mut changes = HashMap::new();
        changes.insert(
            notebook_path.clone(),
            FileChange::Update {
                unified_diff: diff_body.to_string(),
                move_path: None,
            },
        );

        Some(NotebookPatchRenderPayload {
            summary: summary.to_string(),
            changes,
            turn_diff: format!(
                "diff --git a/{0} b/{0}\n--- a/{0}\n+++ b/{0}\n{1}",
                notebook_path_display, diff_body
            ),
        })
    }

    fn normalize_tool_arguments(tool_name: &str, raw_args: &str) -> Result<String, String> {
        let value: serde_json::Value = if raw_args.trim().is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(raw_args).map_err(|e| format!("invalid JSON: {e}"))?
        };

        let pick_str = |keys: &[&str]| -> Option<String> {
            keys.iter().find_map(|key| {
                value
                    .get(*key)
                    .and_then(|v| v.as_str())
                    .map(ToOwned::to_owned)
            })
        };

        let pick_usize = |keys: &[&str]| -> Option<usize> {
            keys.iter().find_map(|key| {
                value
                    .get(*key)
                    .and_then(serde_json::Value::as_u64)
                    .map(|n| n as usize)
            })
        };

        match tool_name {
            "search_history" => pick_str(&["query", "keyword", "text", "pattern"])
                .ok_or_else(|| "missing query".to_string()),
            "read_recent" => Ok(pick_usize(&["count", "n"]).unwrap_or(5).to_string()),
            "read_turn" => pick_usize(&["index"])
                .map(|v| v.to_string())
                .ok_or_else(|| "missing index".to_string()),
            "history_stats" | "read_notebook" => Ok(String::new()),
            "apply_patch_notebook" => {
                pick_str(&["patch"]).ok_or_else(|| "missing patch".to_string())
            }
            "read_file" => {
                let path = pick_str(&["path"]).ok_or_else(|| "missing path".to_string())?;
                let offset = pick_usize(&["offset"]).unwrap_or(1);
                let limit = pick_usize(&["limit"]).unwrap_or(100);
                if value.get("offset").is_some() || value.get("limit").is_some() {
                    Ok(format!("{path}|{offset}|{limit}"))
                } else {
                    Ok(path)
                }
            }
            "glob" => pick_str(&["pattern"]).ok_or_else(|| "missing pattern".to_string()),
            "shell" => pick_str(&["cmd", "command"]).ok_or_else(|| "missing cmd".to_string()),
            "rg" => pick_str(&["pattern"]).ok_or_else(|| "missing pattern".to_string()),
            "ls" => pick_str(&["path"]).ok_or_else(|| "missing path".to_string()),
            _ => {
                if let Some(s) = value.as_str() {
                    Ok(s.to_string())
                } else {
                    Ok(value.to_string())
                }
            }
        }
    }

    async fn execute_tool_call(&self, tool_name: &str, args: &str, cwd: &Path) -> String {
        match tool_name {
            "search_history" => self
                .search_history(args)
                .await
                .unwrap_or_else(|| format!("search_history(\"{args}\"): Search failed")),
            "read_recent" => {
                let n: usize = args.parse().unwrap_or(5).clamp(1, 20);
                self.read_recent(n)
                    .await
                    .unwrap_or_else(|| format!("read_recent({n}): Error"))
            }
            "read_turn" => {
                let index: usize = match args.parse() {
                    Ok(i) => i,
                    Err(_) => return format!("read_turn(\"{args}\"): Invalid index"),
                };
                self.read_turn(index)
                    .await
                    .unwrap_or_else(|| format!("read_turn({index}): Error"))
            }
            "history_stats" => self
                .history_stats()
                .await
                .unwrap_or_else(|| "history_stats: Error".to_string()),
            "read_notebook" => {
                let content = self.read_notebook_raw().await;
                if content.trim().is_empty() {
                    "read_notebook: notebook is empty".to_string()
                } else {
                    format!("read_notebook:\n{content}")
                }
            }
            "apply_patch_notebook" => self.handle_apply_patch_notebook(args).await,
            "read_file" => {
                let parts: Vec<&str> = args.splitn(3, '|').collect();
                let path = parts.first().copied().unwrap_or_default().trim();
                let offset = parts
                    .get(1)
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .unwrap_or(1);
                let limit = parts
                    .get(2)
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .unwrap_or(100);

                match Self::read_file_lines(cwd, path, offset, limit).await {
                    Ok(content) => format!("read_file(\"{path}\", {offset}, {limit}):\n{content}"),
                    Err(e) => format!("read_file(\"{path}\"): Error: {e}"),
                }
            }
            "glob" => match Self::glob_files(cwd, args).await {
                Ok(files) => {
                    if files.is_empty() {
                        format!("glob(\"{args}\"): No matches")
                    } else {
                        let display: Vec<String> = files
                            .iter()
                            .take(30)
                            .map(|p| p.display().to_string())
                            .collect();
                        let suffix = if files.len() > 30 {
                            format!("\n... and {} more", files.len() - 30)
                        } else {
                            String::new()
                        };
                        format!(
                            "glob(\"{args}\"): {} matches\n{}{}",
                            files.len(),
                            display.join("\n"),
                            suffix
                        )
                    }
                }
                Err(e) => format!("glob(\"{args}\"): Error: {e}"),
            },
            "shell" | "rg" | "grep" => {
                let cmd = if tool_name == "rg" || tool_name == "grep" {
                    format!("rg {args}")
                } else {
                    args.to_string()
                };

                match Self::execute_shell(cwd, &cmd).await {
                    Ok(output) => format!("shell(\"{cmd}\"):\n{output}"),
                    Err(e) => format!("shell(\"{cmd}\"): Error: {e}"),
                }
            }
            "ls" => {
                let cmd = format!("ls -la {args}");
                match Self::execute_shell(cwd, &cmd).await {
                    Ok(output) => format!("ls(\"{args}\"):\n{output}"),
                    Err(e) => format!("ls(\"{args}\"): Error: {e}"),
                }
            }
            _ => format!("Unknown tool: {tool_name}"),
        }
    }

    async fn read_file_lines(
        cwd: &Path,
        path: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, String> {
        if path.trim().is_empty() {
            return Err("empty path".to_string());
        }

        let resolved_path = resolve_path(cwd, path);
        let content = tokio::fs::read_to_string(&resolved_path)
            .await
            .map_err(|e| format!("Failed to read file: {e}"))?;

        let lines: Vec<&str> = content.lines().collect();
        let safe_limit = limit.clamp(1, 500);
        let start = offset.saturating_sub(1).min(lines.len());
        let end = (start + safe_limit).min(lines.len());

        let result: Vec<String> = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("L{}: {}", start + i + 1, line))
            .collect();

        Ok(result.join("\n"))
    }

    async fn glob_files(cwd: &Path, pattern: &str) -> Result<Vec<PathBuf>, String> {
        let cwd = cwd.to_path_buf();
        let pattern = pattern.trim().to_string();
        tokio::task::spawn_blocking(move || glob_files_blocking(&cwd, &pattern))
            .await
            .map_err(|e| format!("glob worker failed: {e}"))?
    }

    async fn execute_shell(cwd: &Path, cmd: &str) -> Result<String, String> {
        let parts = shlex::split(cmd).ok_or_else(|| "failed to parse command".to_string())?;
        if parts.is_empty() {
            return Err("empty command".to_string());
        }

        let program = std::path::Path::new(parts[0].as_str())
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(parts[0].as_str());

        if !Self::is_safe_command(program, &parts) {
            return Err(format!("Command '{program}' is not in the safe whitelist"));
        }

        let output = Command::new("sh")
            .arg("-lc")
            .arg(cmd)
            .current_dir(cwd)
            .output()
            .await
            .map_err(|e| format!("Failed to execute: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();

        if !stdout.is_empty() {
            result.push_str(&truncate_text(&stdout, TOOL_OUTPUT_LIMIT, "(no output)"));
        }

        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push_str("\n[stderr]\n");
            }
            result.push_str(&truncate_text(&stderr, TOOL_OUTPUT_LIMIT, ""));
        }

        if result.trim().is_empty() {
            result = "(no output)".to_string();
        }

        Ok(result)
    }

    fn is_safe_command(program: &str, args: &[String]) -> bool {
        match program {
            "cat" | "cd" | "cut" | "echo" | "expr" | "false" | "grep" | "head" | "id" | "ls"
            | "nl" | "paste" | "pwd" | "rev" | "seq" | "stat" | "tail" | "tr" | "true"
            | "uname" | "uniq" | "wc" | "which" | "whoami" => true,
            "numfmt" | "tac" => cfg!(target_os = "linux"),
            "base64" => !args.iter().any(|arg| {
                arg == "--output" || arg.starts_with("--output=") || arg.starts_with("-o")
            }),
            "find" => {
                let unsafe_opts = [
                    "-exec", "-execdir", "-ok", "-okdir", "-delete", "-fls", "-fprint", "-fprint0",
                    "-fprintf",
                ];
                !args.iter().any(|arg| unsafe_opts.contains(&arg.as_str()))
            }
            "rg" => !args.iter().any(|arg| {
                arg == "--search-zip"
                    || arg == "-z"
                    || arg == "--pre"
                    || arg.starts_with("--pre=")
                    || arg == "--hostname-bin"
                    || arg.starts_with("--hostname-bin=")
            }),
            "git" => matches!(
                args.get(1).map(String::as_str),
                Some("branch") | Some("status") | Some("log") | Some("diff") | Some("show")
            ),
            "cargo" => matches!(args.get(1).map(String::as_str), Some("check")),
            "sed" => {
                args.len() <= 4
                    && args.get(1).map(String::as_str) == Some("-n")
                    && args.get(2).is_some_and(|arg| {
                        arg.ends_with('p')
                            && arg
                                .trim_end_matches('p')
                                .chars()
                                .all(|c| c.is_ascii_digit() || c == ',')
                    })
            }
            _ => false,
        }
    }

    async fn handle_apply_patch_notebook(&self, patch: &str) -> String {
        let patch = patch.trim();
        if patch.is_empty() {
            return "apply_patch_notebook: patch is empty".to_string();
        }

        let hunks = match Self::parse_apply_patch_hunks(patch) {
            Ok(hunks) => hunks,
            Err(err) => return format!("apply_patch_notebook: Invalid patch format: {err}"),
        };

        let mut notebook = self.notebook.lock().await;
        let original = match serde_json::to_string_pretty(&*notebook) {
            Ok(content) => content,
            Err(err) => return format!("apply_patch_notebook: Failed to read notebook: {err}"),
        };

        let (updated, applied_hunks) = match Self::apply_patch_hunks(&original, &hunks) {
            Ok(result) => result,
            Err(err) => return format!("apply_patch_notebook: {err}"),
        };

        if updated == original {
            return "apply_patch_notebook: No changes applied".to_string();
        }

        let updated_value: serde_json::Value = match serde_json::from_str(&updated) {
            Ok(value) => value,
            Err(err) => {
                return format!(
                    "apply_patch_notebook: Patch result is not valid notebook JSON: {err}"
                );
            }
        };

        let now = Utc::now();
        let parsed = match Self::normalize_notebook_value(&updated_value, now) {
            Ok(parsed) => parsed,
            Err(report) => return report.into_message(),
        };

        let normalized = match serde_json::to_string_pretty(&parsed) {
            Ok(content) => content,
            Err(err) => {
                return format!("apply_patch_notebook: Failed to serialize notebook: {err}");
            }
        };

        *notebook = parsed;
        let snapshot = notebook.clone();
        drop(notebook);
        self.persist_notebook(&snapshot).await;

        let diff = Self::build_notebook_diff(
            &NotebookSnapshot {
                raw: original.clone(),
            },
            &NotebookSnapshot { raw: normalized },
        );

        match diff {
            Some(unified_diff) => {
                format!("apply_patch_notebook: Applied {applied_hunks} hunk(s)\n{unified_diff}")
            }
            None => format!("apply_patch_notebook: Applied {applied_hunks} hunk(s)"),
        }
    }

    fn normalize_notebook_value(
        value: &serde_json::Value,
        now: DateTime<Utc>,
    ) -> Result<SupervisorNotebook, NotebookValidationReport> {
        let mut report = NotebookValidationReport::default();
        let Some(root) = value.as_object() else {
            report.push("$", "invalid type", "object");
            return Err(report);
        };

        Self::validate_allowed_keys(
            root,
            "$",
            &[
                "current_activity",
                "completed",
                "attention",
                "mistakes",
                "last_updated",
            ],
            &mut report,
        );

        let current_activity = match root.get("current_activity") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(value)) => Some(value.clone()),
            Some(_) => {
                report.push("$.current_activity", "invalid type", "string or null");
                None
            }
        };

        let completed =
            Self::normalize_completed_items(root.get("completed"), now.clone(), &mut report);
        let attention =
            Self::normalize_attention_items(root.get("attention"), now.clone(), &mut report);
        let mistakes =
            Self::normalize_mistake_items(root.get("mistakes"), now.clone(), &mut report);

        if !report.is_empty() {
            return Err(report);
        }

        Ok(SupervisorNotebook {
            current_activity,
            completed,
            attention,
            mistakes,
            last_updated: Some(now),
        })
    }

    fn normalize_completed_items(
        value: Option<&serde_json::Value>,
        now: DateTime<Utc>,
        report: &mut NotebookValidationReport,
    ) -> Vec<CompletedItem> {
        let Some(value) = value else {
            return Vec::new();
        };
        if value.is_null() {
            return Vec::new();
        }

        let Some(items) = value.as_array() else {
            report.push("$.completed", "invalid type", "array of objects");
            return Vec::new();
        };

        let mut out = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let item_path = format!("$.completed[{index}]");
            let Some(item_obj) = item.as_object() else {
                report.push(item_path, "invalid item type", "object");
                continue;
            };

            Self::validate_allowed_keys(
                item_obj,
                &item_path,
                &["what", "significance", "timestamp"],
                report,
            );

            let what = Self::require_string_field(item_obj, &item_path, "what", report);
            let significance =
                Self::require_string_field(item_obj, &item_path, "significance", report);
            if let (Some(what), Some(significance)) = (what, significance) {
                out.push(CompletedItem {
                    timestamp: now.clone(),
                    what,
                    significance,
                });
            }
        }

        out
    }

    fn normalize_attention_items(
        value: Option<&serde_json::Value>,
        now: DateTime<Utc>,
        report: &mut NotebookValidationReport,
    ) -> Vec<AttentionItem> {
        let Some(value) = value else {
            return Vec::new();
        };
        if value.is_null() {
            return Vec::new();
        }

        let Some(items) = value.as_array() else {
            report.push("$.attention", "invalid type", "array of objects");
            return Vec::new();
        };

        let mut out = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let item_path = format!("$.attention[{index}]");
            let Some(item_obj) = item.as_object() else {
                report.push(item_path, "invalid item type", "object");
                continue;
            };

            Self::validate_allowed_keys(
                item_obj,
                &item_path,
                &["content", "source", "priority", "added_at"],
                report,
            );

            let content = Self::require_string_field(item_obj, &item_path, "content", report);
            let source = Self::parse_attention_source(item_obj, &item_path, report);
            let priority = Self::parse_attention_priority(item_obj, &item_path, report);
            if let Some(content) = content {
                out.push(AttentionItem {
                    content,
                    source,
                    priority,
                    added_at: now.clone(),
                });
            }
        }

        out
    }

    fn normalize_mistake_items(
        value: Option<&serde_json::Value>,
        now: DateTime<Utc>,
        report: &mut NotebookValidationReport,
    ) -> Vec<MistakeEntry> {
        let Some(value) = value else {
            return Vec::new();
        };
        if value.is_null() {
            return Vec::new();
        }

        let Some(items) = value.as_array() else {
            report.push("$.mistakes", "invalid type", "array of objects");
            return Vec::new();
        };

        let mut out = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let item_path = format!("$.mistakes[{index}]");
            let Some(item_obj) = item.as_object() else {
                report.push(item_path, "invalid item type", "object");
                continue;
            };

            Self::validate_allowed_keys(
                item_obj,
                &item_path,
                &["what_happened", "how_corrected", "lesson", "timestamp"],
                report,
            );

            let what_happened =
                Self::require_string_field(item_obj, &item_path, "what_happened", report);
            let how_corrected =
                Self::require_string_field(item_obj, &item_path, "how_corrected", report);
            let lesson = Self::require_string_field(item_obj, &item_path, "lesson", report);

            if let (Some(what_happened), Some(how_corrected), Some(lesson)) =
                (what_happened, how_corrected, lesson)
            {
                out.push(MistakeEntry {
                    timestamp: now.clone(),
                    what_happened,
                    how_corrected,
                    lesson,
                });
            }
        }

        out
    }

    fn validate_allowed_keys(
        object: &serde_json::Map<String, serde_json::Value>,
        base_path: &str,
        allowed: &[&str],
        report: &mut NotebookValidationReport,
    ) {
        let expected = format!("one of: {}", allowed.join(", "));
        for key in object.keys() {
            if !allowed.contains(&key.as_str()) {
                report.push(
                    format!("{base_path}.{key}"),
                    "unknown field",
                    expected.clone(),
                );
            }
        }
    }

    fn require_string_field(
        object: &serde_json::Map<String, serde_json::Value>,
        base_path: &str,
        field: &str,
        report: &mut NotebookValidationReport,
    ) -> Option<String> {
        let path = format!("{base_path}.{field}");
        match object.get(field) {
            Some(serde_json::Value::String(value)) if !value.trim().is_empty() => {
                Some(value.clone())
            }
            Some(serde_json::Value::String(_)) => {
                report.push(path, "empty string", "non-empty string");
                None
            }
            Some(_) => {
                report.push(path, "invalid type", "string");
                None
            }
            None => {
                report.push(path, "missing field", "string");
                None
            }
        }
    }

    fn parse_attention_source(
        object: &serde_json::Map<String, serde_json::Value>,
        base_path: &str,
        report: &mut NotebookValidationReport,
    ) -> AttentionSource {
        let path = format!("{base_path}.source");
        match object.get("source") {
            None | Some(serde_json::Value::Null) => AttentionSource::Inference,
            Some(serde_json::Value::String(raw)) => {
                match raw.trim().to_ascii_lowercase().as_str() {
                    "inference" => AttentionSource::Inference,
                    "mistake" => AttentionSource::Mistake,
                    "user_instruction" | "user" => AttentionSource::UserInstruction,
                    _ => {
                        report.push(
                            path,
                            format!("invalid value '{raw}'"),
                            "user_instruction | mistake | inference",
                        );
                        AttentionSource::Inference
                    }
                }
            }
            Some(_) => {
                report.push(path, "invalid type", "string");
                AttentionSource::Inference
            }
        }
    }

    fn parse_attention_priority(
        object: &serde_json::Map<String, serde_json::Value>,
        base_path: &str,
        report: &mut NotebookValidationReport,
    ) -> Priority {
        let path = format!("{base_path}.priority");
        match object.get("priority") {
            None | Some(serde_json::Value::Null) => Priority::Medium,
            Some(serde_json::Value::String(raw)) => {
                match raw.trim().to_ascii_lowercase().as_str() {
                    "high" => Priority::High,
                    "medium" => Priority::Medium,
                    "low" => Priority::Low,
                    _ => {
                        report.push(
                            path,
                            format!("invalid value '{raw}'"),
                            "high | medium | low",
                        );
                        Priority::Medium
                    }
                }
            }
            Some(_) => {
                report.push(path, "invalid type", "string");
                Priority::Medium
            }
        }
    }

    fn parse_apply_patch_hunks(patch: &str) -> Result<Vec<NotebookPatchHunk>, String> {
        let lines: Vec<&str> = patch.lines().collect();
        if lines.first() != Some(&"*** Begin Patch") {
            return Err("missing '*** Begin Patch' header".to_string());
        }
        if lines.last() != Some(&"*** End Patch") {
            return Err("missing '*** End Patch' footer".to_string());
        }

        let mut index = 1usize;
        let mut saw_update_file = false;
        let mut hunks = Vec::new();

        while index + 1 < lines.len() {
            let line = lines[index];
            if line.starts_with("*** Update File:") {
                saw_update_file = true;
                index += 1;
                continue;
            }
            if line.starts_with("*** Add File:")
                || line.starts_with("*** Delete File:")
                || line.starts_with("*** Move to:")
            {
                return Err("only '*** Update File:' patches are supported".to_string());
            }
            if line.starts_with("@@") {
                index += 1;
                let mut old_lines = Vec::new();
                let mut new_lines = Vec::new();

                while index + 1 < lines.len() {
                    let hunk_line = lines[index];
                    if hunk_line.starts_with("@@")
                        || hunk_line.starts_with("*** Update File:")
                        || hunk_line == "*** End Patch"
                    {
                        break;
                    }
                    if hunk_line == "*** End of File" || hunk_line == r"\ No newline at end of file"
                    {
                        index += 1;
                        continue;
                    }
                    if hunk_line.is_empty() {
                        return Err(
                            "empty hunk line must include prefix (' ', '+', '-')".to_string()
                        );
                    }

                    let (prefix, body) = hunk_line.split_at(1);
                    match prefix {
                        " " => {
                            old_lines.push(body.to_string());
                            new_lines.push(body.to_string());
                        }
                        "-" => old_lines.push(body.to_string()),
                        "+" => new_lines.push(body.to_string()),
                        _ => {
                            return Err(format!("unsupported hunk line prefix: '{hunk_line}'"));
                        }
                    }
                    index += 1;
                }

                if old_lines.is_empty() && new_lines.is_empty() {
                    return Err("empty hunk body".to_string());
                }
                hunks.push(NotebookPatchHunk {
                    old_lines,
                    new_lines,
                });
                continue;
            }

            if line.trim().is_empty() {
                index += 1;
                continue;
            }

            return Err(format!("unexpected patch line: '{line}'"));
        }

        if !saw_update_file {
            return Err("missing '*** Update File:' section".to_string());
        }
        if hunks.is_empty() {
            return Err("patch does not contain any hunks".to_string());
        }

        Ok(hunks)
    }

    fn apply_patch_hunks(
        original: &str,
        hunks: &[NotebookPatchHunk],
    ) -> Result<(String, usize), String> {
        let mut lines: Vec<String> = original.lines().map(ToOwned::to_owned).collect();
        let trailing_newline = original.ends_with('\n');
        let mut search_start = 0usize;

        for (idx, hunk) in hunks.iter().enumerate() {
            if hunk.old_lines.is_empty() {
                let insert_at = search_start.min(lines.len());
                lines.splice(insert_at..insert_at, hunk.new_lines.clone());
                search_start = insert_at + hunk.new_lines.len();
                continue;
            }

            let Some(start) = Self::find_hunk_start(&lines, &hunk.old_lines, search_start)
                .or_else(|| Self::find_hunk_start(&lines, &hunk.old_lines, 0))
            else {
                return Err(format!(
                    "failed to apply hunk {}: target block not found",
                    idx + 1
                ));
            };

            let end = start + hunk.old_lines.len();
            lines.splice(start..end, hunk.new_lines.clone());
            search_start = start + hunk.new_lines.len();
        }

        let mut merged = lines.join("\n");
        if trailing_newline && !merged.ends_with('\n') {
            merged.push('\n');
        }

        Ok((merged, hunks.len()))
    }

    fn find_hunk_start(lines: &[String], needle: &[String], from: usize) -> Option<usize> {
        if needle.is_empty() {
            return Some(from.min(lines.len()));
        }
        if lines.len() < needle.len() || from > lines.len().saturating_sub(needle.len()) {
            return None;
        }

        (from..=lines.len() - needle.len()).find(|&start| {
            lines[start..start + needle.len()]
                .iter()
                .map(String::as_str)
                .eq(needle.iter().map(String::as_str))
        })
    }

    fn build_notebook_diff(before: &NotebookSnapshot, after: &NotebookSnapshot) -> Option<String> {
        if before.raw == after.raw {
            return None;
        }

        let before_lines: Vec<&str> = before.raw.lines().collect();
        let after_lines: Vec<&str> = after.raw.lines().collect();
        Some(Self::build_unified_diff(&before_lines, &after_lines))
    }

    fn build_unified_diff(before_lines: &[&str], after_lines: &[&str]) -> String {
        let mut prefix = 0usize;
        while prefix < before_lines.len()
            && prefix < after_lines.len()
            && before_lines[prefix] == after_lines[prefix]
        {
            prefix += 1;
        }

        let mut suffix = 0usize;
        while suffix < before_lines.len().saturating_sub(prefix)
            && suffix < after_lines.len().saturating_sub(prefix)
            && before_lines[before_lines.len() - 1 - suffix]
                == after_lines[after_lines.len() - 1 - suffix]
        {
            suffix += 1;
        }

        let old_end = before_lines.len().saturating_sub(suffix);
        let new_end = after_lines.len().saturating_sub(suffix);
        let removed = &before_lines[prefix..old_end];
        let added = &after_lines[prefix..new_end];
        let old_start = prefix + 1;
        let new_start = prefix + 1;

        let mut out = format!(
            "@@ -{},{} +{},{} @@",
            old_start,
            removed.len(),
            new_start,
            added.len()
        );

        for line in removed {
            out.push('\n');
            out.push('-');
            out.push_str(line);
        }
        for line in added {
            out.push('\n');
            out.push('+');
            out.push_str(line);
        }

        out
    }

    async fn read_notebook_raw(&self) -> String {
        let notebook = self.notebook.lock().await;
        serde_json::to_string_pretty(&*notebook).unwrap_or_default()
    }

    async fn build_recent_history_excerpt(&self, count: usize) -> String {
        let Some(turns) = self.load_history_turns().await else {
            return "(history unavailable)".to_string();
        };

        if turns.is_empty() {
            return "(no history yet)".to_string();
        }

        let start = turns.len().saturating_sub(count.max(1));
        turns[start..]
            .iter()
            .enumerate()
            .map(|(offset, turn)| {
                format!(
                    "#{} [{} @ {}] {}",
                    start + offset,
                    turn.role,
                    turn.timestamp.to_rfc3339(),
                    truncate_text(turn.content.as_str(), 320, "")
                )
            })
            .collect::<Vec<_>>()
            .join("\n---\n")
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

    async fn append_history_turn(&self, role: &str, content: &str) {
        let Some(path) = self.history_archive_path.as_ref() else {
            return;
        };

        if content.trim().is_empty() {
            return;
        }

        if let Some(parent) = path.parent()
            && let Err(err) = tokio::fs::create_dir_all(parent).await
        {
            warn!(
                path = %path.display(),
                error = %err,
                "failed to create supervisor history parent directory"
            );
            return;
        }

        let turn = HistoryTurn {
            timestamp: Utc::now(),
            role: role.to_string(),
            content: content.to_string(),
        };

        let line = match serde_json::to_string(&turn) {
            Ok(line) => line,
            Err(err) => {
                warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to serialize supervisor history turn"
                );
                return;
            }
        };

        let mut payload = line;
        payload.push('\n');
        let mut file = match tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
        {
            Ok(file) => file,
            Err(err) => {
                warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to open supervisor history archive"
                );
                return;
            }
        };

        use tokio::io::AsyncWriteExt;
        if let Err(err) = file.write_all(payload.as_bytes()).await {
            warn!(
                path = %path.display(),
                error = %err,
                "failed to append supervisor history"
            );
        }
    }

    async fn load_history_turns(&self) -> Option<Vec<HistoryTurn>> {
        let path = self.history_archive_path.as_ref()?;
        let raw = match tokio::fs::read_to_string(path).await {
            Ok(raw) => raw,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    return Some(Vec::new());
                }
                warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to read supervisor history archive"
                );
                return None;
            }
        };

        let mut turns = Vec::new();
        for line in raw.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<HistoryTurn>(line) {
                Ok(turn) => turns.push(turn),
                Err(err) => {
                    warn!(
                        path = %path.display(),
                        error = %err,
                        "failed to parse supervisor history line; skipping"
                    );
                }
            }
        }
        Some(turns)
    }
}

fn supervisor_tools_schema() -> Vec<ToolSpec> {
    fn function_tool(
        name: &str,
        description: &str,
        properties: BTreeMap<String, JsonSchema>,
        required: &[&str],
    ) -> ToolSpec {
        ToolSpec::Function(ResponsesApiTool {
            name: name.to_string(),
            description: description.to_string(),
            strict: false,
            parameters: JsonSchema::Object {
                properties,
                required: (!required.is_empty())
                    .then(|| required.iter().map(ToString::to_string).collect()),
                additional_properties: Some(AdditionalProperties::Boolean(false)),
            },
        })
    }

    fn string_property(description: &str) -> JsonSchema {
        JsonSchema::String {
            description: Some(description.to_string()),
        }
    }

    fn integer_property(description: &str) -> JsonSchema {
        JsonSchema::Number {
            description: Some(description.to_string()),
        }
    }

    vec![
        function_tool(
            "read_notebook",
            "Read full notebook file content before analysis or edits.",
            BTreeMap::new(),
            &[],
        ),
        function_tool(
            "apply_patch_notebook",
            "Apply an apply_patch-style patch to the notebook file. Update business fields only; system timestamps are auto-maintained.",
            BTreeMap::from([(
                "patch".to_string(),
                string_property(
                    "Patch text using *** Begin Patch / *** End Patch format. For list entries, write business fields only.",
                ),
            )]),
            &["patch"],
        ),
        function_tool(
            "search_history",
            "Search archived conversation history by keyword.",
            BTreeMap::from([(
                "query".to_string(),
                string_property("Keyword query for archived turns."),
            )]),
            &["query"],
        ),
        function_tool(
            "read_recent",
            "Read recent conversation turns.",
            BTreeMap::from([(
                "count".to_string(),
                integer_property("Number of recent turns to read."),
            )]),
            &[],
        ),
        function_tool(
            "read_turn",
            "Read one conversation turn by index.",
            BTreeMap::from([(
                "index".to_string(),
                integer_property("0-based turn index in archive."),
            )]),
            &["index"],
        ),
        function_tool(
            "history_stats",
            "Get history and token stats.",
            BTreeMap::new(),
            &[],
        ),
        function_tool(
            "read_file",
            "Read file content by path and optional line window.",
            BTreeMap::from([
                (
                    "path".to_string(),
                    string_property("File path relative to workspace cwd or absolute path."),
                ),
                (
                    "offset".to_string(),
                    integer_property("1-based line offset."),
                ),
                ("limit".to_string(), integer_property("Max lines to read.")),
            ]),
            &["path"],
        ),
        function_tool(
            "glob",
            "List files matching a glob pattern.",
            BTreeMap::from([(
                "pattern".to_string(),
                string_property("Glob pattern like *.rs or src/**/*.ts."),
            )]),
            &["pattern"],
        ),
        function_tool(
            "shell",
            "Execute a read-only shell command in safety whitelist.",
            BTreeMap::from([("cmd".to_string(), string_property("Shell command to run."))]),
            &["cmd"],
        ),
        function_tool(
            "rg",
            "Search files via ripgrep pattern.",
            BTreeMap::from([(
                "pattern".to_string(),
                string_property("Ripgrep search pattern."),
            )]),
            &["pattern"],
        ),
        function_tool(
            "ls",
            "List directory entries.",
            BTreeMap::from([(
                "path".to_string(),
                string_property("Directory path to list."),
            )]),
            &["path"],
        ),
    ]
}

fn build_supervisor_prompt(
    input_messages: &[String],
    last_assistant_message: &str,
    notebook_text: &str,
    history_excerpt: &str,
) -> String {
    let latest_user_message = input_messages
        .last()
        .map(String::as_str)
        .unwrap_or("(no user message captured)");

    format!(
        r#"You are GugaCodex, the supervision agent for Codex. You have your own notebook and long-term memory.

=== Your Notebook File (Persistent) ===
{notebook_text}

=== Persistent Memory (Recent History) ===
{history_excerpt}

=== Current Turn ===
User:
{latest_user_message}

Codex Output:
{last_assistant_message}

Default stance:
- Assume Codex is doing fine unless there is clear evidence of a violation.
- Most turns should return "ok" (Codex completes tasks, explains results, writes code).
- If confidence is low, prefer "ok".

Your duties (in priority order):
1. Judge if behavior is reasonable **given the user's specific instructions and preferences**
2. If you see a clear violation with high confidence, provide correction
3. Before deciding, read notebook content with `read_notebook`.
4. Keep notebook updates high-signal: write only durable information that
   improves future decisions (new progress, a new risk/attention item, or a
   correction lesson), and avoid near-duplicate entries unless something
   materially changed.
5. Minimal action principle: if current turn content is already sufficient, do not call tools.

Available tools (structured function calls, use when needed):

Notebook:
- read_notebook
- apply_patch_notebook
  (When updating notebook entries, write business fields only. System fields
   like timestamp/added_at/last_updated are auto-maintained.)

History:
- search_history
- read_recent
- read_turn
- history_stats

File verification (read-only):
- read_file
- glob
- shell
- rg
- ls

=== Normal behavior (do not flag) ===
- Codex completing a task and summarizing what it did ("Done! I created X with features Y and Z")
- Codex writing code with reasonable features (error handling, input validation, comments)
- Codex explaining how to use something it just built
- Codex listing files, reading context, then acting — this is good practice
- Codex responding with a plan or explanation when the user asked a question
- Adding standard best practices (e.g. error handling for a calculator) — this is not over-engineering

=== Violations (flag only when clearly present) ===
- FALLBACK: Codex refuses the task ("can't do it", "let's simplify", "skip for now") instead of trying to complete it.
- IGNORED_INSTRUCTION: Codex does the opposite of an explicit user instruction (for example, user asked for Python but Codex used JavaScript).
- UNNECESSARY_INTERACTION: Codex pauses mid-task to ask permission or narrate, and the user explicitly asked for autonomous execution ("just do it", "don't ask", "work autonomously", "finish before talking to me"). Both conditions must hold. If the task is already complete, summarizing results is normal. If the user gave no such instruction, narration is normal.
- OVER_ENGINEERING: Codex adds architectural complexity the user did not ask for (for example, introducing a full caching layer, adding redundant fallback systems, or refactoring an entire module for a narrow fix). Standard robustness work (error handling, input validation, clean structure) is not over-engineering.
- UNAUTHORIZED_CHANGE: Codex changes unrelated behavior not requested by user.

Decision threshold: high confidence only.
- If Codex completed what the user asked, even with extra explanation or features, that is OK.
- Avoid nitpicking. Summarizing completed work is normal behavior, not unnecessary interaction.
- For straightforward requests (confirmations, short Q&A, obvious edits), default to no tool calls.

Final response format:
- If tool calls are used, continue until tool outputs are incorporated.
- Return exactly one final JSON object (no extra text).

If no violation (this should be your answer ~90% of the time):
{{"result": "ok", "summary": "What Codex did, one sentence"}}

If violation found (only when you are highly confident):
{{"result": "violation", "type": "VIOLATION_TYPE", "description": "What went wrong specifically", "correction": "Specific instruction to fix it"}}

Valid violation types: FALLBACK, IGNORED_INSTRUCTION, UNAUTHORIZED_CHANGE, UNNECESSARY_INTERACTION, OVER_ENGINEERING

Final answer must be JSON only, with no extra text before or after."#
    )
}

fn build_supervisor_chat_prompt(
    user_message: &str,
    notebook_text: &str,
    history_excerpt: &str,
) -> String {
    format!(
        r#"You are GugaCodex, an AI supervision agent that monitors another AI (Codex).
You have full access to the conversation history and your personal notebook.

=== Your Notebook File (Persistent) ===
{notebook_text}

=== Persistent Memory (Recent History) ===
{history_excerpt}

The user is speaking to you directly. Answer helpfully, concisely, and in the
same language the user used. You can:
- Explain your past supervision decisions
- Discuss the current task and Codex's behavior
- Share observations from your notebook
- Answer questions about the codebase (based on what you've seen)
- Use structured tools when needed
- Be explicit when you are uncertain

Available tools (structured function calls):
- read_notebook
- apply_patch_notebook
  (Write business fields only; timestamp/added_at/last_updated are auto-maintained.)
- search_history
- read_recent
- read_turn
- history_stats
- read_file
- glob
- shell
- rg
- ls

User message:
{user_message}"#
    )
}

fn extract_structured_tool_calls(items: &[ResponseItem]) -> Vec<StructuredToolCall> {
    items
        .iter()
        .filter_map(|item| match item {
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => Some(StructuredToolCall {
                call_id: call_id.clone(),
                tool_name: name.clone(),
                arguments: arguments.clone(),
                item: item.clone(),
            }),
            _ => None,
        })
        .collect()
}

fn extract_response_item_text(item: &ResponseItem) -> Option<String> {
    let ResponseItem::Message { content, .. } = item else {
        return None;
    };

    let mut pieces = Vec::new();
    for part in content {
        match part {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                if !text.trim().is_empty() {
                    pieces.push(text.as_str());
                }
            }
            ContentItem::InputImage { .. } => {}
        }
    }

    if pieces.is_empty() {
        None
    } else {
        Some(pieces.join("\n"))
    }
}

fn parse_supervisor_decision(response: &str) -> SupervisorDecision {
    #[derive(Debug, Deserialize)]
    struct ParsedDecision {
        result: Option<String>,
        summary: Option<String>,
        #[serde(rename = "type")]
        violation_type: Option<String>,
        description: Option<String>,
        correction: Option<String>,
    }

    let candidate = extract_first_json_object(response).unwrap_or_else(|| response.to_string());
    if let Ok(parsed) = serde_json::from_str::<ParsedDecision>(&candidate) {
        let result = parsed.result.unwrap_or_else(|| "ok".to_string());
        let summary = parsed
            .summary
            .or_else(|| parsed.description.clone())
            .unwrap_or_else(|| {
                truncate_text(
                    response.trim(),
                    220,
                    "Supervisor returned no summary; defaulting to conservative OK.",
                )
            });
        return SupervisorDecision {
            result,
            summary,
            violation_type: parsed.violation_type,
            description: parsed.description,
            correction: parsed.correction,
        };
    }

    SupervisorDecision::ok(truncate_text(
        response.trim(),
        220,
        "Supervisor returned no parseable JSON; defaulting to conservative OK.",
    ))
}

fn extract_first_json_object(text: &str) -> Option<String> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    for (start_idx, start_char) in &chars {
        if *start_char != '{' {
            continue;
        }

        let mut depth = 0usize;
        let mut in_string = false;
        let mut escape = false;

        for (idx, ch) in chars.iter().copied().skip_while(|(idx, _)| idx < start_idx) {
            if in_string {
                if escape {
                    escape = false;
                    continue;
                }
                match ch {
                    '\\' => escape = true,
                    '"' => in_string = false,
                    _ => {}
                }
                continue;
            }

            match ch {
                '"' => in_string = true,
                '{' => depth += 1,
                '}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return Some(text[*start_idx..idx + ch.len_utf8()].to_string());
                    }
                }
                _ => {}
            }
        }
    }
    None
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

fn resolve_path(cwd: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        cwd.join(candidate)
    }
}

fn glob_files_blocking(cwd: &Path, pattern: &str) -> Result<Vec<PathBuf>, String> {
    if pattern.trim().is_empty() {
        return Err("empty pattern".to_string());
    }

    let matcher = WildMatch::new(pattern);
    let pattern_no_slash = !pattern.contains('/');
    let mut stack = vec![cwd.to_path_buf()];
    let mut matches = Vec::new();
    let mut visited = 0usize;

    while let Some(dir) = stack.pop() {
        let read_dir = std::fs::read_dir(&dir).map_err(|e| format!("read_dir failed: {e}"))?;
        for entry in read_dir {
            let entry = entry.map_err(|e| format!("read_dir entry failed: {e}"))?;
            let path = entry.path();
            visited += 1;
            if visited > MAX_GLOB_VISITS {
                return Err(format!(
                    "glob scan exceeded {MAX_GLOB_VISITS} filesystem entries"
                ));
            }

            let rel = path.strip_prefix(cwd).unwrap_or(path.as_path());
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            let abs_str = path.to_string_lossy().replace('\\', "/");

            let filename_match = pattern_no_slash
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| matcher.matches(name));

            if matcher.matches(rel_str.as_str())
                || matcher.matches(abs_str.as_str())
                || filename_match
            {
                matches.push(path.clone());
            }

            if path.is_dir() {
                stack.push(path);
            }
        }
    }

    matches.sort();
    Ok(matches)
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

fn format_turn_ack_message(review_summary: &str) -> String {
    let summary = truncate_text(
        review_summary.trim(),
        320,
        "Supervisor finished reviewing this turn.",
    );
    if summary.starts_with("🛡️") {
        summary
    } else {
        format!("🛡️ {summary}")
    }
}

fn normalize_supervisor_chat_reply(raw_response: &str) -> String {
    const FALLBACK: &str = "I read your message but do not have a clear answer yet.";
    let raw = raw_response.trim();
    if raw.is_empty() {
        return FALLBACK.to_string();
    }

    if let Some(json_reply) = extract_supervisor_chat_reply_from_json(raw) {
        return truncate_text(json_reply.trim(), 4_000, FALLBACK);
    }

    truncate_text(raw, 4_000, FALLBACK)
}

fn extract_supervisor_chat_reply_from_json(raw_response: &str) -> Option<String> {
    let candidate =
        extract_first_json_object(raw_response).unwrap_or_else(|| raw_response.to_string());
    let value: serde_json::Value = serde_json::from_str(&candidate).ok()?;

    if let Some(text) = value.as_str() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    let keys = ["reply", "message", "content", "response", "text", "summary"];
    for key in keys {
        let Some(text) = value.get(key).and_then(serde_json::Value::as_str) else {
            continue;
        };
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    None
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

fn tool_args_preview(args: &str) -> String {
    let compact = args.trim().replace('\n', "\\n");
    truncate_text(compact.as_str(), 180, "")
}

fn supervisor_tool_display_prefix(tool_name: &str) -> Vec<String> {
    match tool_name {
        "read_notebook" => vec!["guga-codex/notebook".to_string(), "read".to_string()],
        "apply_patch_notebook" => vec!["guga-codex/notebook".to_string(), "patch".to_string()],
        "search_history" => vec!["guga-codex/history".to_string(), "search".to_string()],
        "read_recent" => vec!["guga-codex/history".to_string(), "read_recent".to_string()],
        "read_turn" => vec!["guga-codex/history".to_string(), "read_turn".to_string()],
        "history_stats" => vec!["guga-codex/history".to_string(), "stats".to_string()],
        "read_file" => vec!["guga-codex/fs".to_string(), "read".to_string()],
        "glob" => vec!["guga-codex/fs".to_string(), "glob".to_string()],
        "ls" => vec!["guga-codex/fs".to_string(), "list".to_string()],
        "rg" => vec!["guga-codex/fs".to_string(), "search".to_string()],
        "shell" => vec!["guga-codex/fs".to_string(), "shell".to_string()],
        _ => vec!["guga-codex/tool".to_string(), tool_name.to_string()],
    }
}

fn supervisor_tool_parsed_command(
    tool_name: &str,
    args: &str,
    cwd: &Path,
    notebook_path: Option<&PathBuf>,
    success: bool,
    command: &[String],
) -> Vec<ParsedCommand> {
    let unknown = || {
        vec![ParsedCommand::Unknown {
            cmd: command.join(" "),
        }]
    };

    if !success {
        return unknown();
    }

    match tool_name {
        "read_notebook" => {
            let path = notebook_path
                .cloned()
                .unwrap_or_else(|| PathBuf::from("guga-codex/notebook.json"));
            vec![ParsedCommand::Read {
                cmd: "read_notebook".to_string(),
                name: "notebook".to_string(),
                path,
            }]
        }
        "read_file" => {
            let path_text = args.split('|').next().unwrap_or_default().trim();
            if path_text.is_empty() {
                return unknown();
            }
            let path_buf = PathBuf::from(path_text);
            let path = if path_buf.is_absolute() {
                path_buf
            } else {
                cwd.join(path_buf)
            };
            vec![ParsedCommand::Read {
                cmd: "read_file".to_string(),
                name: path_text.to_string(),
                path,
            }]
        }
        "search_history" => vec![ParsedCommand::Search {
            cmd: "search_history".to_string(),
            query: if args.trim().is_empty() {
                None
            } else {
                Some(args.to_string())
            },
            path: Some("history".to_string()),
        }],
        "rg" => vec![ParsedCommand::Search {
            cmd: "rg".to_string(),
            query: if args.trim().is_empty() {
                None
            } else {
                Some(args.to_string())
            },
            path: None,
        }],
        "glob" | "ls" => vec![ParsedCommand::ListFiles {
            cmd: tool_name.to_string(),
            path: if args.trim().is_empty() {
                None
            } else {
                Some(args.to_string())
            },
        }],
        _ => unknown(),
    }
}

fn notebook_patch_status_from_output(output: &str) -> Option<(String, bool, PatchApplyStatus)> {
    let summary = output.lines().next()?.trim();
    if !summary.starts_with("apply_patch_notebook:") {
        return None;
    }

    let detail = summary.trim_start_matches("apply_patch_notebook:").trim();
    if detail.starts_with("Applied") || detail == "No changes applied" {
        Some((summary.to_string(), true, PatchApplyStatus::Completed))
    } else {
        Some((summary.to_string(), false, PatchApplyStatus::Failed))
    }
}

fn supervisor_enabled_from_env() -> bool {
    match std::env::var(SUPERVISOR_ENV_VAR) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => true,
    }
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

        let warning = runtime
            .after_agent("turn-1", &[String::from("hello")], Some("world"))
            .await;
        assert_eq!(warning, None);

        let notebook_path = dir
            .path()
            .join("guga-codex")
            .join("notebooks")
            .join(format!("{conversation_id}.json"));
        assert!(notebook_path.exists());

        let stored = tokio::fs::read_to_string(notebook_path)
            .await
            .expect("read notebook");
        assert!(stored.contains("Reviewed guga-codex turn turn-1"));
    }

    #[tokio::test]
    async fn enabled_runtime_generates_turn_ack_message() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = SupervisorRuntime::new(dir.path(), ThreadId::default(), true).await;

        let outcome = runtime
            .after_agent_with_turn(
                "turn-ack",
                &[String::from("hello")],
                Some("world"),
                None,
                None,
            )
            .await
            .expect("enabled runtime should return outcome");

        assert_eq!(outcome.warning_message, None);
        assert_eq!(outcome.user_ack_message, Some("🛡️ world".to_string()));
        assert_eq!(outcome.correction_user_message, None);
        assert_eq!(outcome.correction_to_codex, None);
    }

    #[test]
    fn violation_decision_extracts_correction_for_codex() {
        let decision = SupervisorDecision {
            result: "violation".to_string(),
            summary: "ignored instruction".to_string(),
            violation_type: Some("IGNORED_INSTRUCTION".to_string()),
            description: Some("Codex ignored user instruction".to_string()),
            correction: Some("  Follow the explicit user instruction first.  ".to_string()),
        };

        assert_eq!(
            decision.correction_for_codex(),
            Some("Follow the explicit user instruction first.".to_string())
        );
        assert_eq!(
            decision.correction_user_message(),
            Some("🛡️ Corrected: Follow the explicit user instruction first.".to_string())
        );
    }

    #[test]
    fn ok_decision_does_not_emit_correction() {
        let decision = SupervisorDecision::ok("looks good".to_string());
        assert_eq!(decision.correction_for_codex(), None);
        assert_eq!(decision.correction_user_message(), None);
    }

    #[test]
    fn normalize_supervisor_chat_reply_unwraps_reply_json() {
        let normalized = normalize_supervisor_chat_reply(
            r#"{"reply":"你好，我是 GugaCodex（监督代理）。目前状态正常。"}"#,
        );
        assert_eq!(
            normalized,
            "你好，我是 GugaCodex（监督代理）。目前状态正常。"
        );
    }

    #[test]
    fn normalize_supervisor_chat_reply_accepts_plain_text() {
        let normalized = normalize_supervisor_chat_reply("你好，我在这里。");
        assert_eq!(normalized, "你好，我在这里。");
    }

    #[tokio::test]
    async fn deduplicates_equivalent_after_agent_updates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = SupervisorRuntime::new(dir.path(), ThreadId::default(), true).await;

        let first = runtime
            .after_agent("turn-1", &[String::from("hello")], Some("world"))
            .await;
        let completed_after_first = {
            let notebook = runtime.notebook.lock().await;
            notebook.completed.len()
        };
        let second = runtime
            .after_agent("turn-1", &[String::from("hello")], Some("world"))
            .await;
        let completed_after_second = {
            let notebook = runtime.notebook.lock().await;
            notebook.completed.len()
        };

        assert_eq!(first, None);
        assert_eq!(second, None);
        assert_eq!(completed_after_first, 1);
        assert_eq!(completed_after_second, 1);
    }

    #[test]
    fn truncate_text_returns_fallback_for_empty_input() {
        assert_eq!(truncate_text("", 20, "fallback"), "fallback");
    }

    #[test]
    fn normalize_tool_arguments_maps_apply_patch_notebook() {
        let normalized = SupervisorRuntime::normalize_tool_arguments(
            "apply_patch_notebook",
            r#"{"patch":"*** Begin Patch\n*** Update File: notebook\n@@\n-a\n+b\n*** End Patch"}"#,
        )
        .expect("should normalize");
        assert!(normalized.contains("*** Begin Patch"));
    }

    #[test]
    fn normalize_tool_arguments_maps_read_file_window() {
        let normalized = SupervisorRuntime::normalize_tool_arguments(
            "read_file",
            r#"{"path":"src/main.rs","offset":5,"limit":10}"#,
        )
        .expect("should normalize");
        assert_eq!(normalized, "src/main.rs|5|10");
    }

    #[test]
    fn normalize_tool_arguments_maps_read_notebook_without_args() {
        let normalized =
            SupervisorRuntime::normalize_tool_arguments("read_notebook", "{}").expect("normalize");
        assert!(normalized.is_empty());
    }

    #[test]
    fn supervisor_tool_display_prefix_uses_notebook_friendly_name() {
        assert_eq!(
            supervisor_tool_display_prefix("apply_patch_notebook"),
            vec!["guga-codex/notebook".to_string(), "patch".to_string()]
        );
    }

    #[test]
    fn supervisor_tool_parsed_command_maps_read_notebook_to_read() {
        let path = PathBuf::from("/tmp/notebook.json");
        let parsed = supervisor_tool_parsed_command(
            "read_notebook",
            "",
            Path::new("/tmp"),
            Some(&path),
            true,
            &["guga-codex/notebook".to_string(), "read".to_string()],
        );
        assert_eq!(
            parsed,
            vec![ParsedCommand::Read {
                cmd: "read_notebook".to_string(),
                name: "notebook".to_string(),
                path,
            }]
        );
    }

    #[test]
    fn notebook_patch_status_from_output_maps_no_changes_to_success() {
        let parsed = notebook_patch_status_from_output("apply_patch_notebook: No changes applied")
            .expect("expected status");
        assert_eq!(parsed.0, "apply_patch_notebook: No changes applied");
        assert!(parsed.1);
        assert_eq!(parsed.2, PatchApplyStatus::Completed);
    }

    #[test]
    fn notebook_patch_status_from_output_maps_validation_to_failure() {
        let parsed = notebook_patch_status_from_output(
            "apply_patch_notebook: Validation failed\n- $.completed[0]: invalid item type",
        )
        .expect("expected status");
        assert_eq!(parsed.0, "apply_patch_notebook: Validation failed");
        assert!(!parsed.1);
        assert_eq!(parsed.2, PatchApplyStatus::Failed);
    }

    #[test]
    fn normalize_tool_arguments_maps_read_recent_default_count() {
        let normalized =
            SupervisorRuntime::normalize_tool_arguments("read_recent", "{}").expect("normalize");
        assert_eq!(normalized, "5");
    }

    #[test]
    fn normalize_tool_arguments_accepts_search_history_aliases() {
        let normalized =
            SupervisorRuntime::normalize_tool_arguments("search_history", r#"{"keyword":"oauth"}"#)
                .expect("normalize");
        assert_eq!(normalized, "oauth");
    }

    #[test]
    fn normalize_tool_arguments_accepts_shell_command_alias() {
        let normalized =
            SupervisorRuntime::normalize_tool_arguments("shell", r#"{"command":"pwd"}"#)
                .expect("normalize");
        assert_eq!(normalized, "pwd");
    }

    #[test]
    fn normalize_tool_arguments_rejects_invalid_json() {
        let err = SupervisorRuntime::normalize_tool_arguments("shell", "not-json")
            .expect_err("invalid arguments should fail");
        assert!(err.contains("invalid JSON"));
    }

    #[test]
    fn parse_apply_patch_hunks_and_apply() {
        let patch = "*** Begin Patch\n*** Update File: notebook\n@@\n-  \"current_activity\": \"A\"\n+  \"current_activity\": \"B\"\n*** End Patch";
        let hunks = SupervisorRuntime::parse_apply_patch_hunks(patch).expect("parse patch");
        let original = "{\n  \"current_activity\": \"A\"\n}\n";
        let (updated, _) = SupervisorRuntime::apply_patch_hunks(original, &hunks).expect("apply");
        assert!(updated.contains("\"B\""));
    }

    #[test]
    fn normalize_notebook_value_accepts_business_fields_and_auto_populates_system_fields() {
        let now = DateTime::parse_from_rfc3339("2026-02-22T13:31:30Z")
            .expect("parse timestamp")
            .with_timezone(&Utc);
        let value = serde_json::json!({
            "current_activity": "测试 notebook 写入",
            "completed": [
                {
                    "what": "实现 apply_patch_notebook 新行为",
                    "significance": "用户不需要手写 timestamp"
                }
            ],
            "attention": [
                {
                    "content": "确认 UI 继续显示文件变更块"
                }
            ],
            "mistakes": [
                {
                    "what_happened": "把系统字段交给模型手写",
                    "how_corrected": "后端统一自动维护",
                    "lesson": "让模型只写业务字段"
                }
            ],
            "last_updated": "2000-01-01T00:00:00Z"
        });

        let notebook = SupervisorRuntime::normalize_notebook_value(&value, now.clone())
            .expect("normalize notebook");
        assert_eq!(
            notebook.current_activity,
            Some("测试 notebook 写入".to_string())
        );
        assert_eq!(notebook.completed.len(), 1);
        assert_eq!(notebook.completed[0].timestamp.clone(), now);
        assert_eq!(notebook.attention.len(), 1);
        assert_eq!(format!("{}", notebook.attention[0].source), "inference");
        assert_eq!(format!("{}", notebook.attention[0].priority), "medium");
        assert_eq!(notebook.attention[0].added_at.clone(), now);
        assert_eq!(notebook.mistakes.len(), 1);
        assert_eq!(notebook.mistakes[0].timestamp.clone(), now);
        assert_eq!(notebook.last_updated, Some(now));
    }

    #[test]
    fn normalize_notebook_value_rejects_invalid_completed_item_shape() {
        let now = DateTime::parse_from_rfc3339("2026-02-22T13:31:30Z")
            .expect("parse timestamp")
            .with_timezone(&Utc);
        let value = serde_json::json!({
            "completed": ["just-a-string"]
        });
        let report = SupervisorRuntime::normalize_notebook_value(&value, now)
            .expect_err("invalid completed shape should fail");
        let message = report.into_message();
        assert!(message.contains("apply_patch_notebook: Validation failed"));
        assert!(message.contains("$.completed[0]: invalid item type"));
    }

    #[test]
    fn normalize_notebook_value_rejects_invalid_attention_priority() {
        let now = DateTime::parse_from_rfc3339("2026-02-22T13:31:30Z")
            .expect("parse timestamp")
            .with_timezone(&Utc);
        let value = serde_json::json!({
            "attention": [
                {
                    "content": "needs triage",
                    "priority": "urgent"
                }
            ]
        });
        let report = SupervisorRuntime::normalize_notebook_value(&value, now)
            .expect_err("invalid priority should fail");
        let message = report.into_message();
        assert!(message.contains("$.attention[0].priority"));
        assert!(message.contains("high | medium | low"));
    }

    #[tokio::test]
    async fn apply_patch_notebook_returns_structured_validation_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = SupervisorRuntime::new(dir.path(), ThreadId::default(), true).await;
        let patch = "*** Begin Patch\n*** Update File: notebook\n@@\n-  \"completed\": [],\n+  \"completed\": [\n+    \"bad\"\n+  ],\n*** End Patch";

        let output = runtime.handle_apply_patch_notebook(patch).await;
        assert!(output.contains("apply_patch_notebook: Validation failed"));
        assert!(output.contains("$.completed[0]"));
    }

    #[tokio::test]
    async fn build_notebook_patch_render_payload_extracts_change_and_diff() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = SupervisorRuntime::new(dir.path(), ThreadId::default(), true).await;
        let output = "apply_patch_notebook: Applied 1 hunk(s)\n@@ -1,1 +1,1 @@\n-a\n+b";

        let payload = runtime
            .build_notebook_patch_render_payload(output)
            .expect("payload");
        assert_eq!(payload.summary, "apply_patch_notebook: Applied 1 hunk(s)");
        assert!(payload.turn_diff.contains("diff --git a/"));
        assert!(payload.turn_diff.contains("@@ -1,1 +1,1 @@"));
        assert_eq!(payload.changes.len(), 1);
        let change = payload.changes.values().next().expect("single change");
        match change {
            FileChange::Update {
                unified_diff,
                move_path,
            } => {
                assert_eq!(move_path, &None);
                assert!(unified_diff.contains("@@ -1,1 +1,1 @@"));
            }
            _ => panic!("expected update change"),
        }
    }

    #[tokio::test]
    async fn build_notebook_patch_render_payload_ignores_non_patch_output() {
        let dir = tempfile::tempdir().expect("tempdir");
        let runtime = SupervisorRuntime::new(dir.path(), ThreadId::default(), true).await;
        assert!(
            runtime
                .build_notebook_patch_render_payload("apply_patch_notebook: No changes applied")
                .is_none()
        );
    }

    #[test]
    fn extract_first_json_object_handles_prefixed_text() {
        let text = "Some prefix {\"result\":\"ok\",\"summary\":\"done\"} trailing";
        let parsed = extract_first_json_object(text).expect("json object");
        assert_eq!(parsed, "{\"result\":\"ok\",\"summary\":\"done\"}");
    }

    #[test]
    fn supervisor_prompt_keeps_notebook_memory_turn_order() {
        let prompt = build_supervisor_prompt(
            &[String::from("User asks for strict alignment")],
            "Codex completed the requested changes.",
            "{\"current_activity\":\"checking\"}",
            "#0 [user] prior note",
        );

        let notebook_idx = prompt
            .find("=== Your Notebook File (Persistent) ===")
            .expect("notebook section present");
        let memory_idx = prompt
            .find("=== Persistent Memory (Recent History) ===")
            .expect("memory section present");
        let turn_idx = prompt
            .find("=== Current Turn ===")
            .expect("current turn section present");

        assert!(notebook_idx < memory_idx);
        assert!(memory_idx < turn_idx);
        assert!(prompt.contains("User:\nUser asks for strict alignment"));
        assert!(prompt.contains("Codex Output:\nCodex completed the requested changes."));
    }

    #[test]
    fn supervisor_prompt_preserves_old_instruction_style() {
        let prompt = build_supervisor_prompt(&[String::from("x")], "y", "{}", "(no history yet)");

        assert!(prompt.contains(
            "Judge if behavior is reasonable **given the user's specific instructions and preferences**"
        ));
        assert!(prompt.contains("Before deciding, read notebook content with `read_notebook`."));
        assert!(prompt.contains("If no violation (this should be your answer ~90% of the time):"));
        assert!(prompt.contains(
            "Valid violation types: FALLBACK, IGNORED_INSTRUCTION, UNAUTHORIZED_CHANGE, UNNECESSARY_INTERACTION, OVER_ENGINEERING"
        ));
    }

    #[test]
    fn supervisor_chat_prompt_keeps_notebook_memory_user_order() {
        let prompt = build_supervisor_chat_prompt(
            "为什么你要这样判断？",
            "{\"current_activity\":\"reviewing\"}",
            "#0 [codex] prior",
        );
        let notebook_idx = prompt
            .find("=== Your Notebook File (Persistent) ===")
            .expect("notebook section present");
        let memory_idx = prompt
            .find("=== Persistent Memory (Recent History) ===")
            .expect("memory section present");
        let user_idx = prompt.find("User message:").expect("user section present");

        assert!(notebook_idx < memory_idx);
        assert!(memory_idx < user_idx);
        assert!(prompt.contains("为什么你要这样判断？"));
    }

    #[test]
    fn supervisor_chat_prompt_mentions_structured_tools() {
        let prompt = build_supervisor_chat_prompt("status?", "n", "h");
        assert!(prompt.contains("Use structured tools when needed"));
        assert!(prompt.contains("- read_notebook"));
        assert!(prompt.contains("- apply_patch_notebook"));
    }
}
