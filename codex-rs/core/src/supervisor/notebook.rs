use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Priority {
    High,
    #[default]
    Medium,
    Low,
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::High => write!(f, "high"),
            Self::Medium => write!(f, "medium"),
            Self::Low => write!(f, "low"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttentionSource {
    UserInstruction,
    Mistake,
    #[default]
    Inference,
}

impl std::fmt::Display for AttentionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UserInstruction => write!(f, "user"),
            Self::Mistake => write!(f, "mistake"),
            Self::Inference => write!(f, "inference"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CompletedItem {
    pub(crate) timestamp: DateTime<Utc>,
    pub(crate) what: String,
    pub(crate) significance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AttentionItem {
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) source: AttentionSource,
    #[serde(default)]
    pub(crate) priority: Priority,
    pub(crate) added_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MistakeEntry {
    pub(crate) timestamp: DateTime<Utc>,
    pub(crate) what_happened: String,
    pub(crate) how_corrected: String,
    pub(crate) lesson: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct SupervisorNotebook {
    #[serde(default)]
    pub(crate) current_activity: Option<String>,
    #[serde(default)]
    pub(crate) completed: Vec<CompletedItem>,
    #[serde(default)]
    pub(crate) attention: Vec<AttentionItem>,
    #[serde(default)]
    pub(crate) mistakes: Vec<MistakeEntry>,
    #[serde(default)]
    pub(crate) last_updated: Option<DateTime<Utc>>,
}

impl SupervisorNotebook {
    pub(crate) fn apply_after_agent_update(
        &mut self,
        what: String,
        significance: String,
        activity: String,
    ) {
        self.current_activity = Some(activity);
        self.completed.push(CompletedItem {
            timestamp: Utc::now(),
            what,
            significance,
        });
        self.trim();
        self.last_updated = Some(Utc::now());
    }

    pub(crate) fn apply_request_user_input_update(
        &mut self,
        turn_id: &str,
        answer_groups: usize,
        selected_answers: usize,
    ) {
        self.current_activity = Some(format!(
            "Captured request_user_input response for turn {turn_id}"
        ));
        self.completed.push(CompletedItem {
            timestamp: Utc::now(),
            what: format!("Captured user choice(s) for turn {turn_id}"),
            significance: format!(
                "Recorded {answer_groups} answer group(s), {selected_answers} selected option(s)."
            ),
        });
        self.trim();
        self.last_updated = Some(Utc::now());
    }

    #[allow(dead_code)]
    pub(crate) fn add_attention(
        &mut self,
        content: String,
        source: AttentionSource,
        priority: Priority,
    ) {
        if self.attention.iter().any(|item| item.content == content) {
            return;
        }

        self.attention.push(AttentionItem {
            content,
            source,
            priority,
            added_at: Utc::now(),
        });
        self.trim();
        self.last_updated = Some(Utc::now());
    }

    #[allow(dead_code)]
    pub(crate) fn remove_attention(&mut self, content: &str) -> bool {
        let initial_len = self.attention.len();
        self.attention.retain(|item| item.content != content);
        if self.attention.len() != initial_len {
            self.last_updated = Some(Utc::now());
            true
        } else {
            false
        }
    }

    #[allow(dead_code)]
    pub(crate) fn record_mistake(
        &mut self,
        what_happened: String,
        how_corrected: String,
        lesson: String,
    ) {
        self.mistakes.push(MistakeEntry {
            timestamp: Utc::now(),
            what_happened,
            how_corrected,
            lesson: lesson.clone(),
        });
        self.add_attention(
            format!("Avoid: {lesson}"),
            AttentionSource::Mistake,
            Priority::High,
        );
        self.trim();
        self.last_updated = Some(Utc::now());
    }

    fn trim(&mut self) {
        const COMPLETED_LIMIT: usize = 20;
        const ATTENTION_LIMIT: usize = 30;
        const MISTAKE_LIMIT: usize = 15;

        if self.completed.len() > COMPLETED_LIMIT {
            let to_remove = self.completed.len() - COMPLETED_LIMIT;
            self.completed.drain(0..to_remove);
        }
        if self.attention.len() > ATTENTION_LIMIT {
            let to_remove = self.attention.len() - ATTENTION_LIMIT;
            self.attention.drain(0..to_remove);
        }
        if self.mistakes.len() > MISTAKE_LIMIT {
            let to_remove = self.mistakes.len() - MISTAKE_LIMIT;
            self.mistakes.drain(0..to_remove);
        }
    }
}
