use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CompletedItem {
    pub(crate) timestamp: DateTime<Utc>,
    pub(crate) what: String,
    pub(crate) significance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AttentionItem {
    pub(crate) content: String,
    pub(crate) priority: String,
    pub(crate) added_at: DateTime<Utc>,
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

    pub(crate) fn summary_line(&self) -> String {
        format!(
            "notebook {} completed / {} attention",
            self.completed.len(),
            self.attention.len()
        )
    }

    fn trim(&mut self) {
        const COMPLETED_LIMIT: usize = 20;
        const ATTENTION_LIMIT: usize = 30;
        if self.completed.len() > COMPLETED_LIMIT {
            let to_remove = self.completed.len() - COMPLETED_LIMIT;
            self.completed.drain(0..to_remove);
        }
        if self.attention.len() > ATTENTION_LIMIT {
            let to_remove = self.attention.len() - ATTENTION_LIMIT;
            self.attention.drain(0..to_remove);
        }
    }
}
