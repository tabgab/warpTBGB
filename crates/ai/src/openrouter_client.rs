//! OpenRouter API client.
//!
//! OpenRouter is an OpenAI-API-compatible aggregator at https://openrouter.ai/api/v1.
//! It serves models from many providers using a unified chat-completions API.
//!
//! Supports OpenAI-format tool calling for compatible models.

use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default OpenRouter base URL.
pub const DEFAULT_OPENROUTER_URL: &str = "https://openrouter.ai/api/v1";

/// Default request timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(180);

/// A chat message in the OpenAI-compatible format, including optional tool
/// call and tool result semantics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls emitted by the assistant (role = "assistant").
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// When role = "tool", this is the id of the tool call this message is
    /// responding to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional display name (used by the `tool` role to identify which tool).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            ..Default::default()
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            ..Default::default()
        }
    }
    pub fn assistant_text(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            ..Default::default()
        }
    }
    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: None,
            tool_calls,
            ..Default::default()
        }
    }
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_call_id: Some(tool_call_id.into()),
            name: Some(tool_name.into()),
            ..Default::default()
        }
    }
}

/// OpenAI-format tool call emitted by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "default_tool_type")]
    pub call_type: String,
    pub function: FunctionCall,
}

fn default_tool_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// Arguments as a JSON string (OpenAI's native format).
    pub arguments: String,
}

/// OpenAI-format tool definition sent in the request.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Chat completion request (OpenAI format).
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

/// Model entry from OpenRouter's /api/v1/models endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub pricing: Option<Pricing>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Pricing {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub completion: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListModelsResponse {
    pub data: Vec<ModelInfo>,
}

/// Non-streaming chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    #[serde(default)]
    pub total_tokens: Option<u32>,
}

/// A single streaming chunk. Can contain text content OR an incremental
/// slice of a tool call's arguments string.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    /// Incremental tool call slices. When present, these need to be
    /// accumulated by index until the stream finishes.
    #[serde(default)]
    pub tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub call_type: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum OpenRouterError {
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Server error: {0}")]
    ServerError(String),

    #[error("Parse error: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Missing API key")]
    MissingApiKey,
}

pub type OpenRouterResult<T> = std::result::Result<T, OpenRouterError>;

/// One streamed event surfaced to the caller.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Incremental text content from the assistant.
    TextDelta(String),
    /// A fully-assembled tool call (emitted once all its argument slices have arrived).
    ToolCallComplete(ToolCall),
}

/// OpenRouter client.
#[derive(Debug, Clone)]
pub struct OpenRouterClient {
    base_url: String,
    api_key: String,
    http_client: Client,
}

impl OpenRouterClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(DEFAULT_OPENROUTER_URL, api_key)
    }

    pub fn with_base_url(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        let http_client = Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            http_client,
        }
    }

    /// List available models. Does not require authentication (but we send it anyway).
    pub async fn list_models(&self) -> OpenRouterResult<Vec<ModelInfo>> {
        let response = self
            .http_client
            .get(format!("{}/models", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(OpenRouterError::ServerError(format!(
                "Server returned status: {status}"
            )));
        }

        let body = response.text().await?;
        let result: ListModelsResponse = serde_json::from_str(&body)?;
        Ok(result.data)
    }

    /// Streaming chat completion; yields text deltas and complete tool calls.
    pub async fn chat_streaming(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDef>,
    ) -> OpenRouterResult<impl futures::Stream<Item = OpenRouterResult<StreamEvent>> + Send + 'static>
    {
        let request = ChatRequest {
            model: model.to_string(),
            messages,
            stream: Some(true),
            temperature: None,
            max_tokens: None,
            tools,
            tool_choice: None,
        };

        let response = self
            .http_client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .header("HTTP-Referer", "https://warp.dev")
            .header("X-Title", "Warp")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(OpenRouterError::ServerError(format!(
                "Server returned status {status}: {body}"
            )));
        }

        let stream = async_stream::stream! {
            let mut bytes_stream = response.bytes_stream();
            let mut buffer = String::new();
            // Accumulators for in-flight tool calls, keyed by OpenAI's stream index.
            let mut tool_accum: std::collections::BTreeMap<u32, ToolCallAccum> =
                std::collections::BTreeMap::new();
            let mut saw_finish = false;

            while let Some(chunk) = bytes_stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        while let Some(newline_idx) = buffer.find('\n') {
                            let line = buffer[..newline_idx].trim().to_string();
                            buffer.drain(..=newline_idx);
                            let Some(data) = line.strip_prefix("data: ") else {
                                continue;
                            };
                            if data.trim() == "[DONE]" {
                                // Flush any remaining tool calls.
                                for (idx, acc) in std::mem::take(&mut tool_accum) {
                                    log::debug!(
                                        "openrouter stream: finalizing tool call index={} name={:?} args={}",
                                        idx,
                                        acc.name,
                                        acc.arguments,
                                    );
                                    if let Some(tc) = acc.into_tool_call() {
                                        yield Ok(StreamEvent::ToolCallComplete(tc));
                                    }
                                }
                                return;
                            }
                            match serde_json::from_str::<ChatCompletionChunk>(data) {
                                Ok(chunk) => {
                                    let Some(choice) = chunk.choices.first() else { continue; };
                                    if let Some(content) = &choice.delta.content {
                                        if !content.is_empty() {
                                            yield Ok(StreamEvent::TextDelta(content.clone()));
                                        }
                                    }
                                    for tcd in &choice.delta.tool_calls {
                                        let acc = tool_accum.entry(tcd.index).or_default();
                                        if let Some(id) = &tcd.id {
                                            if !id.is_empty() {
                                                acc.id = Some(id.clone());
                                            }
                                        }
                                        if let Some(t) = &tcd.call_type {
                                            if !t.is_empty() {
                                                acc.call_type = t.clone();
                                            }
                                        }
                                        if let Some(func) = &tcd.function {
                                            if let Some(name) = &func.name {
                                                if !name.is_empty() {
                                                    acc.name = Some(name.clone());
                                                }
                                            }
                                            // arguments chunks are additive strings
                                            if let Some(args) = &func.arguments {
                                                acc.arguments.push_str(args);
                                            }
                                        }
                                    }
                                    if let Some(reason) = &choice.finish_reason {
                                        log::debug!(
                                            "openrouter stream: finish_reason={}, {} tool_call(s) accumulated",
                                            reason,
                                            tool_accum.len()
                                        );
                                        saw_finish = true;
                                        if reason == "tool_calls" {
                                            // Definitive end-of-tool-calls. Flush.
                                            for (idx, acc) in std::mem::take(&mut tool_accum) {
                                                log::debug!(
                                                    "openrouter stream: finalizing tool call index={} name={:?} args_len={} args={}",
                                                    idx,
                                                    acc.name,
                                                    acc.arguments.len(),
                                                    acc.arguments,
                                                );
                                                if let Some(tc) = acc.into_tool_call() {
                                                    yield Ok(StreamEvent::ToolCallComplete(tc));
                                                }
                                            }
                                        }
                                        // Don't flush on "stop" here — that's end of a
                                        // text-only turn, and tool_accum should already
                                        // be empty. If it isn't, we'll flush at stream
                                        // close below.
                                    }
                                }
                                Err(e) => {
                                    log::debug!("Failed to parse OpenRouter chunk '{data}': {e}");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(OpenRouterError::HttpError(e));
                        return;
                    }
                }
            }
            // Stream ended without [DONE]; flush whatever's accumulated.
            let _ = saw_finish;
            for (idx, acc) in std::mem::take(&mut tool_accum) {
                log::debug!(
                    "openrouter stream: end-of-stream flush tool call index={} name={:?} args_len={} args={}",
                    idx,
                    acc.name,
                    acc.arguments.len(),
                    acc.arguments,
                );
                if let Some(tc) = acc.into_tool_call() {
                    yield Ok(StreamEvent::ToolCallComplete(tc));
                }
            }
        };

        Ok(stream)
    }
}

#[derive(Default, Debug, Clone)]
struct ToolCallAccum {
    id: Option<String>,
    call_type: String,
    name: Option<String>,
    arguments: String,
}

impl ToolCallAccum {
    fn into_tool_call(self) -> Option<ToolCall> {
        let id = self.id?;
        let name = self.name?;
        Some(ToolCall {
            id,
            call_type: if self.call_type.is_empty() {
                "function".to_string()
            } else {
                self.call_type
            },
            function: FunctionCall {
                name,
                arguments: self.arguments,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_chat_request_without_tools() {
        let req = ChatRequest {
            model: "x/y".to_string(),
            messages: vec![ChatMessage::user("Hi")],
            stream: Some(true),
            temperature: None,
            max_tokens: None,
            tools: vec![],
            tool_choice: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("\"tools\""));
    }

    #[test]
    fn test_serialize_chat_request_with_tools() {
        let req = ChatRequest {
            model: "x/y".to_string(),
            messages: vec![ChatMessage::user("Hi")],
            stream: Some(true),
            temperature: None,
            max_tokens: None,
            tools: vec![ToolDef {
                tool_type: "function".to_string(),
                function: FunctionDef {
                    name: "run".to_string(),
                    description: "run a thing".to_string(),
                    parameters: serde_json::json!({"type":"object"}),
                },
            }],
            tool_choice: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("\"run\""));
    }

    #[test]
    fn test_parse_text_chunk() {
        let json = r#"{"id":"1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_parse_tool_call_chunk() {
        let json = r#"{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"run_shell","arguments":"{\"cmd\":"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        let tcd = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(tcd.id, Some("call_abc".to_string()));
        assert_eq!(
            tcd.function.as_ref().and_then(|f| f.name.clone()),
            Some("run_shell".to_string())
        );
    }

    #[test]
    fn test_tool_call_accum() {
        let mut acc = ToolCallAccum::default();
        acc.id = Some("call_1".to_string());
        acc.name = Some("run_shell_command".to_string());
        acc.arguments = r#"{"command":"ls"}"#.to_string();
        let tc = acc.into_tool_call().unwrap();
        assert_eq!(tc.id, "call_1");
        assert_eq!(tc.function.arguments, r#"{"command":"ls"}"#);
    }
}