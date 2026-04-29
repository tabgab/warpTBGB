//! OpenRouter API client.
//!
//! OpenRouter is an OpenAI-API-compatible aggregator at https://openrouter.ai/api/v1.
//! It serves models from many providers using a unified chat-completions API.

use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default OpenRouter base URL.
pub const DEFAULT_OPENROUTER_URL: &str = "https://openrouter.ai/api/v1";

/// Default request timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(180);

/// A chat message (OpenAI format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat completion request (OpenAI format, OpenRouter-compatible subset).
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

/// Non-streaming chat completion response (OpenAI format).
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

/// Streaming chat completion chunk (OpenAI SSE format).
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

    /// Non-streaming chat completion.
    pub async fn chat(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
    ) -> OpenRouterResult<ChatCompletionResponse> {
        let request = ChatRequest {
            model: model.to_string(),
            messages,
            stream: Some(false),
            temperature: None,
            max_tokens: None,
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

        let body = response.text().await?;
        let result: ChatCompletionResponse = serde_json::from_str(&body)?;
        Ok(result)
    }

    /// Streaming chat completion; yields content deltas as strings.
    pub async fn chat_streaming(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
    ) -> OpenRouterResult<impl futures::Stream<Item = OpenRouterResult<String>> + Send + 'static>
    {
        let request = ChatRequest {
            model: model.to_string(),
            messages,
            stream: Some(true),
            temperature: None,
            max_tokens: None,
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

        // Server-Sent Events: lines prefixed with "data: ", terminated by blank line.
        // "data: [DONE]" indicates end of stream.
        let stream = async_stream::stream! {
            let mut bytes_stream = response.bytes_stream();
            let mut buffer = String::new();
            while let Some(chunk) = bytes_stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        while let Some(newline_idx) = buffer.find('\n') {
                            let line = buffer[..newline_idx].trim().to_string();
                            buffer.drain(..=newline_idx);
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data.trim() == "[DONE]" {
                                    return;
                                }
                                match serde_json::from_str::<ChatCompletionChunk>(data) {
                                    Ok(chunk) => {
                                        if let Some(choice) = chunk.choices.first() {
                                            if let Some(content) = &choice.delta.content {
                                                if !content.is_empty() {
                                                    yield Ok(content.clone());
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        log::debug!("Failed to parse OpenRouter chunk '{data}': {e}");
                                    }
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
        };

        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_chat_request() {
        let req = ChatRequest {
            model: "deepseek/deepseek-v4-pro".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            }],
            stream: Some(true),
            temperature: None,
            max_tokens: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"stream\":true"));
        assert!(!json.contains("temperature"));
    }

    #[test]
    fn test_parse_chunk() {
        let json = r#"{"id":"1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }
}