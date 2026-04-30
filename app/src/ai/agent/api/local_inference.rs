//! TBGB: Local inference routing for Ollama and OpenRouter models.
//!
//! When the user selects a model whose ID is prefixed with `ollama:` or `openrouter:`,
//! this module handles the request locally instead of sending it to Warp's server.
//!
//! The local client speaks OpenAI chat-completions protocol (for OpenRouter) or
//! Ollama's native chat protocol, and translates streaming output back into the
//! `warp_multi_agent_api::ResponseEvent` stream that the rest of Warp consumes.
//!
//! LIMITATIONS:
//! - No tool calls. The model sees the user prompt + conversation history only.
//!   Tool routing is out of scope for this phase.
//! - No agent orchestration, no ambient/cloud tasks, no artifact storage.

use anyhow::anyhow;
use futures::StreamExt;
use uuid::Uuid;
use warp_multi_agent_api as api;

use super::{RequestParams, ResponseStream};
use crate::ai::agent::AIAgentInput;
use super::ConvertToAPITypeError;
use ai::ollama_client::{ChatMessage as OllamaMessage, OllamaClient};
use ai::openrouter_client::{ChatMessage as OpenRouterMessage, OpenRouterClient};

/// Prefix that marks an Ollama-provided model in our LLMId namespace.
pub const OLLAMA_ID_PREFIX: &str = "ollama:";

/// Prefix that marks an OpenRouter-provided model in our LLMId namespace.
pub const OPENROUTER_ID_PREFIX: &str = "openrouter:";

/// Returns true if this model ID should be routed through local inference.
pub fn is_local_model(model_id: &str) -> bool {
    model_id.starts_with(OLLAMA_ID_PREFIX) || model_id.starts_with(OPENROUTER_ID_PREFIX)
}

/// Entry point: handle a request locally and return a ResponseStream.
pub async fn generate_local_output(
    params: RequestParams,
    cancellation_rx: futures::channel::oneshot::Receiver<()>,
) -> Result<ResponseStream, ConvertToAPITypeError> {
    let model_id_str: String = params.model.clone().into();

    let system_prompt = build_system_prompt();
    let messages = build_chat_messages(&system_prompt, &params.tasks, &params.input);

    // Generate stable IDs for this local exchange.
    let conversation_id = params
        .conversation_token
        .as_ref()
        .map(|t| t.as_str().to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let request_id = Uuid::new_v4().to_string();
    let run_id = format!("local-{}", Uuid::new_v4());

    // Stream events via an async channel so we can produce the ResponseStream type expected.
    let (tx, rx) = async_channel::unbounded::<super::Event>();

    // 1. Emit StreamInit immediately (even for failure cases — the UI needs
    //    it to bind the stream id to the conversation).
    let init_event = api::ResponseEvent {
        r#type: Some(api::response_event::Type::Init(
            api::response_event::StreamInit {
                conversation_id: conversation_id.clone(),
                request_id: request_id.clone(),
                run_id: run_id.clone(),
            },
        )),
    };
    let _ = tx.send(Ok(init_event)).await;

    // Determine whether this is turn 1 (no server-authoritative task yet)
    // or a continuation (server task already in params.tasks).
    //
    // compute_active_tasks() in AIConversation filters to only server-upgraded
    // tasks via task.source()?. On turn 1 the root task is still optimistic,
    // so params.tasks is EMPTY and we need to emit CreateTask to trigger the
    // optimistic->server upgrade of the root task. On turn 2+ the root is
    // already upgraded and present in params.tasks; emitting CreateTask again
    // would fail (UpgradeOptimisticTaskError::UnexpectedUpgrade from
    // into_server_created_task) and mangle added_exchanges_by_response,
    // producing the TaskNotFound error on the subsequent AddMessagesToTask.
    let existing_server_task_id = params
        .tasks
        .iter()
        .last()
        .map(|t| t.id.clone())
        .filter(|id| !id.is_empty());

    let (task_id, needs_create_task) = match existing_server_task_id {
        Some(id) => (id, false),
        None => (Uuid::new_v4().to_string(), true),
    };

    let response_message_id = Uuid::new_v4().to_string();

    // 2. Kick off streaming generation on a background task.
    let tx_clone = tx.clone();
    let model_id_for_task = model_id_str.clone();
    let messages_for_task = messages.clone();
    let api_keys = params.api_keys.clone();
    let ollama_url = params.ollama_url.clone();
    let task_id_for_spawn = task_id.clone();
    let response_message_id_for_spawn = response_message_id.clone();
    let request_id_for_spawn = request_id.clone();

    #[cfg(not(target_family = "wasm"))]
    let spawn = tokio::spawn(async move {
        run_local_inference(
            model_id_for_task,
            messages_for_task,
            api_keys,
            ollama_url,
            task_id_for_spawn,
            needs_create_task,
            response_message_id_for_spawn,
            request_id_for_spawn,
            tx_clone,
            cancellation_rx,
        )
        .await
    });

    // On wasm we don't have a tokio runtime; run inline. Currently this module
    // is not called in wasm paths but we guard anyway.
    #[cfg(target_family = "wasm")]
    {
        let _ = (
            tx_clone,
            model_id_for_task,
            messages_for_task,
            api_keys,
            ollama_url,
            task_id_for_spawn,
            needs_create_task,
            response_message_id_for_spawn,
            request_id_for_spawn,
            cancellation_rx,
        );
    }

    #[cfg(not(target_family = "wasm"))]
    let _ = spawn; // detached

    // Return the receiver as the ResponseStream.
    Ok(Box::pin(rx))
}

/// The core generation loop.
#[allow(clippy::too_many_arguments)]
async fn run_local_inference(
    model_id_str: String,
    messages: Vec<ChatMsg>,
    api_keys: Option<api::request::settings::ApiKeys>,
    ollama_url: Option<String>,
    task_id: String,
    needs_create_task: bool,
    response_message_id: String,
    request_id: String,
    tx: async_channel::Sender<super::Event>,
    mut cancellation_rx: futures::channel::oneshot::Receiver<()>,
) {
    // 1. BeginTransaction
    if tx
        .send(Ok(wrap_actions(vec![api::client_action::Action::BeginTransaction(
            api::client_action::BeginTransaction {},
        )])))
        .await
        .is_err()
    {
        return;
    }

    // 2. CreateTask only on turn 1, to upgrade the optimistic root task id
    //    to our deterministic one. On subsequent turns, params.tasks already
    //    contains a server-upgraded task; emitting CreateTask again would
    //    hit UpgradeOptimisticTaskError::UnexpectedUpgrade and corrupt the
    //    conversation's pending-exchange state.
    if needs_create_task {
        let task_proto = api::Task {
            id: task_id.clone(),
            description: String::new(),
            dependencies: None,
            messages: vec![],
            summary: String::new(),
            server_data: String::new(),
        };
        if tx
            .send(Ok(wrap_actions(vec![
                api::client_action::Action::CreateTask(api::client_action::CreateTask {
                    task: Some(task_proto),
                }),
            ])))
            .await
            .is_err()
        {
            return;
        }
    }

    // 3. Add an empty assistant message to the task.
    let initial_message = api::Message {
        id: response_message_id.clone(),
        task_id: task_id.clone(),
        request_id,
        timestamp: Some(current_proto_timestamp()),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput {
                text: String::new(),
            },
        )),
    };
    if tx
        .send(Ok(wrap_actions(vec![
            api::client_action::Action::AddMessagesToTask(
                api::client_action::AddMessagesToTask {
                    task_id: task_id.clone(),
                    messages: vec![initial_message],
                },
            ),
        ])))
        .await
        .is_err()
    {
        return;
    }

    // 4. Call the LLM and stream chunks.
    let result = stream_chunks(
        &model_id_str,
        messages,
        api_keys,
        ollama_url,
        &task_id,
        &response_message_id,
        &tx,
        &mut cancellation_rx,
    )
    .await;

    let finish_reason = match result {
        Ok(()) => api::response_event::stream_finished::Reason::Done(
            api::response_event::stream_finished::Done {},
        ),
        Err(e) => {
            // Append the error text to the assistant message so the user sees what went wrong.
            let err_text = format!("\n\n[local inference error: {e}]");
            let _ = tx
                .send(Ok(wrap_actions(vec![
                    api::client_action::Action::AppendToMessageContent(
                        append_action(&task_id, &response_message_id, &err_text),
                    ),
                ])))
                .await;
            api::response_event::stream_finished::Reason::Other(
                api::response_event::stream_finished::Other {},
            )
        }
    };

    // 5. CommitTransaction
    let _ = tx
        .send(Ok(wrap_actions(vec![
            api::client_action::Action::CommitTransaction(
                api::client_action::CommitTransaction {},
            ),
        ])))
        .await;

    // 6. StreamFinished
    let finished = api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(
            api::response_event::StreamFinished {
                reason: Some(finish_reason),
                conversation_usage_metadata: None,
                token_usage: vec![],
                should_refresh_model_config: false,
                request_cost: None,
            },
        )),
    };
    let _ = tx.send(Ok(finished)).await;
}

/// Call the LLM and stream chunks as AppendToMessageContent actions.
#[allow(clippy::too_many_arguments)]
async fn stream_chunks(
    model_id_str: &str,
    messages: Vec<ChatMsg>,
    api_keys: Option<api::request::settings::ApiKeys>,
    ollama_url: Option<String>,
    task_id: &str,
    message_id: &str,
    tx: &async_channel::Sender<super::Event>,
    cancellation_rx: &mut futures::channel::oneshot::Receiver<()>,
) -> Result<(), anyhow::Error> {
    if let Some(bare_model) = model_id_str.strip_prefix(OLLAMA_ID_PREFIX) {
        let url = ollama_url
            .filter(|u| !u.is_empty())
            .unwrap_or_else(|| ai::ollama_client::DEFAULT_OLLAMA_URL.to_string());
        let client = OllamaClient::with_base_url(&url);
        let ollama_messages: Vec<OllamaMessage> = messages
            .into_iter()
            .map(|m| OllamaMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        let mut stream = client.chat_streaming(bare_model, ollama_messages).await?;
        while let Some(chunk) = stream.next().await {
            if cancellation_rx.try_recv().ok().flatten().is_some() {
                return Ok(());
            }
            match chunk {
                Ok(ai::ollama_client::StreamChunk::Partial { message, .. }) => {
                    if !message.content.is_empty() {
                        tx.send(Ok(wrap_actions(vec![
                            api::client_action::Action::AppendToMessageContent(append_action(
                                task_id,
                                message_id,
                                &message.content,
                            )),
                        ])))
                        .await
                        .map_err(|_| anyhow!("receiver dropped"))?;
                    }
                }
                Ok(ai::ollama_client::StreamChunk::Complete { .. }) => break,
                Err(e) => return Err(anyhow!("ollama stream error: {e}")),
            }
        }
        Ok(())
    } else if let Some(bare_model) = model_id_str.strip_prefix(OPENROUTER_ID_PREFIX) {
        let key = api_keys
            .as_ref()
            .map(|k| k.open_router.clone())
            .filter(|k| !k.is_empty())
            .ok_or_else(|| anyhow!("OpenRouter API key not configured"))?;
        let client = OpenRouterClient::new(key);
        let or_messages: Vec<OpenRouterMessage> = messages
            .into_iter()
            .map(|m| OpenRouterMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        let stream = client.chat_streaming(bare_model, or_messages).await?;
        let mut stream = Box::pin(stream);
        while let Some(chunk) = stream.next().await {
            if cancellation_rx.try_recv().ok().flatten().is_some() {
                return Ok(());
            }
            match chunk {
                Ok(content) if !content.is_empty() => {
                    tx.send(Ok(wrap_actions(vec![
                        api::client_action::Action::AppendToMessageContent(append_action(
                            task_id, message_id, &content,
                        )),
                    ])))
                    .await
                    .map_err(|_| anyhow!("receiver dropped"))?;
                }
                Ok(_) => {}
                Err(e) => return Err(anyhow!("openrouter stream error: {e}")),
            }
        }
        Ok(())
    } else {
        Err(anyhow!("unknown local model id: {model_id_str}"))
    }
}

fn wrap_actions(actions: Vec<api::client_action::Action>) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: actions
                    .into_iter()
                    .map(|a| api::ClientAction { action: Some(a) })
                    .collect(),
            },
        )),
    }
}

fn append_action(
    task_id: &str,
    message_id: &str,
    chunk: &str,
) -> api::client_action::AppendToMessageContent {
    // We target the agent_output.text field for simple text appends.
    use prost_types::FieldMask;
    let message = api::Message {
        id: message_id.to_string(),
        task_id: task_id.to_string(),
        request_id: String::new(),
        timestamp: None,
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput {
                text: chunk.to_string(),
            },
        )),
    };
    api::client_action::AppendToMessageContent {
        task_id: task_id.to_string(),
        message: Some(message),
        mask: Some(FieldMask {
            paths: vec!["agent_output.text".to_string()],
        }),
    }
}

/// Minimal system prompt for local models. Keeps things neutral — we're not the
/// full Warp agent, but we mention the context.
fn build_system_prompt() -> String {
    r#"You are a helpful AI assistant running inside the Warp terminal.
The user is in a command-line environment. Keep answers concise and practical.
You do not currently have access to tool-calling (shell commands, file access).
Just respond with plain text."#
        .to_string()
}

#[derive(Clone, Debug)]
struct ChatMsg {
    role: String,
    content: String,
}

/// Walk existing tasks → messages to build conversation history, then append
/// the latest user input as the final user turn.
fn build_chat_messages(
    system_prompt: &str,
    tasks: &[api::Task],
    input: &[AIAgentInput],
) -> Vec<ChatMsg> {
    let mut messages = vec![ChatMsg {
        role: "system".to_string(),
        content: system_prompt.to_string(),
    }];

    // Include conversation history from existing tasks.
    for task in tasks {
        for m in &task.messages {
            if let Some(inner) = &m.message {
                match inner {
                    api::message::Message::UserQuery(q) => {
                        if !q.query.is_empty() {
                            messages.push(ChatMsg {
                                role: "user".to_string(),
                                content: q.query.clone(),
                            });
                        }
                    }
                    api::message::Message::AgentOutput(a) => {
                        if !a.text.is_empty() {
                            messages.push(ChatMsg {
                                role: "assistant".to_string(),
                                content: a.text.clone(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Append the current user input.
    let current_user_text = extract_user_text(input);
    if !current_user_text.is_empty() {
        messages.push(ChatMsg {
            role: "user".to_string(),
            content: current_user_text,
        });
    }

    messages
}

fn extract_user_text(input: &[AIAgentInput]) -> String {
    for ai in input {
        match ai {
            AIAgentInput::UserQuery { query, .. } => {
                if !query.is_empty() {
                    return query.clone();
                }
            }
            AIAgentInput::AutoCodeDiffQuery { query, .. } => {
                if !query.is_empty() {
                    return query.clone();
                }
            }
            AIAgentInput::CreateNewProject { query, .. } => {
                if !query.is_empty() {
                    return query.clone();
                }
            }
            _ => {}
        }
    }
    String::new()
}

fn current_proto_timestamp() -> prost_types::Timestamp {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    prost_types::Timestamp {
        seconds: duration.as_secs() as i64,
        nanos: duration.subsec_nanos() as i32,
    }
}

// End of file.