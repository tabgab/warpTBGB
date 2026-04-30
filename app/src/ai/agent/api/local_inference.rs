//! TBGB: Local inference routing for Ollama and OpenRouter models.
//!
//! When the user selects a model whose ID is prefixed with `ollama:` or `openrouter:`,
//! this module handles the request locally instead of sending it to Warp's server.
//!
//! Supports OpenAI-format tool calling:
//! - Turn 1: send user prompt + tool schemas; if model responds with tool_calls,
//!   emit them as proto Message::ToolCall actions, which Warp's existing
//!   executors will handle (preprocessing / approval UI / execution).
//! - Turn N: receive AIAgentInput::ActionResult entries in params.input,
//!   translate to OpenAI-format tool-role chat messages, send back to model.
//! - Warp's `send_follow_up_for_conversation` automatically triggers turn N
//!   after tool results are collected, so the loop runs without special plumbing.

use anyhow::anyhow;
use futures::StreamExt;
use uuid::Uuid;
use warp_multi_agent_api as api;

use super::local_tools;
use super::ConvertToAPITypeError;
use super::{RequestParams, ResponseStream};
use crate::ai::agent::AIAgentInput;
use ai::ollama_client::{
    self as ollama, ChatMessage as OllamaMessage, FunctionCall as OllamaFunctionCall,
    OllamaClient, ToolCall as OllamaToolCall,
};
use ai::openrouter_client::{
    ChatMessage as OpenRouterMessage, FunctionCall as OpenRouterFunctionCall, OpenRouterClient,
    StreamEvent as OpenRouterStreamEvent, ToolCall as OpenRouterToolCall,
};

pub const OLLAMA_ID_PREFIX: &str = "ollama:";
pub const OPENROUTER_ID_PREFIX: &str = "openrouter:";

pub fn is_local_model(model_id: &str) -> bool {
    model_id.starts_with(OLLAMA_ID_PREFIX) || model_id.starts_with(OPENROUTER_ID_PREFIX)
}

pub async fn generate_local_output(
    params: RequestParams,
    cancellation_rx: futures::channel::oneshot::Receiver<()>,
) -> Result<ResponseStream, ConvertToAPITypeError> {
    let model_id_str: String = params.model.clone().into();

    let chat_messages = build_chat_messages(&params);

    // Stable conversation / request / run ids for this turn.
    let conversation_id = params
        .conversation_token
        .as_ref()
        .map(|t| t.as_str().to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let request_id = Uuid::new_v4().to_string();
    let run_id = format!("local-{}", Uuid::new_v4());

    let (tx, rx) = async_channel::unbounded::<super::Event>();

    // Emit StreamInit immediately.
    let _ = tx
        .send(Ok(api::ResponseEvent {
            r#type: Some(api::response_event::Type::Init(
                api::response_event::StreamInit {
                    conversation_id: conversation_id.clone(),
                    request_id: request_id.clone(),
                    run_id: run_id.clone(),
                },
            )),
        }))
        .await;

    // Decide whether we still need CreateTask (turn 1 iff params.tasks is empty).
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
    let tx_clone = tx.clone();
    let api_keys = params.api_keys.clone();
    let ollama_url = params.ollama_url.clone();

    #[cfg(not(target_family = "wasm"))]
    let _ = tokio::spawn(async move {
        run_local_inference(
            model_id_str,
            chat_messages,
            api_keys,
            ollama_url,
            task_id,
            needs_create_task,
            response_message_id,
            request_id,
            tx_clone,
            cancellation_rx,
        )
        .await
    });

    #[cfg(target_family = "wasm")]
    {
        let _ = (
            tx_clone,
            model_id_str,
            chat_messages,
            api_keys,
            ollama_url,
            task_id,
            needs_create_task,
            response_message_id,
            request_id,
            cancellation_rx,
        );
    }

    Ok(Box::pin(rx))
}

#[allow(clippy::too_many_arguments)]
async fn run_local_inference(
    model_id_str: String,
    chat_messages: Vec<ChatMsg>,
    api_keys: Option<api::request::settings::ApiKeys>,
    ollama_url: Option<String>,
    task_id: String,
    needs_create_task: bool,
    response_message_id: String,
    request_id: String,
    tx: async_channel::Sender<super::Event>,
    mut cancellation_rx: futures::channel::oneshot::Receiver<()>,
) {
    // BeginTransaction
    if tx
        .send(Ok(wrap_actions(vec![
            api::client_action::Action::BeginTransaction(api::client_action::BeginTransaction {}),
        ])))
        .await
        .is_err()
    {
        return;
    }

    // Turn 1 only: CreateTask to upgrade the optimistic root task.
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

    // Initial empty assistant message for text streaming.
    let initial_message = api::Message {
        id: response_message_id.clone(),
        task_id: task_id.clone(),
        request_id: request_id.clone(),
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
            api::client_action::Action::AddMessagesToTask(api::client_action::AddMessagesToTask {
                task_id: task_id.clone(),
                messages: vec![initial_message],
            }),
        ])))
        .await
        .is_err()
    {
        return;
    }

    // Stream the model. Collect any tool calls it emits.
    let stream_result = stream_and_collect(
        &model_id_str,
        chat_messages,
        api_keys,
        ollama_url,
        &task_id,
        &response_message_id,
        &request_id,
        &tx,
        &mut cancellation_rx,
    )
    .await;

    let finish_reason = match stream_result {
        Ok(()) => api::response_event::stream_finished::Reason::Done(
            api::response_event::stream_finished::Done {},
        ),
        Err(e) => {
            let err_text = format!("\n\n[local inference error: {e}]");
            let _ = tx
                .send(Ok(wrap_actions(vec![
                    api::client_action::Action::AppendToMessageContent(append_action(
                        &task_id,
                        &response_message_id,
                        &err_text,
                    )),
                ])))
                .await;
            api::response_event::stream_finished::Reason::Other(
                api::response_event::stream_finished::Other {},
            )
        }
    };

    // CommitTransaction
    let _ = tx
        .send(Ok(wrap_actions(vec![
            api::client_action::Action::CommitTransaction(
                api::client_action::CommitTransaction {},
            ),
        ])))
        .await;

    // StreamFinished
    let _ = tx
        .send(Ok(api::ResponseEvent {
            r#type: Some(api::response_event::Type::Finished(
                api::response_event::StreamFinished {
                    reason: Some(finish_reason),
                    conversation_usage_metadata: None,
                    token_usage: vec![],
                    should_refresh_model_config: false,
                    request_cost: None,
                },
            )),
        }))
        .await;
}

#[allow(clippy::too_many_arguments)]
async fn stream_and_collect(
    model_id_str: &str,
    chat_messages: Vec<ChatMsg>,
    api_keys: Option<api::request::settings::ApiKeys>,
    ollama_url: Option<String>,
    task_id: &str,
    message_id: &str,
    request_id: &str,
    tx: &async_channel::Sender<super::Event>,
    cancellation_rx: &mut futures::channel::oneshot::Receiver<()>,
) -> Result<(), anyhow::Error> {
    if let Some(bare_model) = model_id_str.strip_prefix(OLLAMA_ID_PREFIX) {
        run_ollama(
            bare_model,
            chat_messages,
            ollama_url,
            task_id,
            message_id,
            request_id,
            tx,
            cancellation_rx,
        )
        .await
    } else if let Some(bare_model) = model_id_str.strip_prefix(OPENROUTER_ID_PREFIX) {
        run_openrouter(
            bare_model,
            chat_messages,
            api_keys,
            task_id,
            message_id,
            request_id,
            tx,
            cancellation_rx,
        )
        .await
    } else {
        Err(anyhow!("unknown local model id: {model_id_str}"))
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_ollama(
    bare_model: &str,
    chat_messages: Vec<ChatMsg>,
    ollama_url: Option<String>,
    task_id: &str,
    message_id: &str,
    request_id: &str,
    tx: &async_channel::Sender<super::Event>,
    cancellation_rx: &mut futures::channel::oneshot::Receiver<()>,
) -> Result<(), anyhow::Error> {
    let url = ollama_url
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| ollama::DEFAULT_OLLAMA_URL.to_string());
    let client = OllamaClient::with_base_url(&url);

    let tool_defs = local_tools::ollama_tool_defs();
    let messages = chat_messages.into_iter().map(to_ollama_message).collect();

    let stream = client
        .chat_streaming(bare_model, messages, tool_defs)
        .await?;
    let mut stream = Box::pin(stream);

    while let Some(chunk) = stream.next().await {
        if cancellation_rx.try_recv().ok().flatten().is_some() {
            return Ok(());
        }
        match chunk {
            Ok(ollama::StreamChunk::Partial { message, .. })
            | Ok(ollama::StreamChunk::Complete { message, .. }) => {
                // Text content (if any)
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
                // Tool calls
                if !message.tool_calls.is_empty() {
                    let tool_messages =
                        tool_calls_to_proto_messages(&message.tool_calls, task_id, request_id);
                    if !tool_messages.is_empty() {
                        tx.send(Ok(wrap_actions(vec![
                            api::client_action::Action::AddMessagesToTask(
                                api::client_action::AddMessagesToTask {
                                    task_id: task_id.to_string(),
                                    messages: tool_messages,
                                },
                            ),
                        ])))
                        .await
                        .map_err(|_| anyhow!("receiver dropped"))?;
                    }
                }
            }
            Err(e) => return Err(anyhow!("ollama stream error: {e}")),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_openrouter(
    bare_model: &str,
    chat_messages: Vec<ChatMsg>,
    api_keys: Option<api::request::settings::ApiKeys>,
    task_id: &str,
    message_id: &str,
    request_id: &str,
    tx: &async_channel::Sender<super::Event>,
    cancellation_rx: &mut futures::channel::oneshot::Receiver<()>,
) -> Result<(), anyhow::Error> {
    let key = api_keys
        .as_ref()
        .map(|k| k.open_router.clone())
        .filter(|k| !k.is_empty())
        .ok_or_else(|| anyhow!("OpenRouter API key not configured"))?;
    let client = OpenRouterClient::new(key);

    let tool_defs = local_tools::openrouter_tool_defs();
    let messages = chat_messages.into_iter().map(to_openrouter_message).collect();

    let stream = client
        .chat_streaming(bare_model, messages, tool_defs)
        .await?;
    let mut stream = Box::pin(stream);
    let mut emitted_tool_calls: Vec<OpenRouterToolCall> = Vec::new();

    while let Some(event) = stream.next().await {
        if cancellation_rx.try_recv().ok().flatten().is_some() {
            return Ok(());
        }
        match event {
            Ok(OpenRouterStreamEvent::TextDelta(content)) if !content.is_empty() => {
                tx.send(Ok(wrap_actions(vec![
                    api::client_action::Action::AppendToMessageContent(append_action(
                        task_id,
                        message_id,
                        &content,
                    )),
                ])))
                .await
                .map_err(|_| anyhow!("receiver dropped"))?;
            }
            Ok(OpenRouterStreamEvent::TextDelta(_)) => {}
            Ok(OpenRouterStreamEvent::ToolCallComplete(tc)) => {
                emitted_tool_calls.push(tc);
            }
            Err(e) => return Err(anyhow!("openrouter stream error: {e}")),
        }
    }

    // Batch-emit tool calls at end of stream (they were accumulated incrementally).
    if !emitted_tool_calls.is_empty() {
        let tool_messages =
            openrouter_tool_calls_to_proto_messages(&emitted_tool_calls, task_id, request_id);
        if !tool_messages.is_empty() {
            tx.send(Ok(wrap_actions(vec![
                api::client_action::Action::AddMessagesToTask(
                    api::client_action::AddMessagesToTask {
                        task_id: task_id.to_string(),
                        messages: tool_messages,
                    },
                ),
            ])))
            .await
            .map_err(|_| anyhow!("receiver dropped"))?;
        }
    }

    Ok(())
}

fn tool_calls_to_proto_messages(
    tool_calls: &[OllamaToolCall],
    task_id: &str,
    request_id: &str,
) -> Vec<api::Message> {
    tool_calls
        .iter()
        .filter_map(|tc| {
            // Ollama sends arguments as a JSON value (object). Stringify it.
            let args_str = serde_json::to_string(&tc.function.arguments).ok()?;
            let tool_call_id = format!("call-{}", Uuid::new_v4());
            let proto_tc = local_tools::into_proto_tool_call(
                tool_call_id.clone(),
                &tc.function.name,
                &args_str,
            )?;
            Some(api::Message {
                id: Uuid::new_v4().to_string(),
                task_id: task_id.to_string(),
                request_id: request_id.to_string(),
                timestamp: Some(current_proto_timestamp()),
                server_message_data: String::new(),
                citations: vec![],
                message: Some(api::message::Message::ToolCall(proto_tc)),
            })
        })
        .collect()
}

fn openrouter_tool_calls_to_proto_messages(
    tool_calls: &[OpenRouterToolCall],
    task_id: &str,
    request_id: &str,
) -> Vec<api::Message> {
    tool_calls
        .iter()
        .filter_map(|tc| {
            // OpenRouter/OpenAI: arguments is already a JSON string.
            log::debug!(
                "Local inference: model emitted tool call '{}' with args: {}",
                tc.function.name,
                tc.function.arguments
            );
            let proto_tc = local_tools::into_proto_tool_call(
                tc.id.clone(),
                &tc.function.name,
                &tc.function.arguments,
            );
            if proto_tc.is_none() {
                log::warn!(
                    "Local inference: could not translate tool call '{}' — \
                     unknown function name or malformed arguments. Args: {}",
                    tc.function.name,
                    tc.function.arguments
                );
            }
            let proto_tc = proto_tc?;
            Some(api::Message {
                id: Uuid::new_v4().to_string(),
                task_id: task_id.to_string(),
                request_id: request_id.to_string(),
                timestamp: Some(current_proto_timestamp()),
                server_message_data: String::new(),
                citations: vec![],
                message: Some(api::message::Message::ToolCall(proto_tc)),
            })
        })
        .collect()
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

fn build_system_prompt() -> String {
    r#"You are a helpful AI coding assistant running inside the Warp terminal.

You have the following tools:

  - run_shell_command: Run a shell command and see its output. Use this to
    execute code, inspect the environment, or gather information.
  - read_files: Read the contents of one or more files.
  - grep: Search file contents for a pattern.
  - file_glob: Find files by glob pattern.
  - apply_file_diffs: CREATE NEW FILES, edit existing files, or delete files.
    USE THIS TOOL WHENEVER THE USER ASKS YOU TO WRITE CODE. Do NOT use
    run_shell_command with `cat > file <<EOF ...` or `echo > file` to write
    code; always use apply_file_diffs instead, which shows the user a proper
    diff preview.

Workflow for code-writing tasks:
  1. If you need to understand existing code, use read_files / grep first.
  2. Write or modify code by calling apply_file_diffs with the `new_files`
     and/or `edits` arrays populated. Each new_files entry must have a
     non-empty `content` field with the full file contents.
  3. After applying diffs, optionally run_shell_command to verify (e.g.
     compile, test, or execute the result).
  4. When the task is complete, respond with plain text summarizing what
     you did. No more tool calls needed.

Tool call guidelines:
  - Set is_read_only=true on run_shell_command for safe, non-mutating
    commands (ls, cat, git status, etc.).
  - Be concise. Don't repeat full tool output back verbatim in your reply.
  - Never invent file contents — if the user asks for new code, fully
    specify its contents in an apply_file_diffs.new_files entry.
"#
    .to_string()
}

/// Intermediate chat-message representation used before translating to
/// provider-specific types.
#[derive(Clone, Debug)]
enum ChatMsg {
    System(String),
    User(String),
    Assistant {
        text: String,
        tool_calls: Vec<ProtoToolCallRef>,
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: String,
    },
}

#[derive(Clone, Debug)]
struct ProtoToolCallRef {
    id: String,
    name: String,
    arguments_json: String,
}

fn to_ollama_message(m: ChatMsg) -> OllamaMessage {
    match m {
        ChatMsg::System(c) => OllamaMessage {
            role: "system".to_string(),
            content: c,
            ..Default::default()
        },
        ChatMsg::User(c) => OllamaMessage {
            role: "user".to_string(),
            content: c,
            ..Default::default()
        },
        ChatMsg::Assistant { text, tool_calls } => OllamaMessage {
            role: "assistant".to_string(),
            content: text,
            tool_calls: tool_calls
                .into_iter()
                .map(|tc| OllamaToolCall {
                    function: OllamaFunctionCall {
                        name: tc.name,
                        arguments: serde_json::from_str(&tc.arguments_json)
                            .unwrap_or(serde_json::Value::Null),
                    },
                })
                .collect(),
            ..Default::default()
        },
        ChatMsg::ToolResult {
            tool_call_id: _,
            tool_name,
            content,
        } => OllamaMessage {
            role: "tool".to_string(),
            content,
            tool_name: Some(tool_name),
            ..Default::default()
        },
    }
}

fn to_openrouter_message(m: ChatMsg) -> OpenRouterMessage {
    match m {
        ChatMsg::System(c) => OpenRouterMessage::system(c),
        ChatMsg::User(c) => OpenRouterMessage::user(c),
        ChatMsg::Assistant { text, tool_calls } => {
            if tool_calls.is_empty() {
                OpenRouterMessage::assistant_text(text)
            } else {
                OpenRouterMessage::assistant_tool_calls(
                    tool_calls
                        .into_iter()
                        .map(|tc| OpenRouterToolCall {
                            id: tc.id,
                            call_type: "function".to_string(),
                            function: OpenRouterFunctionCall {
                                name: tc.name,
                                arguments: tc.arguments_json,
                            },
                        })
                        .collect(),
                )
            }
        }
        ChatMsg::ToolResult {
            tool_call_id,
            tool_name,
            content,
        } => OpenRouterMessage::tool_result(tool_call_id, tool_name, content),
    }
}

/// Build the chat message history for the model. Pulls:
/// 1. Prior task messages (user queries, agent outputs, tool calls, tool results)
///    from params.tasks — this reconstructs the conversation.
/// 2. The current turn's input (user query OR tool-call-result entries).
fn build_chat_messages(params: &RequestParams) -> Vec<ChatMsg> {
    let mut out = vec![ChatMsg::System(build_system_prompt())];

    // Replay prior conversation. We coalesce consecutive assistant messages
    // (text + tool_calls) into a single ChatMsg::Assistant so the provider
    // sees a valid OpenAI-format sequence:
    //   user -> assistant{text, tool_calls: [...]} -> tool(r1) -> tool(r2) -> ...
    // without extra empty assistant messages in between.
    for task in &params.tasks {
        for msg in &task.messages {
            let Some(inner) = &msg.message else { continue };
            match inner {
                api::message::Message::UserQuery(q) if !q.query.is_empty() => {
                    out.push(ChatMsg::User(q.query.clone()));
                }
                api::message::Message::AgentOutput(a) if !a.text.is_empty() => {
                    // Merge into preceding assistant message if it exists and
                    // has no tool_calls yet; otherwise start a new one.
                    match out.last_mut() {
                        Some(ChatMsg::Assistant { text, tool_calls })
                            if tool_calls.is_empty() =>
                        {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(&a.text);
                        }
                        _ => out.push(ChatMsg::Assistant {
                            text: a.text.clone(),
                            tool_calls: vec![],
                        }),
                    }
                }
                api::message::Message::ToolCall(tc) => {
                    let (name, args) = proto_tool_call_to_name_args(tc);
                    let call_ref = ProtoToolCallRef {
                        id: tc.tool_call_id.clone(),
                        name,
                        arguments_json: args,
                    };
                    // Merge into the preceding assistant message so text + all
                    // tool_calls emitted in the same turn form ONE message.
                    match out.last_mut() {
                        Some(ChatMsg::Assistant { tool_calls, .. }) => {
                            tool_calls.push(call_ref);
                        }
                        _ => out.push(ChatMsg::Assistant {
                            text: String::new(),
                            tool_calls: vec![call_ref],
                        }),
                    }
                }
                api::message::Message::ToolCallResult(tcr) => {
                    out.push(ChatMsg::ToolResult {
                        tool_call_id: tcr.tool_call_id.clone(),
                        tool_name: String::new(),
                        content: stringify_tool_call_result(tcr),
                    });
                }
                _ => {}
            }
        }
    }

    // Current turn's input.
    for input in &params.input {
        match input {
            AIAgentInput::UserQuery { query, .. } if !query.is_empty() => {
                out.push(ChatMsg::User(query.clone()));
            }
            AIAgentInput::ActionResult { result, .. } => {
                out.push(ChatMsg::ToolResult {
                    tool_call_id: result.id.to_string(),
                    tool_name: String::new(),
                    content: format!("{}", result.result),
                });
            }
            AIAgentInput::ResumeConversation { .. } => {
                // Nothing new to add; prior context carries the state.
            }
            _ => {}
        }
    }

    out
}

/// Best-effort extraction of (function_name, arguments_json_string) from a
/// proto ToolCall, matching the OpenAI/Ollama function calling shape.
fn proto_tool_call_to_name_args(tc: &api::message::ToolCall) -> (String, String) {
    match &tc.tool {
        Some(api::message::tool_call::Tool::RunShellCommand(c)) => {
            let args = serde_json::json!({"command": c.command, "is_read_only": c.is_read_only});
            ("run_shell_command".into(), args.to_string())
        }
        Some(api::message::tool_call::Tool::ReadFiles(r)) => {
            let paths: Vec<String> = r.files.iter().map(|f| f.name.clone()).collect();
            let args = serde_json::json!({"paths": paths});
            ("read_files".into(), args.to_string())
        }
        Some(api::message::tool_call::Tool::Grep(g)) => {
            let args = serde_json::json!({"queries": g.queries, "path": g.path});
            ("grep".into(), args.to_string())
        }
        Some(api::message::tool_call::Tool::FileGlobV2(g)) => {
            let args =
                serde_json::json!({"patterns": g.patterns, "search_dir": g.search_dir});
            ("file_glob".into(), args.to_string())
        }
        Some(api::message::tool_call::Tool::ApplyFileDiffs(a)) => {
            let new_files: Vec<serde_json::Value> = a
                .new_files
                .iter()
                .map(|nf| {
                    serde_json::json!({"file_path": nf.file_path, "content": nf.content})
                })
                .collect();
            let edits: Vec<serde_json::Value> = a
                .diffs
                .iter()
                .map(|d| {
                    serde_json::json!({
                        "file_path": d.file_path,
                        "search": d.search,
                        "replace": d.replace,
                    })
                })
                .collect();
            let deleted_files: Vec<&str> =
                a.deleted_files.iter().map(|d| d.file_path.as_str()).collect();
            let args = serde_json::json!({
                "summary": a.summary,
                "new_files": new_files,
                "edits": edits,
                "deleted_files": deleted_files,
            });
            ("apply_file_diffs".into(), args.to_string())
        }
        _ => (
            "<unknown_tool>".into(),
            local_tools::tool_call_summary(tc),
        ),
    }
}

fn stringify_tool_call_result(tcr: &api::message::ToolCallResult) -> String {
    // Produce a clean, model-readable summary for each tool-result variant,
    // mirroring what Warp's conversation_yaml.rs does for its own search
    // serialization. The raw proto Debug dump is unreadable; this is what the
    // model sees when replaying prior turns' tool calls on turn 3+.
    use api::message::tool_call_result::Result as R;
    let Some(result) = tcr.result.as_ref() else {
        return "(no result)".to_string();
    };
    match result {
        R::RunShellCommand(r) => {
            if let Some(res) = &r.result {
                use api::run_shell_command_result::Result as RR;
                match res {
                    RR::CommandFinished(c) => format!(
                        "exit_code: {}\noutput:\n{}",
                        c.exit_code,
                        truncate(&c.output, 8192)
                    ),
                    RR::LongRunningCommandSnapshot(s) => {
                        format!("status: long_running\noutput:\n{}", truncate(&s.output, 8192))
                    }
                    RR::PermissionDenied(_) => "status: permission_denied".to_string(),
                }
            } else {
                "(empty shell result)".to_string()
            }
        }
        R::ReadFiles(r) => {
            if let Some(res) = &r.result {
                use api::read_files_result::Result as RR;
                match res {
                    RR::TextFilesSuccess(s) => {
                        let mut out = String::new();
                        for f in &s.files {
                            out.push_str(&format!("--- {} ---\n{}\n", f.file_path, f.content));
                        }
                        if out.is_empty() {
                            "(no files returned)".to_string()
                        } else {
                            out
                        }
                    }
                    RR::AnyFilesSuccess(s) => {
                        format!("{} file(s) returned (non-text / binary)", s.files.len())
                    }
                    RR::Error(e) => format!("error: {}", e.message),
                }
            } else {
                "(empty read_files result)".to_string()
            }
        }
        R::Grep(r) => {
            if let Some(res) = &r.result {
                use api::grep_result::Result as RR;
                match res {
                    RR::Success(s) => {
                        let mut out = String::from("matched_files:\n");
                        for f in &s.matched_files {
                            out.push_str(&format!("  - {}\n", f.file_path));
                            for line in &f.matched_lines {
                                out.push_str(&format!("      line {}\n", line.line_number));
                            }
                        }
                        out
                    }
                    RR::Error(e) => format!("error: {}", e.message),
                }
            } else {
                "(empty grep result)".to_string()
            }
        }
        R::FileGlobV2(r) => {
            if let Some(res) = &r.result {
                use api::file_glob_v2_result::Result as RR;
                match res {
                    RR::Success(s) => {
                        let paths: Vec<&str> =
                            s.matched_files.iter().map(|f| f.file_path.as_str()).collect();
                        format!("matching paths:\n{}", paths.join("\n"))
                    }
                    RR::Error(e) => format!("error: {}", e.message),
                }
            } else {
                "(empty glob result)".to_string()
            }
        }
        R::ApplyFileDiffs(r) => {
            if let Some(res) = &r.result {
                use api::apply_file_diffs_result::Result as RR;
                match res {
                    RR::Success(s) => {
                        let mut out = String::from("status: success\n");
                        for uf in &s.updated_files_v2 {
                            if let Some(f) = &uf.file {
                                out.push_str(&format!("  updated: {}\n", f.file_path));
                            }
                        }
                        for df in &s.deleted_files {
                            out.push_str(&format!("  deleted: {}\n", df.file_path));
                        }
                        out
                    }
                    RR::Error(e) => format!("error: {}", e.message),
                }
            } else {
                "(empty apply_file_diffs result)".to_string()
            }
        }
        other => format!("{other:?}"),
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}\n... (truncated, {} bytes total)", &s[..max], s.len())
    } else {
        s.to_string()
    }
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