//! TBGB: Tool definitions + translation between OpenAI function-calling format
//! and Warp's multi_agent_api ToolCall proto.
//!
//! The model speaks OpenAI/Ollama-style function calling. Warp internally
//! speaks its own proto. We translate between them here.
//!
//! Each supported tool has:
//! - an OpenAI-format `ToolDef` (schema) we tell the model about
//! - a translator from the model's JSON arguments to a `warp_multi_agent_api::message::tool_call::Tool` proto variant
//! - (on turn 2+) a translator from a finished `AIAgentActionResult` back into
//!   an OpenAI-format tool-role message the model can consume.

use ai::openrouter_client::{FunctionDef, ToolDef as OpenRouterToolDef};
use ai::ollama_client::{
    FunctionDef as OllamaFunctionDef, ToolDef as OllamaToolDef,
};
use serde_json::{json, Value};
use warp_multi_agent_api as api;

/// Build the complete list of tool schemas to advertise to the model.
/// Start with the most useful subset; we can expand later.
pub fn openrouter_tool_defs() -> Vec<OpenRouterToolDef> {
    tool_schemas()
        .into_iter()
        .map(|(name, description, parameters)| OpenRouterToolDef {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name,
                description,
                parameters,
            },
        })
        .collect()
}

pub fn ollama_tool_defs() -> Vec<OllamaToolDef> {
    tool_schemas()
        .into_iter()
        .map(|(name, description, parameters)| OllamaToolDef {
            tool_type: "function".to_string(),
            function: OllamaFunctionDef {
                name,
                description,
                parameters,
            },
        })
        .collect()
}

fn tool_schemas() -> Vec<(String, String, Value)> {
    vec![
        (
            "run_shell_command".to_string(),
            "Run a shell command on the user's machine and return its output. \
             The user may be prompted to approve the command before it runs."
                .to_string(),
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute (runs in the user's shell, e.g. bash or zsh)."
                    },
                    "is_read_only": {
                        "type": "boolean",
                        "description": "Set to true if the command only reads state and has no side effects (e.g. ls, cat, git status)."
                    }
                },
                "required": ["command"]
            }),
        ),
        (
            "read_files".to_string(),
            "Read the contents of one or more files. Use this to inspect source code or configuration before making changes.".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Absolute or working-directory-relative file paths to read."
                    }
                },
                "required": ["paths"]
            }),
        ),
        (
            "grep".to_string(),
            "Search for a text pattern across files. Uses ripgrep semantics.".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Patterns to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to search in (file or directory). Defaults to the working directory."
                    }
                },
                "required": ["queries"]
            }),
        ),
        (
            "file_glob".to_string(),
            "Find files whose names match one or more glob patterns (e.g. \"**/*.rs\").".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns to match (e.g. [\"**/*.rs\", \"Cargo.toml\"])."
                    },
                    "search_dir": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to the working directory."
                    }
                },
                "required": ["patterns"]
            }),
        ),
    ]
}

/// Parse a model-emitted tool call (as OpenAI-format function name + args
/// JSON) into a Warp `Message::ToolCall` proto.
///
/// `arguments_json` may be a stringified JSON (OpenRouter/OpenAI convention)
/// OR an already-parsed JSON value (Ollama convention). Pass as a JSON string
/// in either case; caller is responsible for stringifying Ollama's object.
///
/// Returns `None` if the function name is unknown or arguments are malformed.
pub fn into_proto_tool_call(
    tool_call_id: String,
    function_name: &str,
    arguments_json: &str,
) -> Option<api::message::ToolCall> {
    let args: Value = serde_json::from_str(arguments_json).ok()?;
    let tool = match function_name {
        "run_shell_command" => {
            let command = args.get("command")?.as_str()?.to_string();
            let is_read_only = args
                .get("is_read_only")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            Some(api::message::tool_call::Tool::RunShellCommand(
                api::message::tool_call::RunShellCommand {
                    command,
                    is_read_only,
                    uses_pager: false,
                    citations: vec![],
                    is_risky: !is_read_only,
                    wait_until_complete_value: Some(
                        api::message::tool_call::run_shell_command::WaitUntilCompleteValue::WaitUntilComplete(true),
                    ),
                    risk_category: 0,
                },
            ))
        }
        "read_files" => {
            let files: Vec<api::message::tool_call::read_files::File> = args
                .get("paths")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_str())
                .map(|p| api::message::tool_call::read_files::File {
                    name: p.to_string(),
                    line_ranges: vec![],
                })
                .collect();
            Some(api::message::tool_call::Tool::ReadFiles(
                api::message::tool_call::ReadFiles { files },
            ))
        }
        "grep" => {
            let queries: Vec<String> = args
                .get("queries")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            let path = args
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Some(api::message::tool_call::Tool::Grep(
                api::message::tool_call::Grep { queries, path },
            ))
        }
        "file_glob" => {
            let patterns: Vec<String> = args
                .get("patterns")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            let search_dir = args
                .get("search_dir")
                .and_then(Value::as_str)
                .map(String::from);
            Some(api::message::tool_call::Tool::FileGlobV2(
                api::message::tool_call::FileGlobV2 {
                    patterns,
                    search_dir: search_dir.unwrap_or_default(),
                    min_depth: 0,
                    max_depth: 0,
                    max_matches: 0,
                },
            ))
        }
        _ => None,
    };

    tool.map(|tool| api::message::ToolCall {
        tool_call_id,
        tool: Some(tool),
    })
}

/// Human-readable description of a tool call for inclusion in the model's
/// chat history (turn 2+). Used when we don't have a structured
/// OpenAI-format tool_calls array to re-send.
pub fn tool_call_summary(tc: &api::message::ToolCall) -> String {
    match &tc.tool {
        Some(api::message::tool_call::Tool::RunShellCommand(c)) => {
            format!("run_shell_command: {}", c.command)
        }
        Some(api::message::tool_call::Tool::ReadFiles(r)) => {
            let paths: Vec<&str> = r.files.iter().map(|f| f.name.as_str()).collect();
            format!("read_files: {}", paths.join(", "))
        }
        Some(api::message::tool_call::Tool::Grep(g)) => {
            format!("grep {:?} in {}", g.queries, g.path)
        }
        Some(api::message::tool_call::Tool::FileGlobV2(g)) => {
            format!("file_glob {:?} in {}", g.patterns, g.search_dir)
        }
        _ => "<tool call>".to_string(),
    }
}