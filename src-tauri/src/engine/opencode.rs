use super::{
    command_for_binary, images, safe_prompt_arg, BuiltCommand, Engine,
    EngineEvent, SendRequest,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

pub struct OpenCodeEngine;

impl Engine for OpenCodeEngine {
    fn id(&self) -> &'static str {
        "opencode"
    }

    fn supports_images(&self) -> bool {
        // build_command attaches each image as `run --file <abs path>`;
        // that IS the opencode image transport.
        true
    }

    fn supported_permissions(&self) -> &'static [&'static str] {
        // Bin-agnostic floor. One-shot `run` cannot ask mid-turn ("manual"
        // out); "auto" (no flag) lets the build agent run under its own
        // config; "plan" selects the read-only plan agent. "bypass"
        // (`run --auto`) only exists on newer builds, so it is advertised
        // per binary by [`permissions_for`] instead of here — see that
        // function for the version gate.
        &["auto", "plan"]
    }

    fn build_command(&self, req: &SendRequest, bin: &str) -> Result<BuiltCommand, String> {
        let mut cmd = command_for_binary(bin);
        cmd.arg("run");
        cmd.arg("--format");
        cmd.arg("json");
        // `plan` selects the read-only plan agent; `bypass` maps to
        // `run --auto` (auto-approve everything not explicitly denied), but
        // only on builds that advertise it — older ones exit on an unknown
        // argument, so a selected bypass honestly degrades to auto there.
        match req.permission.as_deref() {
            Some("plan") => {
                cmd.arg("--agent");
                cmd.arg("plan");
            }
            Some("bypass") if auto_flag_cached(bin) => {
                cmd.arg("--auto");
            }
            _ => {}
        }
        if let Some(model) = req.model.as_deref() {
            cmd.arg("--model");
            cmd.arg(model);
        }
        if let Some(session_id) = req.session_id.as_deref() {
            cmd.arg("--session");
            cmd.arg(session_id);
        }
        for image in &req.images {
            if let Some(path) = images::absolutize_image_path(image, &req.workspace) {
                cmd.arg("--file");
                cmd.arg(path);
            }
        }
        // Keep the message positional (no `--` separator: opencode 1.1.x
        // mis-parses it in run mode); safe_prompt_arg guards a leading '-'.
        cmd.arg(safe_prompt_arg(&req.prompt));
        Ok(BuiltCommand {
            command: cmd,
            stdin_payload: None,
            keep_stdin_open: false,
            cleanup_files: Vec::new(),
            preassigned_session_id: None,
        })
    }

    fn parse_line(&self, line: &str, out: &mut Vec<EngineEvent>) {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            return;
        };
        if let Some(session_id) = extract_session_id(&value) {
            out.push(EngineEvent::SessionId(session_id));
        }
        let Some(event_type) = value.get("type").and_then(Value::as_str) else {
            return;
        };
        match event_type {
            "text" | "content_delta" => {
                if let Some(text) = extract_text_delta(&value) {
                    out.push(EngineEvent::Delta(text));
                }
            }
            "reasoning_delta" => {
                if let Some(text) = extract_text_delta(&value) {
                    out.push(EngineEvent::Thinking(text));
                }
            }
            "text_delta" | "output_text_delta" | "assistant_message_delta" | "message_delta"
            | "assistant_message" | "message" => {
                if let Some(text) =
                    extract_text_delta(&value).or_else(|| extract_text_from_message(&value))
                {
                    out.push(EngineEvent::Delta(text));
                }
            }
            "tool_use" => {
                let part = value.get("part");
                let state = part.and_then(|p| p.get("state"));
                let status = first_non_empty_str(&[
                    value.get("status").and_then(Value::as_str),
                    state.and_then(|s| s.get("status")).and_then(Value::as_str),
                    part.and_then(|p| p.get("status")).and_then(Value::as_str),
                ])
                .unwrap_or("started")
                .to_ascii_lowercase();
                let name = first_non_empty_str(&[
                    value.get("name").and_then(Value::as_str),
                    value.get("tool_name").and_then(Value::as_str),
                    part.and_then(|p| p.get("name")).and_then(Value::as_str),
                    part.and_then(|p| p.get("tool_name")).and_then(Value::as_str),
                    part.and_then(|p| p.get("tool")).and_then(Value::as_str),
                    state.and_then(|s| s.get("name")).and_then(Value::as_str),
                ])
                .unwrap_or("tool");
                let input = value
                    .get("input")
                    .cloned()
                    .or_else(|| part.and_then(|p| p.get("input")).cloned())
                    .or_else(|| state.and_then(|s| s.get("input")).cloned());
                if is_terminal_tool_status(&status) {
                    let output = value
                        .get("output")
                        .or_else(|| value.get("result"))
                        .cloned()
                        .or_else(|| part.and_then(|p| p.get("output")).cloned())
                        .or_else(|| state.and_then(|s| s.get("output")).cloned());
                    let error = first_non_empty_str(&[
                        value.get("error").and_then(Value::as_str),
                        part.and_then(|p| p.get("error")).and_then(Value::as_str),
                        state.and_then(|s| s.get("error")).and_then(Value::as_str),
                    ]);
                    // pi convention: the pending event opened the row, the
                    // terminal event patches its result onto it.
                    let result = match (output, error) {
                        (Some(output), _) => output,
                        (None, Some(error)) => {
                            serde_json::json!({ "text": error, "isError": true })
                        }
                        (None, None) => Value::Null,
                    };
                    out.push(super::tool_result_patch(name, Some(&result)));
                } else {
                    out.push(super::tool_call_message(name, input.as_ref()));
                }
            }
            "step_finish" => {
                let reason = value
                    .get("reason")
                    .or_else(|| value.get("part").and_then(|p| p.get("reason")))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if matches!(reason, "stop" | "complete" | "completed" | "done") {
                    out.push(EngineEvent::Done {
                        session_id: None,
                        usage: None,
                    });
                } else if let Some(tokens) =
                    value.get("part").and_then(|p| p.get("tokens")).filter(|t| !t.is_null())
                {
                    out.push(EngineEvent::Usage(tokens.clone()));
                }
            }
            "turn_complete" | "turn_completed" | "turn_done" | "done" | "completed" => {
                out.push(EngineEvent::Done {
                    session_id: None,
                    usage: None,
                });
            }
            "error" => {
                let message = extract_error_message(&value)
                    .unwrap_or_else(|| "Unknown OpenCode error".to_string());
                out.push(EngineEvent::Error(message));
            }
            _ => {}
        }
    }
}

/// Every opencode NDJSON event carries the session id somewhere near the
/// top (spelling varies: `sessionID` / `sessionId` / `session_id`); walk
/// for the first concrete value. "pending" is a placeholder, not an id.
fn extract_session_id(event: &Value) -> Option<String> {
    fn find(node: &Value) -> Option<String> {
        match node {
            Value::Object(map) => {
                for key in ["session_id", "sessionId", "sessionID"] {
                    if let Some(raw) = map.get(key).and_then(Value::as_str) {
                        let trimmed = raw.trim();
                        if !trimmed.is_empty() && trimmed != "pending" {
                            return Some(trimmed.to_string());
                        }
                    }
                }
                map.values().find_map(find)
            }
            Value::Array(items) => items.iter().find_map(find),
            _ => None,
        }
    }
    find(event)
}

fn first_non_empty_str<'a>(candidates: &[Option<&'a str>]) -> Option<&'a str> {
    candidates
        .iter()
        .flatten()
        .map(|text| text.trim())
        .find(|text| !text.is_empty())
}

fn extract_text_delta(event: &Value) -> Option<String> {
    let part = event.get("part");
    let text = first_non_empty_str(&[
        event.get("delta").and_then(Value::as_str),
        event.get("text").and_then(Value::as_str),
        part.and_then(|p| p.get("delta")).and_then(Value::as_str),
        part.and_then(|p| p.get("text")).and_then(Value::as_str),
        part.and_then(|p| p.get("content")).and_then(Value::as_str),
    ])?;
    Some(text.to_string())
}

fn extract_text_from_message(event: &Value) -> Option<String> {
    let text = first_non_empty_str(&[
        event.get("message").and_then(|m| m.get("text")).and_then(Value::as_str),
        event.get("message").and_then(|m| m.get("content")).and_then(Value::as_str),
        event.get("output").and_then(|o| o.get("text")).and_then(Value::as_str),
        event.get("result").and_then(|r| r.get("text")).and_then(Value::as_str),
    ]);
    if let Some(text) = text {
        return Some(text.to_string());
    }
    let parts = event
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)?;
    let mut merged = String::new();
    for part in parts {
        if let Some(segment) = first_non_empty_str(&[
            part.get("text").and_then(Value::as_str),
            part.get("delta").and_then(Value::as_str),
            part.get("content").and_then(Value::as_str),
        ]) {
            merged.push_str(segment);
        }
    }
    let merged = merged.trim();
    (!merged.is_empty()).then(|| merged.to_string())
}

fn is_terminal_tool_status(status: &str) -> bool {
    status.contains("complete")
        || status.contains("success")
        || status.contains("done")
        || status.contains("fail")
        || status.contains("error")
}

fn extract_error_message(event: &Value) -> Option<String> {
    let nested = event.get("error");
    let message = nested
        .and_then(Value::as_str)
        .or_else(|| event.get("message").and_then(Value::as_str))
        .or_else(|| {
            nested.and_then(|error| {
                error.get("message").and_then(Value::as_str).or_else(|| {
                    error
                        .get("data")
                        .and_then(|data| data.get("message"))
                        .and_then(Value::as_str)
                })
            })
        })
        .map(str::trim)
        .filter(|text| !text.is_empty())?;
    Some(message.to_string())
}

/// `run --auto` support per resolved binary, probed once per app run. Only
/// successes are cached: a failed probe (CLI missing, help flake) must not
/// keep hiding the mode after the user upgrades mid-session.
static AUTO_SUPPORT: LazyLock<Mutex<HashMap<String, bool>>> =
    LazyLock::new(Default::default);

/// Modes this binary actually honors: `bypass` (`run --auto`) is offered only
/// when the CLI advertises the flag. `list_engines` uses this for the
/// composer picker so an unsupported mode is greyed rather than promised;
/// older builds reject the flag, which is the whole reason for the gate.
pub(crate) async fn permissions_for(bin: &str) -> Vec<String> {
    let mut modes = vec!["auto".to_string(), "plan".to_string()];
    if auto_flag_available(bin).await {
        modes.push("bypass".to_string());
    }
    modes
}

/// Async capability probe: spawns `run --help` on a cache miss. Only a probe
/// that observed `--auto` counts as support; everything else is conservative.
async fn auto_flag_available(bin: &str) -> bool {
    if let Some(supported) = auto_flag_cached_opt(bin) {
        return supported;
    }
    if !probe_auto_flag(bin).await {
        return false;
    }
    if let Ok(mut cache) = AUTO_SUPPORT.lock() {
        cache.insert(bin.to_string(), true);
    }
    true
}

/// Sync guard for `build_command`: a binary whose probe has not succeeded is
/// treated as unsupported, so a stale pick can never send an unknown flag.
fn auto_flag_cached(bin: &str) -> bool {
    auto_flag_cached_opt(bin).unwrap_or(false)
}

fn auto_flag_cached_opt(bin: &str) -> Option<bool> {
    AUTO_SUPPORT.lock().ok().and_then(|cache| cache.get(bin).copied())
}

/// Reads `opencode run --help` (stdout+stderr); the short budget keeps a hung
/// CLI from stalling engine listing, and any failure counts as unsupported.
async fn probe_auto_flag(bin: &str) -> bool {
    let mut cmd = command_for_binary(bin);
    cmd.args(["run", "--help"]);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);
    #[cfg(windows)]
    super::hide_console(&mut cmd);
    let Ok(Ok(output)) =
        tokio::time::timeout(std::time::Duration::from_secs(5), cmd.output()).await
    else {
        return false;
    };
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    help_advertises_auto(&text)
}

fn help_advertises_auto(help: &str) -> bool {
    help.contains("--auto")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn req() -> SendRequest {
        SendRequest {
            session_id: None,
            provider_id: None,
            workspace: PathBuf::from("/tmp/ws"),
            prompt: "hi".to_string(),
            images: Vec::new(),
            model: None,
            effort: None,
            service_tier: None,
            permission: None,
            additional_dirs: Vec::new(),
        }
    }

    fn argv(req: &SendRequest) -> Vec<String> {
        argv_bin(req, "opencode")
    }

    fn argv_bin(req: &SendRequest, bin: &str) -> Vec<String> {
        let built = OpenCodeEngine.build_command(req, bin).unwrap();
        built
            .command
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect()
    }

    fn events(line: &str) -> Vec<EngineEvent> {
        let mut out = Vec::new();
        OpenCodeEngine.parse_line(line, &mut out);
        out
    }

    #[test]
    fn run_argv_is_json_positional_prompt() {
        let args = argv(&req());
        assert_eq!(args[..3], ["run", "--format", "json"]);
        assert_eq!(args.last().map(String::as_str), Some("hi"));
        // A leading dash must never reach the CLI as a flag.
        let mut request = req();
        request.prompt = "-list".to_string();
        let args = argv(&request);
        assert_eq!(args.last().map(String::as_str), Some(" -list"));
    }

    #[test]
    fn plan_model_and_resume_flags() {
        let mut request = req();
        request.permission = Some("plan".to_string());
        request.model = Some("anthropic/claude-sonnet-5".to_string());
        request.session_id = Some("ses_123".to_string());
        let args = argv(&request);
        assert!(args.windows(2).any(|w| w == ["--agent", "plan"]));
        assert!(args
            .windows(2)
            .any(|w| w == ["--model", "anthropic/claude-sonnet-5"]));
        assert!(args.windows(2).any(|w| w == ["--session", "ses_123"]));
    }

    #[test]
    fn bypass_maps_to_auto_only_after_a_successful_probe() {
        let mut request = req();
        request.permission = Some("bypass".to_string());
        // Unprobed binary: `--auto` is not sent (an older CLI would exit on
        // the unknown argument), so bypass degrades to auto, and no read-only
        // plan agent is selected either.
        let args = argv_bin(&request, "opencode-unprobed");
        assert!(!args.contains(&"--auto".to_string()));
        assert!(!args.contains(&"--agent".to_string()));
        // A successful probe (what `list_engines` runs) unlocks the flag.
        AUTO_SUPPORT
            .lock()
            .unwrap()
            .insert("opencode-supported".to_string(), true);
        let args = argv_bin(&request, "opencode-supported");
        assert!(args.contains(&"--auto".to_string()));
        assert!(!args.contains(&"--agent".to_string()));
    }

    #[test]
    fn help_probe_requires_the_auto_flag() {
        assert!(help_advertises_auto(
            "FLAGS\n  --auto  Auto-approve permissions that are not explicitly denied"
        ));
        assert!(!help_advertises_auto("FLAGS\n  --agent string  Agent to use"));
    }

    #[test]
    fn text_event_maps_to_delta_and_session() {
        let out = events(
            r#"{"type":"text","sessionID":"ses_1","part":{"text":"hello"}}"#,
        );
        assert!(out
            .iter()
            .any(|e| matches!(e, EngineEvent::SessionId(id) if id == "ses_1")));
        assert!(out
            .iter()
            .any(|e| matches!(e, EngineEvent::Delta(text) if text == "hello")));
    }

    #[test]
    fn pending_session_placeholder_is_not_an_id() {
        let out = events(r#"{"type":"text","sessionID":"pending","part":{"text":"hi"}}"#);
        assert!(!out.iter().any(|e| matches!(e, EngineEvent::SessionId(_))));
    }

    #[test]
    fn nested_session_id_is_found() {
        let out = events(
            r#"{"type":"turn_update","parts":[{"meta":{"sessionId":"ses_nested"}}]}"#,
        );
        assert!(out
            .iter()
            .any(|e| matches!(e, EngineEvent::SessionId(id) if id == "ses_nested")));
    }

    #[test]
    fn tool_use_streams_start_then_result_patch() {
        let start = events(
            r#"{"type":"tool_use","sessionID":"ses_1","part":{"tool":"read","state":{"status":"pending","input":{"filePath":"a.rs"}}}}"#,
        );
        assert!(start.iter().any(|e| matches!(
            e,
            EngineEvent::Message { role, patch: false, .. } if role == "tool"
        )));
        let done = events(
            r#"{"type":"tool_use","part":{"tool":"read","state":{"status":"completed","output":"fn main(){}"}}}"#,
        );
        assert!(done.iter().any(|e| matches!(
            e,
            EngineEvent::Message { role, patch: true, result: Some(_), .. } if role == "tool"
        )));
        let failed = events(
            r#"{"type":"tool_use","part":{"tool":"bash","state":{"status":"error","error":"exit 1"}}}"#,
        );
        assert!(failed.iter().any(|e| matches!(
            e,
            EngineEvent::Message { patch: true, result: Some(r), .. }
                if r.get("isError").and_then(Value::as_bool) == Some(true)
        )));
    }

    #[test]
    fn step_finish_stop_completes_turn() {
        let out = events(r#"{"type":"step_finish","part":{"reason":"stop"}}"#);
        assert!(out.iter().any(|e| matches!(e, EngineEvent::Done { .. })));
        let usage = events(
            r#"{"type":"step_finish","part":{"reason":"tool-calls","tokens":{"input":10,"output":4}}}"#,
        );
        assert!(usage.iter().any(|e| matches!(e, EngineEvent::Usage(_))));
        assert!(!usage.iter().any(|e| matches!(e, EngineEvent::Done { .. })));
    }

    #[test]
    fn error_event_supports_nested_message() {
        let out = events(
            r#"{"type":"error","error":{"data":{"message":"provider rate limited"}}}"#,
        );
        assert!(out
            .iter()
            .any(|e| matches!(e, EngineEvent::Error(m) if m == "provider rate limited")));
    }
}
