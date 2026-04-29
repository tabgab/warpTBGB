# Warp TBGB — BYOM Free + OpenRouter + Ollama Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Modify Warp to make BYOM available on free plan, add OpenRouter API key in settings, and incorporate Ollama provider support (PR #9386).

**Architecture:** Three independent changes targeting the same codebase: (1) remove plan gating on BYOK by making `is_byo_api_key_enabled()` always return `true`, (2) wire up the already-existing `open_router` API key field into the settings UI, (3) apply the Ollama PR diff and resolve its TODOs.

**Tech Stack:** Rust (cargo workspace), Warp's Entity/ModelContext UI framework, reqwest for Ollama HTTP client.

**Repo:** https://github.com/tabgab/warpTBGB (fork of warpdotdev/warp, branch: `feat/byom-free-openrouter-ollama`)

---

### Task 1: Create feature branch and apply base setup

**Objective:** Branch from upstream master, configure git identity.

**Files:**
- No code changes — git operations only.

**Step 1: Create branch off upstream/master**

```bash
cd /home/gabor/warpTBGB
git fetch upstream --quiet
git checkout -b feat/byom-free-openrouter-ollama upstream/master
git config user.email "gabor.tabi@omnest.com"
git config user.name "Gabor Tabi"
```

**Verification:** `git branch --show-current` shows the new branch, `git log --oneline -3` shows upstream master history.

---

### Task 2: Make BYOM always available (bypass plan gating)

**Objective:** Make `is_byo_api_key_enabled()` always return `true` regardless of plan tier.

**Files:**
- Modify: `app/src/workspaces/user_workspaces.rs` (line ~476)
- Modify: `app/src/workspaces/workspace.rs` (line ~576, line ~133)

**Step 1: Change user_workspaces.rs**

In `UserWorkspaces::is_byo_api_key_enabled()`, change from:
```rust
pub fn is_byo_api_key_enabled(&self) -> bool {
    self.current_workspace()
        .map(|workspace| workspace.is_byo_api_key_enabled())
        .unwrap_or(FeatureFlag::SoloUserByok.is_enabled())
}
```
To:
```rust
pub fn is_byo_api_key_enabled(&self) -> bool {
    // TBGB modification: BYOM is always available, regardless of plan tier.
    true
}
```

This is the SINGLE most critical change. All other checks in the codebase flow through this function. By making it always return `true`:
- Settings page shows API key editors as interactive (not disabled)
- Upgrade CTA is never shown
- `has_any_ai_remaining` passes BYOK check when keys are configured
- Model selection dropdown treats BYOK-keyed models as available

**Step 2: Verify all callers**

No other changes needed. The function's other callers (`llms.rs:32`, `request_usage_model.rs:402`, `ai_page.rs` multiple, `data_source.rs:496`) all use this same gateway.

**Step 3: Commit**

```bash
git add app/src/workspaces/user_workspaces.rs
git commit -m "feat: make BYOM available on all plans (free, pro, enterprise)

Remove plan-tier gating from is_byo_api_key_enabled().
Users can now bring their own API keys regardless of plan."
```

---

### Task 3: Add OpenRouter API key editor to Settings UI

**Objective:** Add a text input field for OpenRouter API key in the AI Settings page.

**Files:**
- Modify: `app/src/settings_view/ai_page.rs` — add editor and rendering

**What already exists:**
- `ApiKeys::open_router: Option<String>` in `crates/ai/src/api_keys.rs:24`
- `ApiKeyManager::set_open_router_key()` in `crates/ai/src/api_keys.rs:90`
- `ApiKeys::has_any_key()` includes `open_router` check at line 32

**What needs to be added:**

In `ApiKeysWidget` struct (around line 6274):
```rust
struct ApiKeysWidget {
    openai_api_key_editor: ViewHandle<EditorView>,
    anthropic_api_key_editor: ViewHandle<EditorView>,
    google_api_key_editor: ViewHandle<EditorView>,
    open_router_api_key_editor: ViewHandle<EditorView>,  // NEW
    can_use_warp_credits_with_byok: SwitchStateHandle,
    upgrade_highlight_index: HighlightedHyperlink,
}
```

In `ApiKeysWidget::new()`:
1. Destructure `open_router: open_router_key` from the ApiKeys clone (add to pattern at line 6290)
2. Add `create_api_key_editor!` call:
```rust
create_api_key_editor!(
    open_router_api_key_editor,
    open_router_key,
    set_open_router_key,
    "sk-or-..."
);
```
3. Add to the return struct:
```rust
Self {
    openai_api_key_editor,
    anthropic_api_key_editor,
    google_api_key_editor,
    open_router_api_key_editor,  // NEW
    can_use_warp_credits_with_byok: Default::default(),
    upgrade_highlight_index: Default::default(),
}
```

In `render_api_keys_section()` (after Google key, before upgrade CTA):
```rust
column.add_child(render_api_key_input(
    appearance,
    "OpenRouter API Key",
    self.open_router_api_key_editor.clone(),
    is_enabled,
    app,
));
```

Update `search_terms()` (line 6571):
```rust
"api keys bring your own byo openai anthropic google openrouter claude gemini gpt"
```

Update comment at line 3247 to include "openrouter":
```rust
"agent mode natural language detection input hint api keys bring your own byo google anthropic openai openrouter"
```

**Step 2: Also update llms.rs to recognize OpenRouter as BYOK provider**

In `is_using_api_key_for_provider()`:
- OpenRouter is currently handled differently (wired through `open_router` field, not through LLMProvider enum)
- The `LLMProvider::Unknown` case already returns false — OpenRouter models likely use this.
- No change needed here — OpenRouter models use the `open_router` key path, not the provider-based path.

**Verification:** `cargo check -p warp` (or the app crate) — should compile.

**Step 3: Commit**

```bash
git add app/src/settings_view/ai_page.rs
git commit -m "feat: add OpenRouter API key input to settings UI"
```

---

### Task 4: Apply PR #9386 — Ollama provider support

**Objective:** Apply the full diff from https://github.com/warpdotdev/warp/pull/9386.

**Files changed:**
- Modify: `app/src/ai/llms.rs` — add `Ollama` variant to `LLMProvider`, update `is_using_api_key_for_provider`
- Modify: `crates/ai/Cargo.toml` — add `reqwest.workspace = true`
- Modify: `crates/ai/src/api_keys.rs` — add `ollama_url` field, `set_ollama_url()`, update `has_any_key()`
- Modify: `crates/ai/src/lib.rs` — add `pub mod ollama_client;`
- Create: `crates/ai/src/ollama_client.rs` — full Ollama API client (440 lines)

**Step 1: Apply the diff**

The PR author has already addressed all 4 review comments:
1. CRITICAL: api_keys_for_request no longer includes ollama_url ✓
2. CRITICAL: context() → OllamaError conversion fixed ✓
3. IMPORTANT: StreamChunk serde uses helper struct ✓
4. IMPORTANT: Streaming parser properly buffers NDJSON lines ✓

Apply using patch tool or manual file edits.

**Step 2: Update BYOK check for Ollama**

After our Task 1 change (is_byo_api_key_enabled always returns true), the existing logic in `is_using_api_key_for_provider` needs updating:

```rust
LLMProvider::Ollama => api_keys.is_some_and(|keys| keys.ollama_url.is_some()),
```

This is already in the PR diff ✓

**Step 3: Commit**

```bash
git add app/src/ai/llms.rs crates/ai/Cargo.toml crates/ai/src/api_keys.rs crates/ai/src/lib.rs crates/ai/src/ollama_client.rs
git commit -m "feat: add Ollama provider support for local model inference

- Add Ollama variant to LLMProvider enum
- Add ollama_url field to ApiKeys for BYOK configuration
- Create ollama_client module with chat and streaming support
- Add reqwest dependency for HTTP client

Incorporated from warpdotdev/warp#9386 by simpletoolsindia"
```

---

### Task 5: Resolve PR TODOs

**Objective:** Fix the two remaining TODOs from PR #9386.

**TODO 1: Add Ollama icon to the icon system**

In `app/src/ai/llms.rs`, the PR has:
```rust
LLMProvider::Ollama => None, // TODO: Add Ollama icon
```

We can either:
a. Find if an Ollama icon already exists in the icon system
b. If not, use a generic llama/robot icon as a placeholder

**TODO 2: Wire up actual model selection for agent mode**

This requires understanding how models are populated from Ollama. The PR provides `list_models()` and `available_model_names()` — these need to be called at startup to populate the model dropdown.

For now, note this as a follow-up task — the core infrastructure (client, provider enum, API key storage) is in place.

---

### Task 6: Build verification

**Objective:** Verify the changes compile correctly.

```bash
cd /home/gabor/warpTBGB
cargo check -p ai 2>&1 | tail -20
```

Expected: Compilation succeeds (or shows only pre-existing warnings).

Note: Full `cargo build` of the entire Warp app requires GPU toolchain and may take 30+ minutes. `cargo check` is sufficient for verification.

---

### Task 7: Push and finalize

**Objective:** Push the branch to the fork and verify.

```bash
git push -u origin feat/byom-free-openrouter-ollama
```

**Verification:** Check https://github.com/tabgab/warpTBGB/tree/feat/byom-free-openrouter-ollama