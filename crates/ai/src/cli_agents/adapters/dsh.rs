//! DeepSeek Harness (`dsh --profile headless`, with the task on stdin).
//!
//! The harness answers one task, streams its reasoning to stderr and prints its last message on
//! stdout, so its output is the answer text, not JSON lines.
//! This needs a headless profile that accepts piped input (verified with 0.2.2-rc.1); older
//! versions without it reject the missing task and must be updated.
//!
//! It has no flag for a tool server the way other agents do: servers come from the profile's patch
//! layers. Our tool server therefore arrives as one generated `--patch` overlay for this run, which
//! the launcher applies after the profile's own layer, so the person's saved profile is untouched.
//! The overlay is written into a folder of the run's own, owner-only on Unix, so a shared temporary
//! folder can't be used to reach it.
//! Its headless profile keeps no session to continue, so a build that stops starts again from the
//! brief (as it does with the plain adapter).
//!
//! A model is chosen here too, though the headless profile takes no `--model`: the list is what the
//! person set up in their desktop Harness profile (`profiles/desktop/cordis.patch.yml` under
//! `$DSH_HOME`), and a chosen model travels in the same overlay, which also turns the saved-key
//! store back on (the headless bundle leaves it off, so a key saved in the app is never seen) and
//! borrows those providers' settings, so a run uses what the desktop app already uses.
//! The overlay lives in the run's own folder, which the runner deletes with the run, so a copy of a
//! key written inline in that profile does not outlive the turn that needed it.

use super::*;

/// The choice that means "pass no model, use the harness's own default". The saved text is the
/// shared `default` marker every agent without a model of its own uses, because Settings and the
/// CLI compare against that one; this names what it means here rather than borrowing another
/// agent's constant.
const OWN_DEFAULT: &str = crate::anthropic::CLAUDE_CODE_DEFAULT;

/// The tool server's id inside the overlay. Distinct from the one a person may have added to their
/// own profile by hand, so the two never replace each other.
const OVERLAY_ID: &str = "mcp-craft-anything-run";

/// The run's server name: not `craft-anything`, which the person's own registration uses, as the
/// harness refuses to start two clients with one name.
const OVERLAY_SERVER: &str = "craft-anything-run";

/// DeepSeek Harness run as a coding agent: one task in, its last message out.
pub struct DeepSeekHarness;

impl AgentAdapter for DeepSeekHarness {
    fn ids(&self) -> &'static [&'static str] {
        &["dsh", "deepseek-harness"]
    }

    /// Keep task text out of Windows batch arguments and their command-line length limit.
    fn prompt_on_stdin(&self) -> bool {
        true
    }

    /// The headless task includes the preamble, with no added or removed whitespace.
    fn stdin_text(&self, preamble: &str, prompt: &str) -> String {
        format!("{preamble}{prompt}")
    }

    fn args(&self, cx: &ArgsCx) -> Vec<String> {
        let mut v = cx.options.args.clone();
        v.extend(["--profile".into(), "headless".into()]);
        // A question about a build may only read the project's files, so it gets no tool server, and
        // no model overlay either: the overlay is a file on this computer, and this adapter can't
        // tell from here whether the run is on another one (an SSH backend's `ssh_host` isn't in
        // `ArgsCx`). A question therefore behaves exactly as before a model could be chosen. It
        // costs nothing today: questions go only to adapters that say they `answers_questions`, and
        // this one doesn't, so the runner stops before it builds these arguments.
        let mcp = (!cx.question).then_some(cx.mcp);
        let model = (!cx.question).then_some(cx.options.cli_model.as_deref()).flatten().filter(|m| !m.trim().is_empty() && *m != OWN_DEFAULT);
        if mcp.is_some() || model.is_some() {
            let desktop = model.map(|_| read_desktop_patch().unwrap_or_default()).unwrap_or_default();
            v.extend(["--patch".into(), run_patch(cx.run_dir, &overlay_yaml(mcp, model, &desktop))]);
        }
        // No task positional means stdin. Unlike `-`, this fails as a missing task on older
        // Harness versions without stdin support, rather than submitting a literal dash.
        v
    }

    fn streams_json(&self) -> bool {
        false
    }

    /// `--help` lists `--profile`, which the plain `dsh <task>` form would need anyway.
    fn required_flags(&self) -> (&'static [&'static str], &'static [&'static str]) {
        (&["--help"], &["--profile"])
    }

    /// Without `--profile headless` the harness boots a whole application instead of answering, so a
    /// failed flag check keeps this adapter rather than falling back to the plain one.
    fn plain_fallback(&self) -> bool {
        false
    }

    fn model_in_use(&self, options: Option<&BackendOptions>) -> Option<String> {
        Some(options.and_then(|options| options.cli_model.clone()).unwrap_or_else(|| "DeepSeek Harness default".into()))
    }

    /// The models set up in the person's desktop Harness profile, as `provider/model`, after the
    /// headless profile's own default and the built-in DeepSeek models.
    fn model_choices(&self, config: &BackendConfig, _options: &BackendOptions) -> Result<Vec<String>, ToolError> {
        let mut models = vec![OWN_DEFAULT.to_string()];
        // A harness on another machine keeps its profile there, out of reach of this list.
        let local = config.ssh_host.as_deref().is_none_or(|h| h.trim().is_empty());
        if local && let Some(text) = read_desktop_patch() {
            models.extend(desktop_models(&text));
        }
        models.extend(BUILT_IN_MODELS.iter().map(|m| m.to_string()));
        let mut seen = std::collections::HashSet::new();
        models.retain(|m| seen.insert(m.clone()));
        Ok(models)
    }

    fn offers_own_default(&self) -> bool {
        true
    }

    fn run_files(&self) -> bool {
        true
    }

    fn check_tool_server(&self, stop: &CancelToken) -> Result<(), ToolError> {
        cli_can_start(craft_anything_platform::self_cli_path().as_deref(), stop)?;
        // The write a run uses, once, before the build starts: a temporary folder that can't take
        // the overlay stops the build here with a fix, not part-way through one. The run makes and
        // removes its own folder; this only proves one can be written in that place.
        let probe = std::env::temp_dir().join(format!("craft-anything-dsh-probe-{}.yml", std::process::id()));
        let written = craft_anything_platform::write_atomic_private(&probe, b"[]\n")
            .map_err(|e| e.with_remediation("Point TMP at a folder that can be written and has space, then start the build again.".to_string()));
        let _ = std::fs::remove_file(&probe);
        written
    }
}

/// The overlay's file name inside a run's folder.
const OVERLAY_FILE: &str = "tool-server.yml";

/// Our overlay for this run, in the run's own folder: the tool server it was promised, and the
/// model it was asked for. The folder belongs to the run and is deleted with it, so an overlay
/// that carries a key inline — a provider's `apiKey` written into the desktop profile — does not
/// outlive the run. A run that got no folder writes none, and the harness is left the tool server
/// it would otherwise not have; [`check_tool_server`](AgentAdapter::check_tool_server) reports the
/// everyday causes first, with a fix.
fn run_patch(dir: Option<&std::path::Path>, yaml: &str) -> String {
    let Some(dir) = dir else { return missing_overlay() };
    let path = dir.join(OVERLAY_FILE);
    match craft_anything_platform::write_atomic_private(&path, yaml.as_bytes()) {
        Ok(()) => path.display().to_string(),
        Err(e) => {
            log::warn!("the DeepSeek Harness tool-server overlay couldn't be written: {}", e.message);
            missing_overlay()
        }
    }
}

/// A path no run wrote anything to.
fn missing_overlay() -> String {
    std::env::temp_dir().join(format!("craft-anything-dsh-{}-missing.yml", std::process::id())).display().to_string()
}

/// Models every Harness ships with, as `provider/model`.
const BUILT_IN_MODELS: &[&str] = &["deepseek-official/deepseek-flash", "deepseek-official/deepseek-v4-pro"];

/// The person's desktop profile patch layer (`$DSH_HOME/profiles/desktop/cordis.patch.yml`), where
/// the desktop app saves the model providers set up in it.
fn read_desktop_patch() -> Option<String> {
    let home = std::env::var_os("DSH_HOME").filter(|h| !h.is_empty()).map(std::path::PathBuf::from).or_else(|| dirs::home_dir().map(|h| h.join(".dsh")))?;
    let text = std::fs::read_to_string(home.join("profiles").join("desktop").join("cordis.patch.yml")).ok()?;
    Some(text.trim_start_matches('\u{feff}').to_string())
}

/// `provider/model` for each model a desktop patch layer names: the providers' model lists, the
/// default model and the models allowed for helpers.
fn desktop_models(patch: &str) -> Vec<String> {
    use yaml_rust2::{Yaml, YamlLoader};
    let Ok(docs) = YamlLoader::load_from_str(patch) else { return Vec::new() };
    let Some(Yaml::Array(entries)) = docs.into_iter().next() else { return Vec::new() };
    let pair = |e: &Yaml| match (e["provider"].as_str(), e["model"].as_str()) {
        (Some(p), Some(m)) if !p.is_empty() && !m.is_empty() => Some(format!("{p}/{m}")),
        _ => None,
    };
    let mut out = Vec::new();
    for e in &entries {
        let id = e["id"].as_str().unwrap_or_default();
        if id.starts_with("llm-") {
            let Some(providers) = e["config"]["providers"].as_hash() else { continue };
            for (name, p) in providers {
                let Some(name) = name.as_str() else { continue };
                for m in p["models"].as_vec().map(Vec::as_slice).unwrap_or_default() {
                    if let Some(model) = m["id"].as_str().or_else(|| m.as_str()) {
                        out.push(format!("{name}/{model}"));
                    }
                }
            }
        } else if id == "agent-default-model" {
            out.extend(pair(&e["config"]));
        } else if id == "subagent-model-selection-settings" {
            out.extend(e["config"]["allowedModels"].as_vec().map(Vec::as_slice).unwrap_or_default().iter().filter_map(pair));
        }
    }
    out
}

/// The desktop layer's provider entries (ids starting `llm-`), as written, so the headless run
/// knows the same providers.
///
/// An entry starts at the list's own indentation, which YAML lets sit anywhere, and runs until the
/// next one: a line at that indentation opening an entry, or anything less indented (a comment, or
/// the end). A deeper line belongs to the entry. Whether an entry is a provider is read from the
/// patch's own parsed entries, as [`desktop_models`] reads it, so the two always agree on what a
/// provider entry is: `name` before `id`, a `-` on a line of its own, any indentation, a
/// column-zero comment above the list and an anchor declared by an earlier entry all behave.
fn provider_entries(patch: &str) -> String {
    let lines: Vec<&str> = patch.lines().map(|l| l.trim_end_matches('\r')).collect();
    // The list's indentation: the least that opens an entry, so a patch indented as a whole still
    // finds its entries. None: no entry at all, so there's nothing to copy.
    let indent = |l: &str| l.len() - l.trim_start_matches(' ').len();
    let Some(base) = lines
        .iter()
        .filter(|l| {
            let t = l.trim_start();
            t == "-" || t.starts_with("- ")
        })
        .map(|l| indent(l))
        .min()
    else {
        return String::new();
    };
    let mut entries: Vec<Vec<&str>> = Vec::new();
    for line in &lines {
        let t = line.trim_start();
        let at_entry = indent(line) == base && (t == "-" || t.starts_with("- "));
        if at_entry {
            entries.push(vec![line]);
        } else if indent(line) > base
            && let Some(last) = entries.last_mut()
        {
            last.push(line);
        }
        // Anything else sits outside the list: a comment above or below it, which belongs to no entry.
    }
    // Written back at this overlay's own indentation (none), since it sits in the same list as the
    // entries we add; the entry's own shape, and its comments, stay as they were. Only the entries
    // are dedented: a line outside the list keeps every character it had.
    let dedent = |l: &str| l.get(base.min(l.len())..).unwrap_or_default().to_string();
    let texts: Vec<String> = entries.iter().map(|e| e.iter().map(|l| dedent(l)).collect::<Vec<_>>().join("\n")).collect();
    // Which entry is a provider is read from the whole patch as it was written, not one entry at a
    // time: an entry may use an anchor an earlier one defined (`{api: &api …}` then `{api: *api}`),
    // and a parser handed that entry alone rejects the unknown anchor, which would silently leave
    // out the very provider its models were listed from. The patch is parsed as given because it is
    // already a valid document: dedenting it would also strip a column-zero comment's `#` when the
    // list is indented, which breaks the parse and takes the anchors with it.
    let ids: Vec<String> = yaml_rust2::YamlLoader::load_from_str(patch)
        .ok()
        .and_then(|docs| {
            docs.first().and_then(yaml_rust2::Yaml::as_vec).map(|items| items.iter().map(|i| i["id"].as_str().unwrap_or_default().to_string()).collect())
        })
        .unwrap_or_default();
    // The text split and the parsed document name the same entries whenever the split is right. When
    // their counts differ the split and the parser disagree, so each entry is read on its own again:
    // that still finds a self-contained one, and never pairs an entry with another's id.
    let aligned = ids.len() == texts.len();
    let is_provider = |i: usize, text: &str| {
        let id = match ids.get(i).filter(|_| aligned) {
            Some(id) => id.clone(),
            None => yaml_rust2::YamlLoader::load_from_str(text)
                .ok()
                .and_then(|docs| {
                    docs.first().and_then(yaml_rust2::Yaml::as_vec).and_then(|items| items.first()).and_then(|item| item["id"].as_str()).map(str::to_string)
                })
                .unwrap_or_default(),
        };
        id.starts_with("llm-")
    };
    // Copied in source order, so an anchor and the entry that declares it stay in that order.
    //
    // An anchor a provider uses may be declared by an entry that isn't one. Copying the provider
    // alone would then leave `*api` with nothing to resolve, and the harness refuses the overlay
    // outright (`found unknown anchor`), while `desktop_models` still offered that model. So an
    // entry declaring an anchor a copied provider uses, without declaring it itself, is copied too.
    // It is a provider in all but name at that point: leaving it out is what breaks the run.
    let providers: Vec<usize> = texts.iter().enumerate().filter(|(i, text)| is_provider(*i, text)).map(|(i, _)| i).collect();
    let missing = |i: usize| -> Vec<String> {
        let Some(text) = texts.get(i) else { return Vec::new() };
        let owned = anchors_declared_in(text);
        aliases_used_in(text).into_iter().filter(|a| !owned.contains(a)).collect()
    };
    let wanted: std::collections::HashSet<String> = providers.iter().flat_map(|i| missing(*i)).collect();
    let keeps = |i: usize| {
        providers.contains(&i) || (!wanted.is_empty() && texts.get(i).is_some_and(|text| anchors_declared_in(text).iter().any(|a| wanted.contains(a))))
    };
    texts.iter().enumerate().filter(|(i, _)| keeps(*i)).map(|(_, text)| text.clone() + "\n").collect()
}

/// The anchors an entry declares (`&api`, `&base`), by name.
fn anchors_declared_in(text: &str) -> Vec<String> {
    name_after(text, '&')
}

/// The names an entry takes values from (`*api`), by name.
fn aliases_used_in(text: &str) -> Vec<String> {
    name_after(text, '*')
}

/// Every name written after `sigil`, in order, without repeats. Scanning the text is enough here
/// because the question is only which entry mentions which name; a name a quoted string happens to
/// contain names an anchor the patch has to declare anyway, so copying that entry stays right.
fn name_after(text: &str, sigil: char) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(sigil) {
        rest = &rest[at + sigil.len_utf8()..];
        let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-').collect();
        if !name.is_empty() && !out.contains(&name) {
            out.push(name.clone());
        }
        rest = &rest[name.len()..];
    }
    out
}

/// The overlay in the harness's patch format: a list of loader entries. It inserts our tool server,
/// and for a chosen model brings in the desktop profile's providers, turns the saved-key store back
/// on (the headless profile leaves it off) and makes the model the default.
fn overlay_yaml(mcp: Option<&McpCommand>, model: Option<&str>, desktop: &str) -> String {
    let q = |s: &str| format!("'{}'", s.replace('\'', "''"));
    let mut out = String::from("# Craft Anything's overlay for this run. Generated; safe to delete.\n");
    if let Some(mcp) = mcp {
        let args: String = mcp.args.iter().map(|a| format!("\n          - {}", q(a))).collect();
        out.push_str(&format!(
            "- insert:\n    \
               - id: {OVERLAY_ID}\n      \
                 name: '@deepseek-ai/dsh-mcp-client'\n      \
                 config:\n        \
                   serverName: {OVERLAY_SERVER}\n        \
                   transport: stdio\n        \
                   command: {}\n        \
                   args:{args}\n        \
                   toolCallTimeoutMs: 300000\n",
            q(&mcp.command)
        ));
    }
    if let Some(model) = model {
        let (provider, name) = model.split_once('/').unwrap_or(("deepseek-official", model));
        out.push_str(&provider_entries(desktop));
        // The headless bundle leaves the key store off ("one-shot jobs use launch-time credentials"),
        // so a key saved in the app is never seen; a chosen model needs it, and only then.
        out.push_str("- id: credentials\n  disabled: false\n");
        out.push_str(&format!(
            "- id: agent-default-model\n  name: '@deepseek-ai/dsh-agent-default-model'\n  config:\n    provider: {}\n    model: {}\n",
            q(provider),
            q(name)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool server command for a run's project.
    fn mcp() -> McpCommand {
        McpCommand {
            command: r"C:\Program Files\Craft Anything\craft-anything-cli.exe".into(),
            args: vec!["mcp".into(), "--project".into(), r"C:\w\p\investigation.cany".into()],
            notes: None,
        }
    }

    /// The arguments for one message, and the server they name, in a run's own folder as the
    /// runner gives it. None: a run that got no folder, so nothing is written.
    fn args_in(options: &BackendOptions, question: bool, run_dir: Option<&std::path::Path>) -> (Vec<String>, McpCommand) {
        let mcp = mcp();
        let cx = ArgsCx {
            options,
            prompt: "make a game",
            preamble: "PRE ",
            mcp: &mcp,
            resume: None,
            builder: !question,
            reads_checked: false,
            question,
            message_file: None,
            run_dir,
        };
        (DeepSeekHarness.args(&cx), mcp.clone())
    }

    /// A folder of this test's own, standing in for the run's.
    fn run_dir(d: &tempfile::TempDir) -> std::path::PathBuf {
        let dir = d.path().join("run");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The arguments for one message, and the server they name, in a folder that stays for the
    /// life of the process: these tests read what was written, and a real run's folder is removed
    /// by the runner (covered by `the_overlay_lives_in_the_runs_folder_and_dies_with_it`).
    fn args(options: &BackendOptions, question: bool) -> (Vec<String>, McpCommand) {
        static KEPT: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
        let d = KEPT.get_or_init(|| tempfile::tempdir().unwrap());
        let dir = d.path().join(format!("run-{}", question as u8));
        std::fs::create_dir_all(&dir).unwrap();
        args_in(options, question, Some(&dir))
    }

    /// Only options go in argv; the task arrives on stdin after the profile starts.
    #[test]
    fn headless_argv_contains_only_profile_and_patch_options() {
        let (a, _) = args(&BackendOptions::default(), false);
        assert_eq!(a[0..2], ["--profile", "headless"], "{a:?}");
        assert_eq!(a.len(), 4, "only the profile and tool-server patch: {a:?}");
        let with_args = BackendOptions { args: vec!["--patch".into(), "extra.yml".into()], ..Default::default() };
        let (a, _) = args(&with_args, false);
        assert_eq!(a[0..4], ["--patch", "extra.yml", "--profile", "headless"], "{a:?}");
        assert_eq!(a.len(), 6, "custom options and the generated patch: {a:?}");
    }

    /// The run's tool server, as an overlay in a folder of its own that only its owner can read.
    #[test]
    fn the_tool_server_arrives_as_a_patch_overlay_in_the_runs_own_folder() {
        let (a, mcp) = args(&BackendOptions::default(), false);
        let i = a.iter().position(|x| x == "--patch").expect("a --patch overlay");
        let path = std::path::PathBuf::from(&a[i + 1]);
        assert!(path.exists(), "{path:?}");
        let written = std::fs::read_to_string(&path).expect("read the overlay");
        assert!(written.contains("- insert:"), "{written}");
        assert!(written.contains(OVERLAY_ID), "{written}");
        assert!(written.contains("serverName: craft-anything-run\n"), "the person's own registration keeps its name: {written}");
        // The server's own arguments, including the run's project, quoted for YAML.
        assert!(written.contains("- 'mcp'"), "{written}");
        assert!(written.contains(&format!("- '{}'", mcp.args[2])), "{written}");
        // A folder of the run's own, so nothing another user prepared in a shared temporary folder
        // is written through.
        let folder = path.parent().expect("a folder of its own").to_path_buf();
        assert_eq!(folder.file_name().and_then(|n| n.to_str()), Some("run-0"), "the run's own folder, as the test gives it: {folder:?}");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let bits = |p: &std::path::Path| std::fs::metadata(p).expect("metadata").permissions().mode() & 0o777;
            assert_eq!(bits(&path), 0o600, "overlay {:o}", bits(&path));
        }
        // Two runs never share a folder: the runner makes one per run, so neither can read or
        // overwrite the other's overlay.
        let (a, b) = (RunDir::make().expect("a folder"), RunDir::make().expect("another folder"));
        assert_ne!(a.0, b.0, "each run gets its own");
    }

    /// A folder that can't take the overlay stops the run, rather than building without the tool
    /// server it was promised.
    #[test]
    fn a_folder_that_cannot_take_the_overlay_leaves_a_path_that_is_not_there() {
        let missing = missing_overlay();
        assert!(!std::path::Path::new(&missing).exists(), "{missing}");
    }

    /// A question may only read, so it gets no tool server.
    #[test]
    fn a_question_gets_no_tool_server() {
        let (a, _) = args(&BackendOptions::default(), true);
        assert!(!a.iter().any(|x| x == "--patch"), "{a:?}");
        assert_eq!(a, ["--profile", "headless"], "{a:?}");
    }

    /// Batch launchers cannot accept line breaks or long tasks in argv; stdin preserves the text.
    #[test]
    fn multiline_and_long_tasks_use_stdin_without_changing_their_bytes() {
        let preamble = "Instructions:\r\n";
        let tasks = ["-question?\r\n中文 %PATH% & ^ \"quoted\"\n\n".to_string(), format!("{}\nlast line\n", "x".repeat(40_831))];
        for id in ["dsh", "deepseek-harness"] {
            let adapter = AgentAdapters::built_in().for_agent(id);
            for task in &tasks {
                for access in [Access::Build, Access::Question] {
                    let a = build_args(adapter.as_ref(), &BackendOptions::default(), task, preamble, &mcp(), None, access);
                    assert!(adapter.prompt_on_stdin(), "{id} must send task text on stdin");
                    assert_eq!(a.len(), if matches!(access, Access::Question) { 2 } else { 4 }, "only options: {a:?}");
                    assert!(!a.iter().any(|arg| arg.contains(task) || arg.contains(preamble)), "{a:?}");
                    assert!(a.iter().map(|arg| arg.len() + 1).sum::<usize>() < 8191, "the task must not grow argv");
                    assert_eq!(adapter.stdin_text(preamble, task), format!("{preamble}{task}"), "{id}");
                }
            }
        }
    }

    /// The agent is registered under both ids, not handed to the plain adapter.
    #[test]
    fn registered_not_the_plain_adapter() {
        assert!(AgentAdapters::built_in().for_agent("dsh").ids().contains(&"dsh"));
        assert!(AgentAdapters::built_in().for_agent("deepseek-harness").ids().contains(&"dsh"));
        assert_eq!(DeepSeekHarness.required_flags(), (&["--help"][..], &["--profile"][..]));
        assert!(!DeepSeekHarness.plain_fallback());
        assert!(!DeepSeekHarness.streams_json());
    }

    /// A path or a flag with a quote in it must not break the overlay's YAML.
    #[test]
    fn quotes_are_escaped_for_yaml() {
        let mcp = McpCommand { command: "cli's.exe".into(), args: vec!["--project".into(), r"C:\it's here".into()], notes: None };
        let y = overlay_yaml(Some(&mcp), None, "");
        assert!(y.contains("command: 'cli''s.exe'"), "{y}");
        assert!(y.contains("- 'C:\\it''s here'"), "{y}");
    }

    const DESKTOP: &str = "# comment\n\
        - id: web\n  config:\n    searchProvider: tavily\n\
        - id: llm-pi-ai\n  name: \"@deepseek-ai/dsh-llm-pi-ai\"\n  config:\n    providers:\n      openrouter:\n        models:\n          - id: meta/muse\n        api: openai-completions\n      claude:\n        # note\n        models:\n          - id: claude-x\n\
        # between\n\
        - id: agent-default-model\n  config:\n    provider: claude\n    model: claude-x\n\
        - id: subagent-model-selection-settings\n  config:\n    allowedModels:\n      - provider: buddy\n        model: flash\n";

    /// Every model the desktop profile names is offered, with its provider.
    #[test]
    fn desktop_models_are_listed_with_their_provider() {
        let models = desktop_models(DESKTOP);
        for m in ["openrouter/meta/muse", "claude/claude-x", "buddy/flash"] {
            assert!(models.iter().any(|x| x == m), "{m} missing from {models:?}");
        }
        assert!(desktop_models("not: [yaml").is_empty());
    }

    /// A chosen model brings the desktop providers and becomes the default, and nothing else of
    /// the desktop profile comes along.
    #[test]
    fn chosen_model_overlay_carries_providers_and_default() {
        let y = overlay_yaml(None, Some("openrouter/meta/muse"), DESKTOP);
        let docs = yaml_rust2::YamlLoader::load_from_str(&y).expect("overlay parses");
        let entries = docs[0].as_vec().expect("a list");
        let ids: Vec<&str> = entries.iter().filter_map(|e| e["id"].as_str()).collect();
        assert!(ids.contains(&"llm-pi-ai") && ids.contains(&"credentials"), "{y}");
        assert!(!ids.contains(&"web") && !ids.contains(&"subagent-model-selection-settings"), "{y}");
        let default = entries.iter().find(|e| e["id"].as_str() == Some("agent-default-model")).expect("default");
        assert_eq!(default["config"]["provider"].as_str(), Some("openrouter"));
        assert_eq!(default["config"]["model"].as_str(), Some("meta/muse"));
    }

    /// The overlay is written into the run's own folder, and it is gone once the run is over: a
    /// copy of a key written inline in the desktop profile must not outlive the turn that needed it.
    #[test]
    fn the_overlay_lives_in_the_runs_folder_and_dies_with_it() {
        let d = tempfile::tempdir().unwrap();
        let dir = run_dir(&d);
        let options = BackendOptions { cli_model: Some("claude/claude-x".into()), ..Default::default() };
        let (a, _) = args_in(&options, false, Some(&dir));
        let i = a.iter().position(|x| x == "--patch").expect("a --patch overlay");
        let path = std::path::PathBuf::from(&a[i + 1]);
        assert!(path.starts_with(&dir), "inside the run's folder: {path:?} vs {dir:?}");
        assert!(path.exists(), "written when the arguments were built: {path:?}");
        // Dropping the guard the runner keeps is what the end of a run does, a stopped run included.
        let guard = RunDir::make().expect("a folder");
        let folder = guard.0.clone();
        assert!(folder.exists());
        drop(guard);
        assert!(!folder.exists(), "the folder and the overlay in it are gone with the run: {folder:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A run that was given no folder writes no file, rather than writing one somewhere it would stay.
    #[test]
    fn without_a_run_folder_no_overlay_is_written() {
        let options = BackendOptions { cli_model: Some("claude/claude-x".into()), ..Default::default() };
        let (a, _) = args_in(&options, false, None);
        let path = std::path::PathBuf::from(&a[a.iter().position(|x| x == "--patch").expect("--patch") + 1]);
        assert!(!path.exists(), "nothing written, and the harness is told where to look: {path:?}");
    }

    /// A chosen model never puts an overlay on a question's command line, so no path written here
    /// can reach a harness that runs on another computer.
    #[test]
    fn a_question_gets_no_model_overlay_even_when_one_is_chosen() {
        let options = BackendOptions { cli_model: Some("deepseek-official/deepseek-v4-pro".into()), ..Default::default() };
        let (argv, _) = args(&options, true);
        assert_eq!(argv, ["--profile", "headless"], "a question is unchanged by a chosen model: {argv:?}");
    }

    /// A provider entry is found however the list is indented, and only the providers are copied.
    #[test]
    fn provider_entries_follow_the_lists_indentation() {
        // The same layer indented as a whole, as a person (or an editor) may leave it.
        let indented: String = DESKTOP.lines().map(|l| format!("  {l}\n")).collect();
        for patch in [DESKTOP.to_string(), indented] {
            let y = overlay_yaml(None, Some("claude/claude-x"), &patch);
            let docs = yaml_rust2::YamlLoader::load_from_str(&y).unwrap_or_else(|e| panic!("overlay parses: {e}\n{patch:?}"));
            let ids: Vec<&str> = docs[0].as_vec().expect("a list").iter().filter_map(|e| e["id"].as_str()).collect();
            assert!(ids.contains(&"llm-pi-ai"), "the provider entry is copied: {y}");
            assert!(!ids.contains(&"web"), "other entries stay behind: {y}");
        }
    }

    /// Which entry is a provider is decided by its own `id` key, so the same entry is copied
    /// whether `name` comes before `id` or the dash stands on a line of its own: the two halves of
    /// this adapter must never disagree about what a provider entry is.
    #[test]
    fn provider_entries_are_chosen_by_their_id_key_not_the_line_order() {
        for provider in [
            // `id` first, sharing the dash, as the desktop app writes it.
            "- id: llm-pi-ai\n  config:\n    providers:\n      claude:\n        models:\n          - id: claude-x\n",
            // `name` before `id`, the order this repo's own patch uses.
            "- name: \"@deepseek-ai/dsh-llm-pi-ai\"\n  id: llm-pi-ai\n  config:\n    providers:\n      claude:\n        models:\n          - id: claude-x\n",
            // The dash alone, so the mapping starts on the next line.
            "-\n  id: llm-pi-ai\n  config:\n    providers:\n      claude:\n        models:\n          - id: claude-x\n",
        ] {
            let patch = format!("{provider}- id: web\n  config:\n    searchProvider: tavily\n");
            let models = desktop_models(&patch);
            assert!(models.iter().any(|m| m == "claude/claude-x"), "listed: {models:?} from {patch:?}");
            let y = overlay_yaml(None, Some("claude/claude-x"), &patch);
            let docs = yaml_rust2::YamlLoader::load_from_str(&y).unwrap_or_else(|e| panic!("overlay parses: {e}\n{y}"));
            let ids: Vec<&str> = docs[0].as_vec().expect("a list").iter().filter_map(|e| e["id"].as_str()).collect();
            assert!(ids.contains(&"llm-pi-ai"), "the provider its models came from is copied too: {y}");
            assert!(!ids.contains(&"web"), "a non-provider entry stays behind: {y}");
        }
    }

    /// An entry may take a value from an anchor an earlier entry declared. A parser handed that
    /// entry alone rejects the unknown anchor, so the copy must read the whole layer: otherwise the
    /// second provider is left out while its models are still offered, and a run choosing one has
    /// no provider to resolve it with.
    #[test]
    fn a_provider_using_an_earlier_entrys_anchor_is_still_copied() {
        let patch = "- id: llm-one\n  name: \"@deepseek-ai/dsh-llm-pi-ai\"\n  config:\n    providers:\n      first: {api: &api openai-completions, baseURL: 'http://127.0.0.1:1/v1', models: [{id: one}]}\n\
                     - id: llm-two\n  name: \"@deepseek-ai/dsh-llm-pi-ai\"\n  config:\n    providers:\n      second: {api: *api, baseURL: 'http://127.0.0.1:2/v1', models: [{id: two}]}\n\
                     - id: web\n  config:\n    searchProvider: tavily\n";
        // Both entries are offered, so choosing either has to bring its provider along.
        let models = desktop_models(patch);
        for m in ["first/one", "second/two"] {
            assert!(models.iter().any(|x| x == m), "{m} is offered: {models:?}");
        }
        let y = overlay_yaml(None, Some("second/two"), patch);
        let docs = yaml_rust2::YamlLoader::load_from_str(&y).unwrap_or_else(|e| panic!("overlay parses: {e}\n{y}"));
        let ids: Vec<&str> = docs[0].as_vec().expect("a list").iter().filter_map(|e| e["id"].as_str()).collect();
        assert!(ids.contains(&"llm-one") && ids.contains(&"llm-two"), "both providers, the anchor's and its user's: {y}");
        assert!(!ids.contains(&"web"), "a non-provider entry stays behind: {y}");
        // Source order, so the anchor is still declared before the entry that uses it.
        let first = ids.iter().position(|id| *id == "llm-one").expect("llm-one");
        let second = ids.iter().position(|id| *id == "llm-two").expect("llm-two");
        assert!(first < second, "the entry declaring the anchor comes first: {y}");
    }

    /// The three awkward shapes at once: the list indented, a column-zero comment above it, and the
    /// second provider taking the first's anchor. The patch is read as written for the ids, so the
    /// comment keeps its `#` and the anchors still resolve; dedenting a line outside the list would
    /// turn `# desktop` into `desktop` and take the whole classification down with it.
    #[test]
    fn an_indented_list_under_a_column_zero_comment_keeps_its_anchors() {
        // Written one line at a time, so the indentation is exactly what the test means: a comment
        // at the margin, then the list two spaces in.
        let patch = [
            "# desktop profile",
            "  - id: llm-one",
            "    name: \"@deepseek-ai/dsh-llm-pi-ai\"",
            "    config:",
            "      providers:",
            "        first: {api: &api openai-completions, models: [{id: one}]}",
            "  - id: llm-two",
            "    name: \"@deepseek-ai/dsh-llm-pi-ai\"",
            "    config:",
            "      providers:",
            "        second: {api: *api, models: [{id: two}]}",
            "  - id: web",
            "    config:",
            "      searchProvider: tavily",
            "",
        ]
        .join("\n");
        let models = desktop_models(&patch);
        for m in ["first/one", "second/two"] {
            assert!(models.iter().any(|x| x == m), "{m} is offered: {models:?}");
        }
        let y = overlay_yaml(None, Some("second/two"), &patch);
        let docs = yaml_rust2::YamlLoader::load_from_str(&y).unwrap_or_else(|e| panic!("overlay parses: {e}\n{y}"));
        let ids: Vec<&str> = docs[0].as_vec().expect("a list").iter().filter_map(|e| e["id"].as_str()).collect();
        assert!(ids.contains(&"llm-one") && ids.contains(&"llm-two"), "both providers survive the comment: {y}");
        assert!(!ids.contains(&"web"), "a non-provider entry stays behind: {y}");
        // Written back at the overlay's own indentation, so the copied entries are siblings.
        let copied = y.split("- id: llm-one").nth(1).expect("the copied provider");
        assert!(copied.lines().next().is_some_and(|l| !l.starts_with(' ')), "not nested under the entry above: {y}");
    }

    /// An anchor a provider uses may be declared by an entry that isn't a provider. Copying the
    /// provider without that entry leaves `*api` undefined, and the harness refuses the overlay
    /// (`found unknown anchor`) while the model was offered. The declaring entry has to come too,
    /// or choosing that model builds an overlay that can't start.
    #[test]
    fn a_provider_using_an_excluded_entrys_anchor_brings_it_along() {
        let patch = [
            "- id: web",
            "  config:",
            "    api: &api openai-completions",
            "    searchProvider: tavily",
            "- id: llm-pi-ai",
            "  name: \"@deepseek-ai/dsh-llm-pi-ai\"",
            "  config:",
            "    providers:",
            "      custom:",
            "        api: *api",
            "        models: [{id: custom-model}]",
            "",
        ]
        .join("\n");
        assert!(desktop_models(&patch).iter().any(|m| m == "custom/custom-model"), "the model is offered");
        let y = overlay_yaml(None, Some("custom/custom-model"), &patch);
        // The overlay has to parse: that is the failure this guards against.
        let docs = yaml_rust2::YamlLoader::load_from_str(&y).unwrap_or_else(|e| panic!("overlay parses: {e}\n{y}"));
        let ids: Vec<&str> = docs[0].as_vec().expect("a list").iter().filter_map(|e| e["id"].as_str()).collect();
        assert!(ids.contains(&"web") && ids.contains(&"llm-pi-ai"), "the declaring entry comes along: {y}");
        // The declaring entry is copied whole rather than line by line: dropping the lines that
        // don't declare the anchor would mean editing another person's YAML entry by guesswork,
        // and a key we cut is a key the harness then misses. Its settings are inert here.
        assert!(!ids.contains(&"agent-default-model") || y.contains("agent-default-model"), "{y}");
        // The anchor is declared before the entry that uses it, or the parser would still refuse it.
        let declared = y.find("&api").expect("the anchor");
        let used = y.find("*api").expect("the alias");
        assert!(declared < used, "the declaration comes first: {y}");
    }
}
