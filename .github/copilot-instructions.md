# Copilot Instructions

The repository (`gpu-task-executor`) currently contains no source code, build tooling, or tests — only `AGENTS.md`. Update this file with build/test/lint commands and architecture once code is added (the directory name suggests a Rust project).

## Workflow conventions (from AGENTS.md)

- Do not overengineer.
- Never commit or push; leave committing to the user.
- After every task, generate a commit message (max 10 lines) and show it to the user. Prefix it with the current branch name exactly as-is.
- Create or update tests for relevant changes.
- If credentials (username, password) are needed, stop and ask the user.
- `AGENTS.md` has an "Additional agent config" section marked "ask user for confirmation before apply" — confirm with the user before relying on it as binding.
