# AGENTS.md

Companion code for the post "Running SemIf locally" (rationalstuff.com/technology/running-semif-locally):
an NLI cross-encoder (Qwen3.5 backbone + 3-class head, from `AlexWortega/openjev`) ported to MLX so it
runs natively on Apple Silicon, a Metal port of the upstream eval harness, and a small
email/project triage pipeline built on it. The published MLX weights are `jacksodj/semif-4b-nli-v5-mlx`.

## Setup

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[torch,eval]"
```

After cloning, run `scripts/setup-hooks.sh` once to turn on the secret-scanning git hooks.

Model defaults live in `semif_mlx.py`. The MLX scripts use `DEFAULT_MODEL` (`jacksodj/semif-4b-nli-v5-mlx`),
fetched into the HF cache on first use. The Transformers paths (`compare_mlx_torch.py`, `bench.py torch`)
use `ORIGINAL_MODEL` (`AlexWortega/openjev/qwen3.5-4b-nli-v5`). `resolve()` accepts a local dir, a repo id,
or `owner/repo/subfolder`, and downloads only the subfolder. Local `openjev/` and `dist/` weight dirs are
git-ignored.

## Layout

- `semif_mlx.py` — the whole port. `SemIfCrossEncoder(path_or_repo, bits=None|8|4).predict(pairs)` returns
  `(n, 3)` probabilities in `[contradiction, entailment, neutral]` order. Loads original HF checkpoints
  (sanitizes on load) or pre-converted ones (keys already prefixed `backbone.`).
- `semif/triage.py` — Part 3 pipeline: `build_premise` → `prefilter` → `classify` (rerank + route) and
  `resolve_unit`.
- Root scripts map 1:1 to the post's Appendix B commands: `run_semif_mlx.py`, `compare_mlx_torch.py`,
  `bench.py`, `eval_mlx.py`, `quant_mnli.py`, `sweep_templates.py`, `benchmark_capability.py`.
- `tools/convert_to_mlx.py` and `tools/publish_hf.py` — build and upload the Hugging Face MLX checkpoint;
  `tools/MODEL_CARD.md` is its README.
- `fixtures/` — fictional taxonomy (52 projects, 10 units), 21 labelled messages, people directory.
- `results/` — measured outputs from the machine named in the file.
- `docs/diagnostics.md` — the debugging order for disappointing numbers. `docs/agents/` — issue tracker,
  triage labels and domain-doc conventions for agent sessions.
- `scripts/setup-hooks.sh` turns on the gitleaks git hooks in `.githooks/`; `scripts/issue-field.sh` sets
  an issue's Priority or Horizon. `.github/workflows/pr-checks.yml` is the merge gate.
- `.claude/skills/` — pinned project copies of the planning skills (`skills-lock.json` at the root).

## Invariants — do not break these

- Pin `mlx-lm==0.31.3`. The port uses `backbone.language_model.model` because 0.31.3 has no `Model.model`.
- Pull `score.weight` out **before** `sanitize()`; quantize **after** `update()` + `mx.eval`; quantize the
  backbone only, never the head.
- Pad on the right and pool the last non-pad token (every layer is causal). Truncate the **premise**,
  never the packed string — the hypothesis is last and would be silently lost.
- Input is one packed string via `config["nli_template"]`, not a sentence pair.
- Verify label order empirically (an obvious contradiction must go high on index 0), not from `id2label`.
- MNLI labels are `0=ent, 1=neu, 2=con`; remap with `{0: 1, 1: 2, 2: 0}`.
- Triage hypotheses: a claim about the content (`"The email discusses {name}."`), using the name the
  prefilter matched, with no description. Descriptions and search terms only widen the prefilter.

## Verifying changes

```bash
.venv/bin/python run_semif_mlx.py --smoke           # must print 4/4 and hypothesis kept=True
.venv/bin/python compare_mlx_torch.py               # labels 6/6 vs Transformers on MPS
.venv/bin/python benchmark_capability.py            # fixture pipeline end to end
```

The 4B model needs ~9 GB of unified memory at bf16; run one model-loading script at a time on a 16 GB Mac.

CI (`pr-checks.yml`) can't load the model: it runs on Linux without MLX or the weights. Its `tests` job
shellchecks the hook scripts and byte-compiles every Python file, so run the three commands above
yourself on an Apple Silicon Mac before merging any change to `semif_mlx.py` or `semif/`.

## Conventions

- **The default branch is `master`.** Where the Guardrails below say `main` (or `origin/main`), read
  `master` here.
- **No roadmap-dates workflow.** The shared action behind it can't run from a public repo, so nothing
  sets Start date and Target date here. Set Horizon and Priority by hand and leave the dates unset
  (see `docs/agents/issue-tracker.md`).
- **This repo is public.** Commit nothing private: no account IDs, emails, tokens, personal paths, private
  repo names, or real mail and project data. `fixtures/` stays fictional.
- **The post's commands are an interface.** Each root script is a command in the post's Appendix B, so keep
  script names, flags and defaults working. A change that moves a published number needs the post updated
  too, so flag it rather than slipping it in.
- **Numbers come from runs.** `results/` holds raw script output from the machine the README names. Re-run
  the script to update a file rather than editing it, and keep the README's results tables in step with it.
- **Docs.** The root holds only `README.md`, `AGENTS.md` and `CLAUDE.md`; other docs go under `docs/`.
  `tools/MODEL_CARD.md` is the Hugging Face repo's README and is published by `tools/publish_hf.py`.

## Guardrails

The same controls as Dennis's other repos. Work with them, not around them.

- After cloning, run `scripts/setup-hooks.sh`. It points git at `.githooks/` (gitleaks scans every commit and push) and installs the pinned gitleaks. The Claude Code hook in `.claude/settings.json` runs it automatically before any commit or push.
- For a gitleaks false positive, add a targeted allowlist entry to `.gitleaks.toml`; commits always go through the hooks (no `--no-verify`).
- Work in a worktree. The main checkout stays on `main`, clean, and read-only for sessions. Before your first edit, call `EnterWorktree` with a name for the work: it cuts a branch from `origin/main` under `.claude/worktrees/` (gitignored). Push that branch and open the PR from it; to rename the branch, use `git branch -m`, never `git switch -c`, which leaves an orphaned branch behind. After the merge, GitHub deletes the remote branch, the Code tab's auto-archive removes a session's worktree, and the next session in the repo sweeps the merged local branch; never use `--force`, `-D`, `reset --hard` or `clean -f` to tidy up.
- `main` is protected: the `tests`, `secrets`, and `vulnerabilities` jobs in `.github/workflows/pr-checks.yml` must pass. Push a branch, open a PR, and merge with `gh pr merge --auto --squash`, including tracker and docs edits. No `--admin`, no force-push to `main`.
- The workflow uses no repository secrets and never `pull_request_target`. Keep the three job names stable; add the project's build and test steps to `tests` with the first build work. The vulnerability steps switch on by themselves when Python requirements, Python code, or npm lockfiles appear; add an audit for any other package manager the project adopts.
- OpenTofu/Terraform: CI runs `tofu fmt`, `tofu validate`, and tflint on any `.tf` files. Destroys and state surgery (`destroy`, `-destroy`, `state rm/push`) never run locally; stateful resources carry `prevent_destroy = true`; real infrastructure with state adopts the standard `tofu-deploy-reusable.yml` (plan, destroy-guard, and the `allow-destroy` label that only Dennis adds).
- Anything Dennis must do is a `needs-human` issue with a `**Next:**` line, filed when it comes up rather than left in a reply, and closed or rewritten when the Next goes stale (see `docs/agents/issue-tracker.md`). If the effort spans repos, every repo is linked to the effort's private GitHub Project so the band shows the siblings' items. Give each one a Priority (`scripts/issue-field.sh <n> Priority High`). Decisions go to him one at a time through AskUserQuestion.
- Roadmap data lives on each issue as organization issue fields: set Horizon (`scripts/issue-field.sh <n> Horizon Now`) when charting a map or triaging a report, file tickets with `--parent` and `--blocked-by`, and leave Start date and Target date to the `roadmap-dates` workflow (see `docs/agents/issue-tracker.md`).
