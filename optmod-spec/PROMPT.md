# Claude Code Prompt — optmod

Paste this entire prompt into Claude Code after pointing it at the spec-kit directory.

---

## The Prompt

```
You are building `optmod` — a local OpenAI-compatible routing proxy written in Python 3.11+.

Start by reading CLAUDE.md in full, then read every file in SPEC/ in numeric order (00 through 12) before writing a single line of code.

After reading all spec files:

1. Create the full project directory structure exactly as described in CLAUDE.md.

2. Implement every file in the build order listed in CLAUDE.md section "Build Order". After each group of files, run `pytest tests/` and fix any failures before continuing.

3. Copy `assets/dashboard.html` to `ui/index.html` verbatim — do not modify it.

4. Copy `assets/stats_reference.py` as a reference when implementing `stats.py` — you may adapt it but must preserve the exact response shape defined in SPEC/09_stats.md.

5. After all files are created and tests pass, run this smoke test sequence and show me the output:
   a. `uvicorn main:app --port 8765 &`
   b. `curl -s http://localhost:8765/optmod/status | python3 -m json.tool`
   c. `curl -s http://localhost:8765/api/stats | python3 -m json.tool`
   d. `curl -s -X POST http://localhost:8765/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"optmod","messages":[{"role":"user","content":"summarize this briefly"}]}' | python3 -m json.tool`

6. Show me the final directory tree: `find optmod/ -type f | sort`

Do not install any dependency not listed in pyproject.toml.
Do not create any file not listed in the directory structure.
Do not use Optional[X] — use X | None.
Do not modify assets/dashboard.html.
Never raise inside a router's route() method.
Never modify messages in-place inside a mutator.
```

---

## Before Running Claude Code

Make sure these are ready on your machine:

```bash
# 1. Ollama running with both models
ollama serve &
ollama pull qwen2.5:7b
ollama pull qwen3:8b

# 2. DeepSeek API key set
export DEEPSEEK_API_KEY=your_key_here

# 3. Point Claude Code at this directory
cd /path/to/optmod-spec
claude  # or however you launch Claude Code
```

## After the Build

Test the proxy manually:

```bash
cd optmod/
pip install -e '.[dev]'

# Start the proxy
uvicorn main:app --host 0.0.0.0 --port 8765 --reload

# In another terminal:

# Check status
curl http://localhost:8765/optmod/status

# Check dashboard
open http://localhost:8765/ui

# Test routing
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"optmod","messages":[{"role":"user","content":"why does quicksort fail on sorted input?"}]}'

# Swap router at runtime
curl -X POST http://localhost:8765/optmod/router/passthrough
curl -X POST http://localhost:8765/optmod/router/rule_based

# Check stats
curl "http://localhost:8765/api/stats?range=all"
```

## Configure Hermes to use optmod

Add to `~/.hermes/config.yaml`:

```yaml
provider: custom
model: optmod-router
base_url: http://localhost:8765/v1
api_key: optmod
```

Then start Hermes normally. All LLM calls will route through optmod.

## What's not built yet (Phase 2)

- `routing_policy.pkl` — produced by running WildClawBench (see BenchRouter_PRD_v3_PoC.docx)
- Hermes plugin (thin wrapper calling /optmod/* REST endpoints)
- GLM-4.7-Flash model tier (add to config.yaml when ready)
- Feedback loop calibration CLI (retrains tree from routing.log.jsonl)
