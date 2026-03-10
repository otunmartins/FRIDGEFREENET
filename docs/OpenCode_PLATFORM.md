# OpenCode as the Materials Discovery Platform

**Insulin AI** is **OpenCode for materials discovery** – we use the [OpenCode](https://github.com/anomalyco/opencode) source as our codebase. The agent is built from source; no upstream opencode.ai install.

## Primary Interface: Chat/TUI (Like Claude Code)

**This is the main way to use the platform** – an interactive chat in the terminal.

### 1. Install

```bash
cd /path/to/insulin-ai
./install
```

The install script:
- Installs OpenCode via [opencode.ai](https://opencode.ai) (if not already installed)
- Creates `insulin-ai` launcher in `~/.local/bin` (or `$XDG_BIN_DIR`)

### 2. Run

```bash
insulin-ai
```

Opens the chat/TUI. OpenCode loads `.opencode/` and the **materials-discovery** agent. Type:

> "Discover materials for fridge-free insulin patches"

The agent orchestrates the discovery loop step by step -- mining literature, translating to PSMILES, evaluating via MD, mutating, and iterating. You can steer the agent between steps.

### 3. Alternative: Run Script

```bash
./run   # Launches from this repo (requires prior ./install)
```

## Fallback: Python CLI (No OpenCode Install)

If you don't have OpenCode installed, use the Python CLI:

```bash
python insulin_ai_cli.py discover --iterations 2 --mutate
```

Ollama is optional; without it, keyword-based extraction is used. For full LLM extraction, install [Ollama](https://ollama.ai) and run `ollama pull llama3.2`.

For PaperQA2 deep reading, add PDFs to `papers/` and build the index:
```bash
OPENAI_API_KEY=sk-... ./scripts/index_papers.sh
```

## OpenCode Workflow

1. You say "Discover materials for fridge-free insulin patches"
2. Agent calls `mine_literature` → material names (includes PaperQA2 synthesis when papers indexed)
3. Agent translates names to PSMILES, validates each
4. Agent calls `evaluate_psmiles` → MD feedback
5. Agent calls `mutate_psmiles` → variants of high performers
6. Agent saves state, reports to you, and waits for input
7. You can steer ("focus on PEG", "skip PLGA", "stop") or let it continue
8. Agent refines query and starts next iteration

The **code** is the platform. The terminal is the interface. OpenCode operates on this codebase directly.
