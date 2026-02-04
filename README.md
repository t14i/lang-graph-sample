# LangGraph Sample

Verification repository for LangGraph's Tool Calling, HITL (Human-in-the-Loop), Durable Execution, and Memory features.

## Setup

```bash
# Install dependencies
uv sync

# Copy .env.example to .env and set your API keys
cp .env.example .env

# Run scripts
uv run --env-file .env python 01_quickstart.py
```

## API Keys Required

| Feature | API Key |
|---------|---------|
| Tool Calling, HITL, Durable | `ANTHROPIC_API_KEY` |
| Memory (embeddings) | `OPENAI_API_KEY` |

## Files

| File | Description |
|------|-------------|
| `01_quickstart.py` | Quick Start - Minimal graph |
| `02_tool_definition.py` | Tool definition methods comparison |
| `03_tool_execution.py` | ToolNode behavior, parallel calls |
| `04_tool_error_handling.py` | Error handling, retry pattern |
| `05_hitl_interrupt.py` | HITL basics - interrupt/Command |
| `06_hitl_approve_reject_edit.py` | Approve/Reject/Edit patterns |
| `07_durable_basic.py` | Durable execution basics |
| `08_durable_hitl.py` | HITL + Durability |
| `09_durable_production.py` | Durable execution production concerns |
| `11_memory_store_basic.py` | Memory Store basic CRUD |
| `12_memory_semantic_search.py` | Semantic search with embeddings |
| `13_memory_cross_thread.py` | Cross-thread memory persistence |
| `14_memory_langmem_tools.py` | LangMem Memory Tools for agents |
| `15_memory_background_extraction.py` | Background memory extraction |
| `16_production_considerations.py` | Overall production considerations |
| `REPORT.md` | Detailed verification report |
| `REPORT_ja.md` | Japanese version of the report |

## Key Findings

### Tool Calling
- `@tool` decorator is simple and clean
- `ToolNode` handles execution and errors automatically
- Parallel tool calls work out of the box
- Retry logic needs custom implementation

### HITL
- `interrupt()` / `Command(resume=...)` API is intuitive
- Approve/Reject/Edit all work well
- Production requires additional infrastructure:
  - Audit logging
  - Timeout handling
  - Notification system
  - Authorization

### Durable Execution
- Checkpoint saved after each node
- Survives process restart (resume from last checkpoint)
- HITL interrupts persist across restarts
- Production concerns:
  - No auto-cleanup of old checkpoints
  - No built-in thread listing API
  - Concurrent execution on same thread_id can cause issues

### Memory
- `InMemoryStore` for development, `PostgresStore` for production
- Namespace-based organization (like folders)
- Semantic search with OpenAI embeddings
- LangMem tools enable agent-managed memory
- Background extraction for automatic fact capture
- Production concerns:
  - Embedding costs
  - Memory cleanup (no TTL)
  - Privacy/PII compliance

See [REPORT.md](./REPORT.md) for details.
