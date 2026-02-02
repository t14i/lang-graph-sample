# LangGraph Sample

Verification repository for LangGraph's Tool Calling and HITL (Human-in-the-Loop) features.

## Setup

```bash
# Install dependencies
uv sync

# Copy .env.example to .env and set your API key
cp .env.example .env

# Run scripts
uv run --env-file .env python 01_quickstart.py
```

## Files

| File | Description |
|------|-------------|
| `01_quickstart.py` | Quick Start - Minimal graph |
| `02_tool_definition.py` | Tool definition methods comparison |
| `03_tool_execution.py` | ToolNode behavior, parallel calls |
| `04_tool_error_handling.py` | Error handling, retry pattern |
| `05_hitl_interrupt.py` | HITL basics - interrupt/Command |
| `06_hitl_approve_reject_edit.py` | Approve/Reject/Edit patterns |
| `07_production_considerations.py` | Production considerations |
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

See [REPORT.md](./REPORT.md) for details.
