# Token Efficiency & Memory Rules for tradV2

1. **Locate Before Reading**:
   - Files > 150 lines (especially `trad.py`, `config.py`, `AGENTS.md`) MUST be searched using `grep_search` first.
   - Always pass `StartLine` and `EndLine` to `view_file` to read at most 50-80 lines at a time. Never dump entire large files into context.

2. **Compute, Don't Read**:
   - For large JSON data or outcome logs (`.data/realized_outcomes.json`, `risk_state.json`), run a quick scratch script to print summary statistics instead of loading raw data into context.

3. **Session Memory First**:
   - When resuming or starting work, read `.agents/memory/ACTIVE_CONTEXT.md` first to understand current invariants and task state in under 50 lines.
