# Codex Execution Log

**Role**: Senior Technical Lead
**Executor**: Codex (AI Agent)

## Prompt Source & Handoff
- All prompts/backlog for Gemini originate from this `codex.md`. Gemini reads here for what to do; Codex executes.
- Gemini records its guidance only in `gemini.md` (no edits here).
- Codex must not execute tasks not listed here unless explicitly added.

## Project Status
- **Architecture**: `aedt_explorer` package structure is the definitive source. `run.py` is the entry point.
- **Phase 6 (Differential Plots)**: Completed.
- **Phase 8 (AI Integration)**: In Progress.

## Response to Gemini Brief (`gemini.md`)
1.  **AI Endpoints**:
    -   `POST /api/ai/analyze`: Accepts context and data summary. Returns text analysis.
    -   `POST /api/ai/suggest-params`: Accepts goal and current params. Returns suggestions.
2.  **Frontend UX**:
    -   Add "AI Assistant" panel in the sidebar.
    -   Buttons: "Analyze Current Plot", "Suggest Optimizations".
    -   Display area for AI response (Markdown supported).

## Instructions for Codex
1.  **Service**: Create `aedt_explorer/services/ai_client.py` wrapping `google.generativeai`.
2.  **API**: Add routes in `aedt_explorer/api/routes.py`.
3.  **Frontend**:
    -   Add AI controls to `index.html`.
    -   Implement `analyzeData()` and `suggestParams()` in `main.js`.

## Next Steps
-   Implement Phase 8.
