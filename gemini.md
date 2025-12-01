# Gemini Lead Brief

You are the lead model (Gemini) guiding the build. Codex executes changes locally. Keep instructions concise and actionable.

## Context Snapshot
- Stack: Flask backend with PyAEDT integration (`aedt_explorer` package), SPA front-end in `templates/home.html` + `static/js/home.js`.
- API implemented: `/api/projects`, `/api/projects/<id>/designs`, reports, S-parameter fetch, 3D image export, and job runner (field animation, parametric sweep) in `aedt_explorer/api/routes.py`.
- PyAEDT glue: `aedt_explorer/services/aedt_client.py` and `services/jobs.py` share a Desktop session and spawn threaded jobs.
- Config: `.env` (not committed) supplies `SECRET_KEY`, `GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-2.5-flash`), `AEDT_PROJECTS_BASE_DIR`, `AEDT_VERSION`. Do **not** check secrets into git.

## Decisions Needed from Gemini (lead)
1) **Phase 6 – Differential plots**  
   - Finalize contract for `/api/compare/sparameters`: request payload shape (two report/config selectors, traces list, frequency alignment strategy), response shape (freq array, paired traces, metadata).  
   - Specify error handling and empty-data policy.
2) **AI endpoints (SPA Step 4)**  
   - Define payload/response for `/api/ai/analyze` (context + summary) and `/api/ai/suggest-params` (parameter name, candidate values, target).  
   - Provide prompt templates and max token/size guidance (avoid sending huge numeric arrays).  
   - Decide whether to stream or return full text; include token usage fields if useful.
3) **Front-end UX**  
   - Describe the UI changes needed in `static/js/home.js` to select two S-parameter configs and to display AI responses (simple text panel vs modal).  
   - Confirm any new routes/buttons to add to the SPA cards (Step 3 and Step 4).
4) **App wiring**  
   - We have two Flask entrypoints (`app.py` using `templates/home.html`; `aedt_explorer/__init__.py` serving `templates/index.html`). Recommend consolidation path if necessary; otherwise, state which to keep as the main app for new endpoints.

## What Codex Will Do After Your Guidance
- Implement the agreed API contracts in `aedt_explorer/api/routes.py` and supporting service modules.  
- Update `static/js/home.js` and templates to match the UX you specify.  
- Add light tests or smoke paths where feasible and keep notes of any caveats/backlog.

## Quick Run Notes (for reference)
```
pip install -r requirements.txt
# env vars: SECRET_KEY, GEMINI_API_KEY, GEMINI_MODEL, AEDT_PROJECTS_BASE_DIR, AEDT_VERSION
python run.py  # entrypoint currently imports create_app from aedt_explorer
```
