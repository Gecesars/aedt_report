"""
Compatibility entrypoint that reuses the main factory from `aedt_explorer`.

This keeps legacy invocations of `python app.py` working while ensuring there is
only one application factory (`aedt_explorer.create_app`) for the project.
"""

from __future__ import annotations

import os

from aedt_explorer import create_app


# Expose the shared factory for WSGI servers if needed.
app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
