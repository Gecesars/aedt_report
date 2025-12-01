from __future__ import annotations

import os

from flask import Flask, jsonify, render_template


def create_app() -> Flask:
    """Factory para criar a aplicação Flask."""
    app = Flask(__name__)

    @app.get("/health")
    def health() -> tuple[dict[str, str], int]:
        """Endpoint simples de verificação."""
        return {"status": "ok"}, 200

    @app.get("/home")
    @app.get("/home.html")
    def home() -> str:
        """Serve a SPA principal."""
        return render_template("home.html")

    @app.get("/")
    def index():
        # Redireciona a rota raiz para a SPA
        return render_template("home.html")

    return app


if __name__ == "__main__":
    # Permite configurar porta via variável de ambiente se desejado.
    port = int(os.getenv("PORT", "5000"))
    create_app().run(host="0.0.0.0", port=port, debug=False)
