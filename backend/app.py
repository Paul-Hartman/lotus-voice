"""lotus-voice Flask application - AI Voice Generation Hub."""

from api import ancient_bp, audiobook_bp, backends_bp, health_bp, synthesis_bp, voices_bp
from config import FLASK_DEBUG, FLASK_PORT
from flask import Flask
from flask_cors import CORS


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(synthesis_bp)
    app.register_blueprint(audiobook_bp)
    app.register_blueprint(voices_bp)
    app.register_blueprint(backends_bp)
    app.register_blueprint(ancient_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"lotus-voice starting on port {FLASK_PORT}")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=FLASK_DEBUG)
