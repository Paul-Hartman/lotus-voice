"""Health check endpoint."""

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "lotus-voice",
        "port": 5031,
    })
