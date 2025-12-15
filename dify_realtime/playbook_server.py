import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response
import json
import base64
from ace import Playbook
import requests

# 将目录直接指向存放 .json 文件的 'playbooks' 文件夹
# NOTE: don't load playbook at import time; load per-request so callers
# can request different task types (e.g. `customer_service`).
PLAYBOOK_DIR = Path("/data/fangsy/jhq/kouqiangACE/agentic-context-engine-main/dify_realtime/playbooks")

PORT = int(os.environ.get("PLAYBOOK_SERVER_PORT", "18579"))

app = Flask(__name__)

# Try to enable CORS if available
try:
    from flask_cors import CORS
    CORS(app)
    app.logger.info("CORS enabled via flask_cors")
except Exception:
    app.logger.info("flask_cors not installed; continuing without CORS")


def _safe_file_path(filename: str) -> Path:
    """Resolve filename under PLAYBOOK_DIR and ensure it doesn't escape the dir."""
    candidate = (PLAYBOOK_DIR / filename).resolve()
    
    # 使用 Flask logger 做调试输出（避免直接 print）
    app.logger.debug(f"Looking for file at: {candidate}")
    
    if PLAYBOOK_DIR.resolve() not in candidate.parents and candidate != PLAYBOOK_DIR.resolve():
        raise ValueError("invalid filename")
    return candidate


@app.route("/playbook/<task_type>", methods=["GET"])
def get_playbook(task_type: str):
    """
    Return the playbook JSON for the given task type.
    访问: /playbook/customer_service 
    将寻找: .../playbooks/customer_service_insights.json
    """

    filename = f"{task_type}_insights.json"
    try:
        path = _safe_file_path(filename)
    except ValueError:
        return jsonify({"error": "invalid file"}), 400

    if not path.exists():
        app.logger.error(f"File not found: {path}") # 记录错误日志
        # Try to give extra diagnostics to help debug 404s
        try:
            listing = [p.name for p in PLAYBOOK_DIR.resolve().iterdir()]
        except Exception as e:
            listing = [f"(error listing playbook dir: {e})"]
        return (
            jsonify({
                "error": "not_found",
                "path_searched": str(path),
                "playbook_dir": str(PLAYBOOK_DIR.resolve()),
                "playbook_dir_listing": listing[:200],
            }),
            404,
        )

    # Use send_from_directory to set correct content-type and streaming
    return send_from_directory(str(PLAYBOOK_DIR.resolve()), filename)


@app.route("/playbooks/<path:filename>", methods=["GET"])
def get_arbitrary_playbook(filename: str):
    """
    Serve arbitrary file under playbooks/ (safe).
    访问: /playbooks/customer_service_insights.json
    """

    try:
        path = _safe_file_path(filename)
    except ValueError:
        return jsonify({"error": "invalid file"}), 400

    if not path.exists() or not path.is_file():
        try:
            listing = [p.name for p in PLAYBOOK_DIR.resolve().iterdir()]
        except Exception as e:
            listing = [f"(error listing playbook dir: {e})"]
        return (
            jsonify({
                "error": "not_found",
                "path_searched": str(path),
                "playbook_dir": str(PLAYBOOK_DIR.resolve()),
                "playbook_dir_listing": listing[:200],
            }),
            404,
        )

    return send_from_directory(str(PLAYBOOK_DIR.resolve()), filename)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "playbook_dir": str(PLAYBOOK_DIR.resolve())})


@app.route("/compose_and_forward/<task_type>", methods=["GET", "POST"]) 
def compose_and_forward(task_type: str):
    """Compose `prompt_text` from a playbook and forward it (POST) to an external URL.

    - Loads `<task_type>_insights.json` from `PLAYBOOK_DIR`.
    - Composes `prompt_text = playbook.as_prompt()`.
    - If `DIFY_FORWARD_URL` env is set, POSTs JSON `{"prompt": prompt_text, "original": <incoming payload>}`
      to that URL and returns the remote response.
    - If `DIFY_FORWARD_URL` is not set, returns the composed prompt for inspection.
    """

    filename = f"{task_type}_insights.json"
    try:
        path = _safe_file_path(filename)
    except ValueError:
        return jsonify({"error": "invalid file"}), 400

    if not path.exists():
        app.logger.error(f"File not found: {path}")
        try:
            listing = [p.name for p in PLAYBOOK_DIR.resolve().iterdir()]
        except Exception as e:
            listing = [f"(error listing playbook dir: {e})"]
        return (
            jsonify({
                "error": "not_found",
                "path_searched": str(path),
                "playbook_dir": str(PLAYBOOK_DIR.resolve()),
                "playbook_dir_listing": listing[:200],
            }),
            404,
        )

    # Load playbook and compose prompt text
    try:
        playbook = Playbook.load_from_file(path)
    except Exception as e:
        app.logger.exception("Failed to load playbook")
        return jsonify({"error": "failed_load", "detail": str(e)}), 500

    prompt_text = playbook.as_prompt()

    # We do not accept/forward the client's question or other fields.
    # Compose only the `as_prompt()` text and forward that to the external service.
    forward_url = os.environ.get("DIFY_FORWARD_URL")
    if not forward_url:
        # 返回非转义的 UTF-8 JSON，并附带 base64 以便客户端校验原始 bytes
        prompt_b64 = base64.b64encode(prompt_text.encode("utf-8")).decode("ascii")
        body = json.dumps({"prompt":prompt_text}, ensure_ascii=False)
        return Response(body, status=200, content_type="application/json; charset=utf-8")

    payload = {"prompt": prompt_text}
    headers = {"Content-Type": "application/json; charset=utf-8"}
    api_key = os.environ.get("DIFY_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        # send UTF-8 bytes without forcing JSON escapes
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        resp = requests.post(forward_url, data=data, headers=headers, timeout=15)
    except Exception as e:
        app.logger.exception("Failed to forward prompt")
        return jsonify({"error": "forward_failed", "detail": str(e)}), 502

    # return the remote response status and body
    try:
        remote_json = resp.json()
    except Exception:
        remote_json = {"text": resp.text}

    return jsonify({"status_code": resp.status_code, "remote": remote_json}), resp.status_code


if __name__ == "__main__":
    app.logger.info(f"Playbook server exposing {PLAYBOOK_DIR.resolve()} on 0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT)
