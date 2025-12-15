"""
Minimal Falcon webhook example with `on_post` to accept Dify chatflow payloads.

Features:
- Simple header-based API key check (header `x-dify-signature`).
- Accepts application/json payloads and simple newline/SSE-style streaming bodies.
- Appends normalized messages to a JSONL file `dify_incoming.jsonl` (in repo root).
- Provides a small WSGI runner for quick local testing; for production use gunicorn/uvicorn.

Usage (PowerShell):
    python .\difyAPI\falcon_webhook.py

Or run via gunicorn for production:
    pip install gunicorn
    gunicorn -b 0.0.0.0:8000 difyAPI.falcon_webhook:app

注意：该示例为最小可运行示例，生产环境建议替换为 Redis/Celery 等可靠队列并校验 HMAC 签名。
"""
import os
import json
import logging
from typing import Optional

import falcon

logger = logging.getLogger("dify_falcon")
logging.basicConfig(level=logging.INFO)

# Persist path (one directory up from this file)
PERSIST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dify_incoming.jsonl"))
EXPECTED_API_KEY = os.environ.get("DIFY_WEBHOOK_API_KEY", "changeme")
API_KEY_HEADER = "x-dify-signature"


def append_jsonl(obj):
    os.makedirs(os.path.dirname(PERSIST_PATH), exist_ok=True)
    with open(PERSIST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class WebhookResource:
    """Falcon resource handling POSTs from Dify.

    The resource accepts either a full JSON payload (Content-Type: application/json)
    or a text/stream body containing newline-delimited JSON or SSE-like `data:` lines.
    """

    def on_post(self, req: falcon.Request, resp: falcon.Response):
        # Basic API key check
        header_val = req.get_header(API_KEY_HEADER) or ""
        if header_val != EXPECTED_API_KEY:
            raise falcon.HTTPForbidden(title="Forbidden", description="Invalid API key")

        content_type = (req.get_header("content-type") or "").lower()

        # Try JSON first
        if "application/json" in content_type:
            try:
                # Falcon provides req.media in modern versions
                try:
                    payload = req.media
                except Exception:
                    # fallback to reading body
                    body = req.stream.read()
                    payload = json.loads(body.decode("utf-8")) if body else None
            except Exception as e:
                logger.exception("Failed to parse JSON body: %s", e)
                raise falcon.HTTPBadRequest(title="Bad Request", description="Invalid JSON")

            items = []
            if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
                for m in payload["messages"]:
                    items.append({
                        "conversation_id": payload.get("conversation_id"),
                        "user": payload.get("user"),
                        "message": m,
                    })
            else:
                items.append(payload)

            for it in items:
                append_jsonl(it)

            resp.status = falcon.HTTP_202
            resp.media = {"status": "accepted", "count": len(items)}
            return

        # Otherwise treat body as a stream of lines (SSE or newline-delimited JSON)
        # Read the raw body (be cautious of very large bodies in production)
        try:
            raw = req.stream.read()
            if not raw:
                resp.status = falcon.HTTP_204
                return
            text = raw.decode("utf-8")
        except Exception as e:
            logger.exception("Failed to read request body: %s", e)
            raise falcon.HTTPBadRequest(title="Bad Request", description="Unable to read body")

        accepted = 0
        for line in text.splitlines():
            if not line:
                continue
            if line.startswith("data:"):
                payload_text = line[len("data:"):].strip()
            else:
                payload_text = line.strip()

            if not payload_text:
                continue

            try:
                obj = json.loads(payload_text)
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON stream line: %r", payload_text[:200])
                continue

            append_jsonl({"stream": True, "message": obj})
            accepted += 1

        resp.status = falcon.HTTP_202
        resp.media = {"status": "accepted", "count": accepted}


# Falcon app and route
app = falcon.App()
app.add_route("/webhook", WebhookResource())


if __name__ == "__main__":
    # Simple WSGI server for testing
    from wsgiref import simple_server

    host = "0.0.0.0"
    port = 8000
    logger.info(f"Starting Falcon webhook server on http://{host}:{port}")
    httpd = simple_server.make_server(host, port, app)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
