"""
FastAPI webhook server for receiving Dify chatflow data (blocking POST and streaming/chunked).

- POST /webhook
  Accepts JSON payloads (blocking mode). Writes each message to a JSONL file and enqueues for processing.

- POST /stream
  Accepts chunked/SSE-like streaming input. Parses incoming chunks and enqueues messages.

Notes:
- This module writes incoming messages to `dify_incoming.jsonl` (append).
- A background worker consumes the in-memory queue and persists messages. Replace with Redis/RQ/Celery for production.
- Shows how to convert incoming dict -> `ace.Sample` if you want to call `OfflineAdapter.run` later.
- Security: verify an HMAC or API key in headers (simple header check implemented).

Run with:
    uvicorn difyAPI.webhook_server:app --reload

"""
from fastapi import FastAPI, Request, Header, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict
import asyncio
import json
import os
import logging

# Optional: import ace Sample if you plan to directly convert to training samples
try:
    from ace import Sample
except Exception:
    Sample = None

logger = logging.getLogger("dify_webhook")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Simple in-memory queue; swap for Redis in production
queue: asyncio.Queue = asyncio.Queue()

# Where incoming messages are persisted for later batch processing
PERSIST_PATH = os.path.join(os.path.dirname(__file__), "..", "dify_incoming.jsonl")
PERSIST_PATH = os.path.abspath(PERSIST_PATH)

# Shared secret header for a very simple authentication example
EXPECTED_API_KEY = os.environ.get("DIFY_WEBHOOK_API_KEY", "changeme")
API_KEY_HEADER = "x-dify-signature"


class WebhookPayload(BaseModel):
    # Adjust fields to Dify's actual webhook shape. This is a flexible example.
    conversation_id: Optional[str]
    user: Optional[str]
    messages: Optional[Any]
    # allow arbitrary extra fields
    class Config:
        extra = "allow"


async def persist_worker():
    """Background worker that persists queue items to JSONL.
    In production, this worker would push items into a durable queue or directly
    call a trainer/adapter in a controlled, rate-limited way.
    """
    os.makedirs(os.path.dirname(PERSIST_PATH), exist_ok=True)
    logger.info(f"Persist worker started, writing to {PERSIST_PATH}")
    while True:
        item = await queue.get()
        try:
            # item is expected to be a dict-serializable object
            with open(PERSIST_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info("Persisted incoming item")
            # Optionally, convert to ace.Sample and hand to training adapter here
            # if Sample is available and you have an adapter instance, you could do:
            # sample = Sample(question=item.get('question', ''), ground_truth=item.get('answer', None))
            # adapter.run([sample], ...)
        except Exception as e:
            logger.exception("Failed to persist item: %s", e)
        finally:
            queue.task_done()


@app.on_event("startup")
async def startup_event():
    # Start the background persist worker
    app.state.persist_task = asyncio.create_task(persist_worker())


def verify_api_key(header_value: Optional[str]):
    if header_value != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.post("/webhook")
async def webhook(payload: WebhookPayload, x_dify_signature: Optional[str] = Header(None)):
    """Receive blocking JSON POSTs from Dify (webhook style).

    Example payload shape (adjust to Dify):
    {
      "conversation_id": "abc",
      "user": "userid",
      "messages": [ {"role": "user", "text": "Hello"}, ... ]
    }
    """
    verify_api_key(x_dify_signature)

    data = payload.dict()
    # canonicalize: extract messages as list of dicts if present
    incoming_items = []
    if isinstance(data.get("messages"), list):
        for m in data["messages"]:
            # attach context metadata
            item = {
                "conversation_id": data.get("conversation_id"),
                "user": data.get("user"),
                "message": m,
            }
            incoming_items.append(item)
    else:
        # not a list: persist the whole payload as single item
        incoming_items.append(data)

    # enqueue all items for persistence/processing
    for it in incoming_items:
        await queue.put(it)

    return {"status": "accepted", "count": len(incoming_items)}


@app.post("/stream")
async def stream_endpoint(request: Request, x_dify_signature: Optional[str] = Header(None)):
    """Accept chunked or SSE-like streaming requests.

    This endpoint iterates the request body stream and tries to decode newline-delimited JSON
    or SSE-style lines that start with `data:`. For each decoded JSON object we enqueue it.
    """
    verify_api_key(x_dify_signature)

    # Starlette Request.stream() yields bytes chunks as they arrive
    async for chunk in request.stream():
        if not chunk:
            continue
        try:
            text = chunk.decode("utf-8")
        except Exception:
            # ignore non-text chunks
            continue

        # split into lines (handle multiple JSON objects in one chunk)
        for line in text.splitlines():
            if not line:
                continue
            # SSE-style: lines may start with 'data:'
            if line.startswith("data:"):
                payload_text = line[len("data:"):].strip()
            else:
                payload_text = line.strip()

            if not payload_text:
                continue

            try:
                obj = json.loads(payload_text)
            except json.JSONDecodeError:
                # Could be part of a multi-chunk JSON; for simplicity skip invalid JSON.
                logger.warning("Received non-JSON stream chunk: %r", payload_text[:200])
                continue

            # enqueue object with metadata
            await queue.put({"stream": True, "message": obj})

    return {"status": "stream_received"}


# Helper: convert persisted JSONL into ace.Sample objects (if `ace.Sample` is available)
def load_persisted_samples(path: Optional[str] = None):
    p = path or PERSIST_PATH
    samples = []
    if Sample is None:
        logger.warning("ace.Sample not available in this environment; returning raw dicts")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Try to extract question/answer pairs from obj. This depends on the shape of your stream.
            message = obj.get("message") or obj
            text = None
            if isinstance(message, dict):
                # common case: message has role/text or content
                text = message.get("text") or message.get("content") or str(message)
            else:
                text = str(message)

            # If you have ground-truth labels, adapt appropriately. Here we'll use a placeholder.
            samples.append(Sample(question=text, ground_truth=None, context={"meta": {"raw": message}}))
    return samples


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("difyAPI.webhook_server:app", host="0.0.0.0", port=8000, reload=True)
