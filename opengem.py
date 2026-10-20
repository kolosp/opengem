"""OpenAI-compatible proxy for Google Gemini. No auth: for private LAN use only.

Endpoints: POST /v1/chat/completions, POST /v1/completions, GET /v1/models

Env:
  GEMINI_API_KEY   required
  GEMINI_MODEL     default model if the client sends none (default gemini-2.5-flash)
  HOST / PORT      bind address (default 0.0.0.0:8000)
"""
import base64
import json
import os
import time
import uuid
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
BASE = "https://generativelanguage.googleapis.com/v1beta"

app = FastAPI(title="opengem")
# Browser clients (e.g. a web UI on :3000) send a CORS preflight for the Authorization header.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))

FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
          "RECITATION": "content_filter", "BLOCKLIST": "content_filter",
          "PROHIBITED_CONTENT": "content_filter", "SPII": "content_filter"}


def error(status: int, message: str, typ: str = "api_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": typ, "code": status}}, status_code=status)


# ---------- OpenAI -> Gemini ----------

def to_parts(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    parts = []
    for p in content:
        if p.get("type") == "text":
            parts.append({"text": p["text"]})
        elif p.get("type") == "image_url":
            url = p["image_url"]["url"]
            if url.startswith("data:"):
                meta, data = url[5:].split(",", 1)
                mime = meta.split(";")[0]
                if ";base64" not in meta:
                    data = base64.b64encode(data.encode()).decode()
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
            else:
                parts.append({"fileData": {"fileUri": url}})
    return parts


def build_request(body: dict, messages: list[dict]) -> dict:
    system, contents = [], []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            system += to_parts(m.get("content"))
            continue
        parts = to_parts(m.get("content"))
        if not parts:
            continue
        g_role = "model" if role == "assistant" else "user"
        if contents and contents[-1]["role"] == g_role:  # Gemini wants alternating roles
            contents[-1]["parts"] += parts
        else:
            contents.append({"role": g_role, "parts": parts})

    cfg: dict[str, Any] = {}
    maps = {"temperature": "temperature", "top_p": "topP", "seed": "seed",
            "presence_penalty": "presencePenalty", "frequency_penalty": "frequencyPenalty",
            "n": "candidateCount"}
    for k, g in maps.items():
        if body.get(k) is not None:
            cfg[g] = body[k]
    mt = body.get("max_completion_tokens") or body.get("max_tokens")
    if mt:
        cfg["maxOutputTokens"] = mt
    stop = body.get("stop")
    if stop:
        cfg["stopSequences"] = [stop] if isinstance(stop, str) else stop
    rf = (body.get("response_format") or {}).get("type")
    if rf in ("json_object", "json_schema"):
        cfg["responseMimeType"] = "application/json"
        schema = (body["response_format"].get("json_schema") or {}).get("schema")
        if schema:
            cfg["responseJsonSchema"] = schema

    req: dict[str, Any] = {"contents": contents, "generationConfig": cfg}
    if system:
        req["systemInstruction"] = {"parts": system}
    return req


# ---------- Gemini -> OpenAI ----------

def cand_text(c: dict) -> str:
    return "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if not p.get("thought"))


def usage(d: dict) -> dict:
    u = d.get("usageMetadata", {})
    return {"prompt_tokens": u.get("promptTokenCount", 0),
            "completion_tokens": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0),
            "total_tokens": u.get("totalTokenCount", 0)}


def finish(c: dict) -> str | None:
    r = c.get("finishReason")
    return FINISH.get(r, "stop") if r else None


def chat_response(d: dict, model: str) -> dict:
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": i, "message": {"role": "assistant", "content": cand_text(c)},
                         "finish_reason": finish(c) or "stop"}
                        for i, c in enumerate(d.get("candidates", []))],
            "usage": usage(d)}


def text_response(d: dict, model: str) -> dict:
    return {"id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": i, "text": cand_text(c), "logprobs": None,
                         "finish_reason": finish(c) or "stop"}
                        for i, c in enumerate(d.get("candidates", []))],
            "usage": usage(d)}


async def stream(model: str, req: dict, chat: bool) -> AsyncIterator[str]:
    cid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(choices: list, extra: dict | None = None) -> str:
        o = {"id": cid, "object": "chat.completion.chunk" if chat else "text_completion",
             "created": created, "model": model, "choices": choices, **(extra or {})}
        return f"data: {json.dumps(o)}\n\n"

    def choice(i: int, text: str | None, fin: str | None) -> dict:
        if chat:
            delta = {"content": text} if text else {}
            return {"index": i, "delta": delta, "finish_reason": fin}
        return {"index": i, "text": text or "", "finish_reason": fin}

    url = f"{BASE}/models/{model}:streamGenerateContent?alt=sse"
    last_usage: dict = {}
    async with client.stream("POST", url, json=req, headers={"x-goog-api-key": API_KEY}) as r:
        if r.status_code != 200:
            msg = (await r.aread()).decode(errors="replace")
            yield f"data: {json.dumps({'error': {'message': msg, 'code': r.status_code}})}\n\n"
            yield "data: [DONE]\n\n"
            return
        if chat:
            yield chunk([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            d = json.loads(line[5:])
            if "usageMetadata" in d:
                last_usage = usage(d)
            for i, c in enumerate(d.get("candidates", [])):
                t, f = cand_text(c), finish(c)
                if t or f:
                    yield chunk([choice(c.get("index", i), t, f)])
    yield chunk([], {"usage": last_usage} if last_usage else None)
    yield "data: [DONE]\n\n"


async def handle(body: dict, messages: list[dict], chat: bool):
    model = (body.get("model") or DEFAULT_MODEL).removeprefix("models/")
    req = build_request(body, messages)
    if not req["contents"]:
        return error(400, "no usable message content", "invalid_request_error")
    if body.get("stream"):
        return StreamingResponse(stream(model, req, chat), media_type="text/event-stream")
    r = await client.post(f"{BASE}/models/{model}:generateContent", json=req,
                          headers={"x-goog-api-key": API_KEY})
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text
        return error(r.status_code, msg)
    d = r.json()
    return JSONResponse((chat_response if chat else text_response)(d, model))


# ---------- routes ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await handle(body, body.get("messages", []), chat=True)


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(map(str, prompt))
    return await handle(body, [{"role": "user", "content": prompt}], chat=False)


# ---------- Responses API (/v1/responses) ----------

def responses_messages(body: dict) -> list[dict]:
    msgs = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    inp = body.get("input", "")
    if isinstance(inp, str):
        return msgs + [{"role": "user", "content": inp}]
    for item in inp:
        if item.get("type", "message") != "message":
            continue  # function_call / reasoning items etc. are not supported
        c = item.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if p.get("type") in ("input_text", "output_text", "text"):
                    parts.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "input_image":
                    u = p.get("image_url")
                    parts.append({"type": "image_url", "image_url": {"url": u["url"] if isinstance(u, dict) else u}})
            c = parts
        msgs.append({"role": item.get("role", "user"), "content": c})
    return msgs


def resp_obj(rid: str, model: str, created: int, status: str, text: str = "", u: dict | None = None,
             mid: str = "") -> dict:
    output = []
    if status == "completed":
        output = [{"type": "message", "id": mid, "status": "completed", "role": "assistant",
                   "content": [{"type": "output_text", "text": text, "annotations": []}]}]
    o: dict[str, Any] = {"id": rid, "object": "response", "created_at": created, "status": status,
                         "model": model, "output": output}
    if u:
        o["usage"] = {"input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"],
                      "total_tokens": u["total_tokens"]}
    return o


async def responses_stream(model: str, req: dict) -> AsyncIterator[str]:
    rid, mid, created = f"resp_{uuid.uuid4().hex}", f"msg_{uuid.uuid4().hex}", int(time.time())
    seq = 0

    def ev(t: str, **kw) -> str:
        nonlocal seq
        seq += 1
        return f"event: {t}\ndata: {json.dumps({'type': t, 'sequence_number': seq, **kw})}\n\n"

    yield ev("response.created", response=resp_obj(rid, model, created, "in_progress"))
    yield ev("response.in_progress", response=resp_obj(rid, model, created, "in_progress"))
    yield ev("response.output_item.added", output_index=0,
             item={"id": mid, "type": "message", "status": "in_progress", "role": "assistant", "content": []})
    yield ev("response.content_part.added", item_id=mid, output_index=0, content_index=0,
             part={"type": "output_text", "text": "", "annotations": []})
    full, u = "", {}
    async with client.stream("POST", f"{BASE}/models/{model}:streamGenerateContent?alt=sse",
                             json=req, headers={"x-goog-api-key": API_KEY}) as r:
        if r.status_code != 200:
            msg = (await r.aread()).decode(errors="replace")
            yield ev("error", code=str(r.status_code), message=msg, param=None)
            return
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            d = json.loads(line[5:])
            if "usageMetadata" in d:
                u = usage(d)
            for c in d.get("candidates", []):
                t = cand_text(c)
                if t:
                    full += t
                    yield ev("response.output_text.delta", item_id=mid, output_index=0,
                             content_index=0, delta=t)
    yield ev("response.output_text.done", item_id=mid, output_index=0, content_index=0, text=full)
    yield ev("response.content_part.done", item_id=mid, output_index=0, content_index=0,
             part={"type": "output_text", "text": full, "annotations": []})
    done = resp_obj(rid, model, created, "completed", full, u, mid)
    yield ev("response.output_item.done", output_index=0, item=done["output"][0])
    yield ev("response.completed", response=done)


@app.post("/v1/responses")
async def responses(request: Request):
    body = await request.json()
    model = (body.get("model") or DEFAULT_MODEL).removeprefix("models/")
    chat_body = {**body, "max_tokens": body.get("max_output_tokens")}
    req = build_request(chat_body, responses_messages(body))
    if not req["contents"]:
        return error(400, "no usable input", "invalid_request_error")
    if body.get("stream"):
        return StreamingResponse(responses_stream(model, req), media_type="text/event-stream")
    r = await client.post(f"{BASE}/models/{model}:generateContent", json=req,
                          headers={"x-goog-api-key": API_KEY})
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text
        return error(r.status_code, msg)
    d = r.json()
    text = "".join(cand_text(c) for c in d.get("candidates", [])[:1])
    return JSONResponse(resp_obj(f"resp_{uuid.uuid4().hex}", model, int(time.time()), "completed",
                                 text, usage(d), f"msg_{uuid.uuid4().hex}"))


@app.get("/v1/models")
async def models():
    r = await client.get(f"{BASE}/models?pageSize=200", headers={"x-goog-api-key": API_KEY})
    if r.status_code != 200:
        return error(r.status_code, r.text)
    data = [{"id": m["name"].removeprefix("models/"), "object": "model",
             "created": 0, "owned_by": "google"}
            for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])]
    return {"object": "list", "data": data}


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("Set GEMINI_API_KEY")
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))
