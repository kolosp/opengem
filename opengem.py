"""OpenAI-compatible proxy for Google Gemini. No auth: for private LAN use only.

Endpoints: POST /v1/chat/completions, /v1/completions, /v1/responses; GET /v1/models
Function calling is supported on chat completions and responses.

Env:
  GEMINI_API_KEY   required
  GEMINI_MODEL     default model if the client sends none (default gemini-2.5-flash)
  HOST / PORT      bind address (default 0.0.0.0:8000)
"""
import asyncio
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
# Transient upstream failures (503 high demand, 429, 5xx, network errors, empty replies) are retried
# with backoff so multi-step agents don't lose progress. Give up after RETRY_MAX_SECONDS.
RETRY_MAX = float(os.environ.get("RETRY_MAX_SECONDS", "900"))
RETRY_EMPTY = int(os.environ.get("RETRY_EMPTY", "3"))
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
HDR = {"x-goog-api-key": API_KEY}

app = FastAPI(title="opengem")
# Browser clients (e.g. a web UI on :3000) send a CORS preflight for the Authorization header.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))

FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
          "RECITATION": "content_filter", "BLOCKLIST": "content_filter",
          "PROHIBITED_CONTENT": "content_filter", "SPII": "content_filter"}

# Gemini 3 requires the thoughtSignature of a function call to be echoed back with the history.
# Clients only echo OpenAI fields, so remember signatures by call id (bounded) and fall back to
# Google's documented bypass value for unknown calls.
SIGS: dict[str, str] = {}
DUMMY_SIG = "skip_thought_signature_validator"


def error(status: int, message: str, typ: str = "api_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": typ, "code": status}}, status_code=status)


# ---------- OpenAI -> Gemini ----------

def to_parts(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    parts = []
    for p in content:
        t = p.get("type")
        if t in ("text", "input_text", "output_text"):
            parts.append({"text": p["text"]})
        elif t in ("image_url", "input_image"):
            u = p.get("image_url")
            url = u["url"] if isinstance(u, dict) else u
            if url.startswith("data:"):
                meta, data = url[5:].split(",", 1)
                mime = meta.split(";")[0]
                if ";base64" not in meta:
                    data = base64.b64encode(data.encode()).decode()
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
            else:
                parts.append({"fileData": {"fileUri": url}})
    return parts


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else json.dumps(content)


def gemini_tools(body: dict) -> dict:
    """OpenAI tools (chat nested or responses flat shape) -> Gemini tools/toolConfig."""
    decls = []
    for t in body.get("tools") or []:
        if t.get("type") != "function":
            continue  # web_search, custom, etc. are not supported
        f = t.get("function", t)
        d: dict[str, Any] = {"name": f["name"]}
        if f.get("description"):
            d["description"] = f["description"]
        if f.get("parameters"):
            d["parametersJsonSchema"] = f["parameters"]
        decls.append(d)
    if not decls:
        return {}
    out: dict[str, Any] = {"tools": [{"functionDeclarations": decls}]}
    tc, cfg = body.get("tool_choice"), None
    if tc == "none":
        cfg = {"mode": "NONE"}
    elif tc == "required":
        cfg = {"mode": "ANY"}
    elif isinstance(tc, dict):
        name = tc.get("name") or (tc.get("function") or {}).get("name")
        if name:
            cfg = {"mode": "ANY", "allowedFunctionNames": [name]}
    if cfg:
        out["toolConfig"] = {"functionCallingConfig": cfg}
    return out


def build_request(body: dict, messages: list[dict]) -> dict:
    system, contents, names = [], [], {}
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            system += to_parts(m.get("content"))
            continue
        if role == "tool":
            name = names.get(m.get("tool_call_id")) or m.get("name") or "tool"
            parts = [{"functionResponse": {"name": name,
                                           "response": {"output": text_of(m.get("content"))}}}]
            g_role = "user"
        else:
            parts = to_parts(m.get("content"))
            for tc in m.get("tool_calls") or []:
                f = tc["function"]
                names[tc["id"]] = f["name"]
                try:
                    args = json.loads(f.get("arguments") or "{}")
                except ValueError:
                    args = {}
                part: dict[str, Any] = {"functionCall": {"name": f["name"], "args": args}}
                sig = SIGS.get(tc["id"]) or (DUMMY_SIG if not any("functionCall" in p for p in parts) else None)
                if sig:
                    part["thoughtSignature"] = sig
                parts.append(part)
            g_role = "model" if role == "assistant" else "user"
        if not parts:
            continue
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

    req: dict[str, Any] = {"contents": contents, "generationConfig": cfg, **gemini_tools(body)}
    if system:
        req["systemInstruction"] = {"parts": system}
    return req


# ---------- Gemini -> OpenAI ----------

def cand_items(c: dict) -> list[tuple]:
    """Candidate parts -> [("text", str)] / [("call", call_id, name, args)] in order."""
    items: list[tuple] = []
    for p in c.get("content", {}).get("parts", []):
        if "functionCall" in p:
            fc = p["functionCall"]
            cid = f"call_{uuid.uuid4().hex[:24]}"
            if p.get("thoughtSignature"):
                SIGS[cid] = p["thoughtSignature"]
                while len(SIGS) > 2000:
                    SIGS.pop(next(iter(SIGS)))
            items.append(("call", cid, fc["name"], fc.get("args") or {}))
        elif p.get("text") and not p.get("thought"):
            items.append(("text", p["text"]))
    return items


def cand_text(c: dict) -> str:
    return "".join(i[1] for i in cand_items(c) if i[0] == "text")


def usage(d: dict) -> dict:
    u = d.get("usageMetadata", {})
    return {"prompt_tokens": u.get("promptTokenCount", 0),
            "completion_tokens": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0),
            "total_tokens": u.get("totalTokenCount", 0)}


def finish(c: dict, calls: bool = False) -> str | None:
    r = c.get("finishReason")
    if not r:
        return None
    return "tool_calls" if calls else FINISH.get(r, "stop")


def oa_call(cid: str, name: str, args: dict, index: int | None = None) -> dict:
    o = {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    if index is not None:
        o["index"] = index
    return o


def chat_response(d: dict, model: str) -> dict:
    choices = []
    for i, c in enumerate(d.get("candidates", [])):
        items = cand_items(c)
        calls = [oa_call(*it[1:]) for it in items if it[0] == "call"]
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(it[1] for it in items if it[0] == "text") or None}
        if calls:
            msg["tool_calls"] = calls
        choices.append({"index": i, "message": msg, "finish_reason": finish(c, bool(calls)) or "stop"})
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": model, "choices": choices, "usage": usage(d)}


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
    ncalls: dict[int, int] = {}  # tool call count per choice

    def chunk(choices: list, extra: dict | None = None) -> str:
        o = {"id": cid, "object": "chat.completion.chunk" if chat else "text_completion",
             "created": created, "model": model, "choices": choices, **(extra or {})}
        return f"data: {json.dumps(o)}\n\n"

    last_usage: dict = {}
    try:
        if chat:
            yield chunk([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
        async for kind, d in gemini_stream(model, req):
            if kind == "wait":
                yield ": keepalive\n\n"
                continue
            if "usageMetadata" in d:
                last_usage = usage(d)
            for i, c in enumerate(d.get("candidates", [])):
                idx = c.get("index", i)
                text, calls = "", []
                for it in cand_items(c):
                    if it[0] == "text":
                        text += it[1]
                    else:
                        n = ncalls.get(idx, 0)
                        ncalls[idx] = n + 1
                        calls.append(oa_call(*it[1:], index=n))
                fin = finish(c, ncalls.get(idx, 0) > 0)
                if not (text or calls or fin):
                    continue
                if chat:
                    delta: dict[str, Any] = {"content": text} if text else {}
                    if calls:
                        delta["tool_calls"] = calls
                    yield chunk([{"index": idx, "delta": delta, "finish_reason": fin}])
                else:
                    yield chunk([{"index": idx, "text": text, "finish_reason": fin}])
    except UpstreamError as e:
        yield f"data: {json.dumps({'error': {'message': e.message, 'code': e.status}})}\n\n"
        yield "data: [DONE]\n\n"
        return
    yield chunk([], {"usage": last_usage} if last_usage else None)
    yield "data: [DONE]\n\n"


async def handle(body: dict, messages: list[dict], chat: bool):
    model = (body.get("model") or DEFAULT_MODEL).removeprefix("models/")
    req = build_request(body, messages)
    if not req["contents"]:
        return error(400, "no usable message content", "invalid_request_error")
    if body.get("stream"):
        return StreamingResponse(stream(model, req, chat), media_type="text/event-stream")
    try:
        d = await gemini_call(model, req)
    except UpstreamError as e:
        return error(e.status, e.message)
    return JSONResponse((chat_response if chat else text_response)(d, model))


class UpstreamError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def err_message(text: str) -> str:
    try:
        return json.loads(text)["error"]["message"]
    except Exception:
        return text


def has_output(d: dict) -> bool:
    return any(p.get("functionCall") or (p.get("text") and not p.get("thought"))
               for c in d.get("candidates", []) for p in c.get("content", {}).get("parts", []))


def final_empty(d: dict) -> bool:
    """Empty because of a block/length limit: retrying would not help."""
    return bool(d.get("promptFeedback", {}).get("blockReason")) or any(
        c.get("finishReason") in ("MAX_TOKENS", *[k for k, v in FINISH.items() if v == "content_filter"])
        for c in d.get("candidates", []))


def log_retry(reason: str, wait: float) -> None:
    print(f"[opengem] upstream {reason}, retrying in {wait:.0f}s", flush=True)


async def gemini_call(model: str, req: dict) -> dict:
    start, delay, empties = time.monotonic(), 1.0, 0
    while True:
        try:
            r = await client.post(f"{BASE}/models/{model}:generateContent", json=req, headers=HDR)
        except httpx.TransportError as e:
            if time.monotonic() - start >= RETRY_MAX:
                raise UpstreamError(502, f"upstream connection failed: {e!r}")
            wait, reason = delay, type(e).__name__
        else:
            if r.status_code == 200:
                d = r.json()
                if has_output(d) or final_empty(d) or empties >= RETRY_EMPTY:
                    return d
                empties += 1
                wait, reason = 1.0, "empty response"
            elif r.status_code in RETRY_STATUS and time.monotonic() - start < RETRY_MAX:
                wait, reason = delay, f"http {r.status_code}"
            else:
                raise UpstreamError(r.status_code, err_message(r.text))
        log_retry(reason, wait)
        await asyncio.sleep(wait)
        delay = min(delay * 1.7, 20)


async def gemini_stream(model: str, req: dict) -> AsyncIterator[tuple]:
    """Yields ("data", chunk) and ("wait", None) keepalive ticks while retrying.
    Retries only happen before any output was forwarded, so nothing is duplicated."""
    start, delay, empties = time.monotonic(), 1.0, 0
    url = f"{BASE}/models/{model}:streamGenerateContent?alt=sse"
    while True:
        wait, reason, got, held, blocked = 0.0, "", False, [], False
        try:
            async with client.stream("POST", url, json=req, headers=HDR) as r:
                if r.status_code != 200:
                    msg = err_message((await r.aread()).decode(errors="replace"))
                    if r.status_code not in RETRY_STATUS or time.monotonic() - start >= RETRY_MAX:
                        raise UpstreamError(r.status_code, msg)
                    wait, reason = delay, f"http {r.status_code}"
                else:
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        d = json.loads(line[5:])
                        blocked = blocked or final_empty(d)
                        got = got or has_output(d)
                        if got:
                            for h in held:
                                yield ("data", h)
                            held = []
                            yield ("data", d)
                        else:
                            held.append(d)  # metadata-only chunks: hold until real output shows up
                    if got:
                        return
                    if blocked or empties >= RETRY_EMPTY:
                        for h in held:
                            yield ("data", h)
                        return
                    empties += 1
                    wait, reason = 1.0, "empty response"
        except httpx.TransportError as e:
            if got or time.monotonic() - start >= RETRY_MAX:
                raise UpstreamError(502, f"upstream connection failed: {e!r}")
            wait, reason = delay, type(e).__name__
        log_retry(reason, wait)
        end = time.monotonic() + wait
        while time.monotonic() < end:
            yield ("wait", None)
            await asyncio.sleep(min(5, max(0.0, end - time.monotonic())))
        delay = min(delay * 1.7, 20)


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
    """Responses `input` items -> chat-style messages (incl. assistant tool_calls / tool results)."""
    msgs = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    inp = body.get("input", "")
    if isinstance(inp, str):
        return msgs + [{"role": "user", "content": inp}]
    for item in inp:
        t = item.get("type", "message")
        if t == "message":
            msgs.append({"role": item.get("role", "user"), "content": item.get("content")})
        elif t == "function_call":
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": item["call_id"], "type": "function",
                 "function": {"name": item["name"], "arguments": item.get("arguments") or "{}"}}]})
        elif t == "function_call_output":
            msgs.append({"role": "tool", "tool_call_id": item["call_id"], "content": item.get("output")})
        # reasoning items and other types are dropped
    return msgs


def resp_obj(rid: str, model: str, created: int, status: str, output: list, u: dict | None = None) -> dict:
    o: dict[str, Any] = {"id": rid, "object": "response", "created_at": created, "status": status,
                         "model": model, "output": output}
    if u:
        o["usage"] = {"input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"],
                      "total_tokens": u["total_tokens"]}
    return o


def msg_item(mid: str, text: str, status: str = "completed") -> dict:
    return {"type": "message", "id": mid, "status": status, "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}] if status == "completed" else []}


def call_item(fid: str, cid: str, name: str, args: str, status: str = "completed") -> dict:
    return {"type": "function_call", "id": fid, "call_id": cid, "name": name,
            "arguments": args, "status": status}


def responses_output(items: list[tuple]) -> list[dict]:
    out, text = [], ""
    for it in items:
        if it[0] == "text":
            text += it[1]
        else:
            if text:
                out.append(msg_item(f"msg_{uuid.uuid4().hex}", text))
                text = ""
            out.append(call_item(f"fc_{uuid.uuid4().hex}", it[1], it[2], json.dumps(it[3])))
    if text:
        out.append(msg_item(f"msg_{uuid.uuid4().hex}", text))
    return out


async def responses_stream(model: str, req: dict) -> AsyncIterator[str]:
    rid, created = f"resp_{uuid.uuid4().hex}", int(time.time())
    seq = 0
    output: list[Any] = []
    msg: dict | None = None  # currently open message: {id, idx, text}
    u: dict = {}

    def ev(t: str, **kw) -> str:
        nonlocal seq
        seq += 1
        return f"event: {t}\ndata: {json.dumps({'type': t, 'sequence_number': seq, **kw})}\n\n"

    def start_msg() -> list[str]:
        nonlocal msg
        msg = {"id": f"msg_{uuid.uuid4().hex}", "idx": len(output), "text": ""}
        output.append(None)
        return [ev("response.output_item.added", output_index=msg["idx"],
                   item=msg_item(msg["id"], "", "in_progress")),
                ev("response.content_part.added", item_id=msg["id"], output_index=msg["idx"],
                   content_index=0, part={"type": "output_text", "text": "", "annotations": []})]

    def end_msg() -> list[str]:
        nonlocal msg
        if not msg:
            return []
        m, msg = msg, None
        item = msg_item(m["id"], m["text"])
        output[m["idx"]] = item
        return [ev("response.output_text.done", item_id=m["id"], output_index=m["idx"],
                   content_index=0, text=m["text"]),
                ev("response.content_part.done", item_id=m["id"], output_index=m["idx"],
                   content_index=0, part=item["content"][0]),
                ev("response.output_item.done", output_index=m["idx"], item=item)]

    def call_events(cid: str, name: str, args: dict) -> list[str]:
        idx, fid, a = len(output), f"fc_{uuid.uuid4().hex}", json.dumps(args)
        item = call_item(fid, cid, name, a)
        output.append(item)
        return [ev("response.output_item.added", output_index=idx, item={**item, "arguments": "", "status": "in_progress"}),
                ev("response.function_call_arguments.delta", item_id=fid, output_index=idx, delta=a),
                ev("response.function_call_arguments.done", item_id=fid, output_index=idx, name=name, arguments=a),
                ev("response.output_item.done", output_index=idx, item=item)]

    yield ev("response.created", response=resp_obj(rid, model, created, "in_progress", []))
    yield ev("response.in_progress", response=resp_obj(rid, model, created, "in_progress", []))
    try:
        async for kind, d in gemini_stream(model, req):
            if kind == "wait":
                yield ": keepalive\n\n"
                continue
            if "usageMetadata" in d:
                u = usage(d)
            for c in d.get("candidates", [])[:1]:
                for it in cand_items(c):
                    if it[0] == "text":
                        if not msg:
                            for e in start_msg():
                                yield e
                        msg["text"] += it[1]
                        yield ev("response.output_text.delta", item_id=msg["id"], output_index=msg["idx"],
                                 content_index=0, delta=it[1])
                    else:
                        for e in end_msg() + call_events(it[1], it[2], it[3]):
                            yield e
    except UpstreamError as e:
        yield ev("error", code=str(e.status), message=e.message, param=None)
        return
    for e in end_msg():
        yield e
    yield ev("response.completed", response=resp_obj(rid, model, created, "completed", output, u))


@app.post("/v1/responses")
async def responses(request: Request):
    body = await request.json()
    model = (body.get("model") or DEFAULT_MODEL).removeprefix("models/")
    req = build_request({**body, "max_tokens": body.get("max_output_tokens")}, responses_messages(body))
    if not req["contents"]:
        return error(400, "no usable input", "invalid_request_error")
    if body.get("stream"):
        return StreamingResponse(responses_stream(model, req), media_type="text/event-stream")
    try:
        d = await gemini_call(model, req)
    except UpstreamError as e:
        return error(e.status, e.message)
    cands = d.get("candidates", [])[:1]
    return JSONResponse(resp_obj(f"resp_{uuid.uuid4().hex}", model, int(time.time()), "completed",
                                 responses_output(cand_items(cands[0]) if cands else []), usage(d)))


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
