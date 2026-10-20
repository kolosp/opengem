# opengem

OpenAI-compatible proxy for Google Gemini. **No auth, for private LAN use only.**

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`, `GET /v1/models` (streaming supported, CORS open for browser clients).

## Run the ghcr image

Images are multi-arch (`linux/amd64`, `linux/arm64` — the latter runs natively on Apple Silicon Macs) and are built and pushed by CI on every push to `main` (tags: `latest`, `sha-<short>`, and `X.Y.Z` for `v*` tags).

```bash
docker run -d --name opengem --restart unless-stopped \
  -e GEMINI_API_KEY=your_key \
  -p 8000:8000 \
  ghcr.io/kolosp/opengem:latest
```

If the package is private, log in first with a GitHub token that has `read:packages`:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u <github-user> --password-stdin
```

To expose it only on one interface, bind the port explicitly, e.g. `-p 192.168.1.10:8000:8000`.

docker compose:

```yaml
services:
  opengem:
    image: ghcr.io/kolosp/opengem:latest
    restart: unless-stopped
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      # GEMINI_MODEL: gemini-3.5-flash-lite   # default model when the client sends none
      # GEMINI_POOL: cheap                    # fail over across cheap models on 429/503
      # GEMINI_THINKING: high                 # force thinking level
      # GEMINI_FORCE_MODEL: "1"               # always use GEMINI_MODEL, ignore the client's
    ports:
      - "8000:8000"
```

## Use it

```bash
curl http://<lan-ip>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.5-flash-lite","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

In OpenAI clients set `base_url=http://<lan-ip>:8000/v1` and use any dummy API key.

See [MODELS.md](MODELS.md) for the available model ids.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | required | Gemini API key |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model used when the request has none |
| `GEMINI_FORCE_MODEL` | off | `1`/`true` = always use `GEMINI_MODEL`, ignoring the model the client sends |
| `GEMINI_THINKING` | unset | `minimal`/`low`/`medium`/`high`: force this thinking level on every request (Gemini 2.5 models get an equivalent token budget). Unset = Gemini's own default |
| `GEMINI_POOL` | unset | Failover pool. On 429/503 the request moves to the next model immediately instead of waiting on the same one. `cheap` = `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-flash-lite-latest`; or a comma-separated list of model ids. Backoff only starts after every model in the pool has failed |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address |
| `RETRY_MAX_SECONDS` | `900` | How long to keep retrying transient upstream failures before returning the error |
| `RETRY_EMPTY` | `3` | Retries when Gemini returns an empty reply |

## Run without Docker

```bash
pip install -r requirements.txt
GEMINI_API_KEY=your_key python opengem.py
```

## Retries

Transient Gemini failures (503 high demand, 429, other 5xx, network errors, empty replies) are retried with backoff (1s growing to 20s) instead of being passed to the client, so multi-step agents don't lose progress. Streaming responses send SSE keepalive comments while waiting; non-streaming requests simply take longer, so set generous client timeouts. Retries only happen before any output is forwarded, so nothing is duplicated. Blocked (safety) and length-limited replies are not retried. Retry attempts are logged (`docker logs`).

With `GEMINI_POOL` set, a 429/503 first switches to the next model in the pool (logged as `failing over to ...`); only when all of them have failed does it back off and retry. The `model` field in replies still shows the model the client asked for; the log shows which one actually answered the retry.

## Limitations

Function calling works on chat completions and responses (non-function tools such as web search are ignored, `previous_response_id` is not supported). No embeddings or logprobs. Each request logs the model asked for and the one used (`docker logs`). Anyone who can reach the port can spend your Gemini quota, so keep it on the LAN.
