# opengem

OpenAI-compatible proxy for Google Gemini. **No auth, for private LAN use only.**

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models` (streaming supported).

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
      # GEMINI_MODEL: gemini-2.5-flash   # default model when the client sends none
    ports:
      - "8000:8000"
```

## Use it

```bash
curl http://<lan-ip>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

In OpenAI clients set `base_url=http://<lan-ip>:8000/v1` and use any dummy API key.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | required | Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model used when the request has none |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address |

## Run without Docker

```bash
pip install -r requirements.txt
GEMINI_API_KEY=your_key python opengem.py
```

## Limitations

No tool/function calling, embeddings or logprobs. Anyone who can reach the port can spend your Gemini quota, so keep it on the LAN.
