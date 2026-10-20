# Models

Model ids returned by `GET /v1/models` (those supporting `generateContent`), queried 2026-10-20. The list varies by API key and changes over time; query your own with `curl http://<lan-ip>:8000/v1/models`.

Default: `gemini-3.5-flash-lite` (override with `GEMINI_MODEL` or the request's `model`).

| Model id | Name | Input tokens | Output tokens |
|---|---|---|---|
| `gemini-2.5-flash` | Gemini 2.5 Flash | 1048576 | 65536 |
| `gemini-2.5-pro` | Gemini 2.5 Pro | 1048576 | 65536 |
| `gemini-2.5-flash-preview-tts` | Gemini 2.5 Flash Preview TTS | 8192 | 16384 |
| `gemini-2.5-pro-preview-tts` | Gemini 2.5 Pro Preview TTS | 8192 | 16384 |
| `gemma-4-26b-a4b-it` | Gemma 4 26B A4B IT | 262144 | 32768 |
| `gemma-4-31b-it` | Gemma 4 31B IT | 262144 | 32768 |
| `gemini-flash-latest` | Gemini Flash Latest | 1048576 | 65536 |
| `gemini-flash-lite-latest` | Gemini Flash-Lite Latest | 1048576 | 65536 |
| `gemini-pro-latest` | Gemini Pro Latest | 1048576 | 65536 |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash-Lite | 1048576 | 65536 |
| `gemini-2.5-flash-image` | Nano Banana | 32768 | 32768 |
| `gemini-3-flash-preview` | Gemini 3 Flash Preview | 1048576 | 65536 |
| `gemini-3.1-pro-preview` | Gemini 3.1 Pro Preview | 1048576 | 65536 |
| `gemini-3.1-pro-preview-customtools` | Gemini 3.1 Pro Preview Custom Tools | 1048576 | 65536 |
| `gemini-3.1-flash-lite-preview` | Gemini 3.1 Flash Lite Preview | 1048576 | 65536 |
| `gemini-3.1-flash-lite` | Gemini 3.1 Flash Lite | 1048576 | 65536 |
| `gemini-3-pro-image-preview` | Nano Banana Pro | 131072 | 32768 |
| `gemini-3-pro-image` | Nano Banana Pro | 131072 | 32768 |
| `nano-banana-pro-preview` | Nano Banana Pro | 131072 | 32768 |
| `gemini-3.1-flash-image-preview` | Nano Banana 2 | 65536 | 65536 |
| `gemini-3.1-flash-image` | Nano Banana 2 | 65536 | 65536 |
| `gemini-3.1-flash-lite-image` | Nano Banana 2 Lite | 65536 | 65536 |
| `gemini-nano-banana-2.1` | Nano Banana 2.1 | 65536 | 65536 |
| `gemini-3.5-flash` | Gemini 3.5 Flash | 1048576 | 65536 |
| `gemini-3.5-flash-lite` | Gemini 3.5 Flash Lite | 1048576 | 65536 |
| `gemini-omni-flash-preview` | Gemini Omni Flash Preview | 131072 | 65536 |
| `gemini-omni-1.1-flash` | Gemini Omni 1.1 Flash | 131072 | 65536 |
| `gemini-3.5-transcribe` | Gemini 3.5 Transcribe | 98304 | 32768 |
| `gemini-3.6-flash` | Gemini 3.6 Flash | 1048576 | 65536 |
| `gemini-3.7-flash` | Gemini 3.7 Flash | 1048576 | 65536 |
| `gemini-3.8-flash` | Gemini 3.8 Flash | 1048576 | 65536 |
| `gemini-4-argon` | Gemini 4 Argon | 1048576 | 32768 |
| `lyria-3-clip-preview` | Lyria 3 Clip Preview | 1048576 | 65536 |
| `lyria-3-pro-preview` | Lyria 3 Pro Preview | 1048576 | 65536 |
| `lyria-3.5` | Lyria 3.5 | 1048576 | 65536 |
| `gemini-3.1-flash-tts-preview` | Gemini 3.1 Flash TTS Preview | 8192 | 16384 |
| `gemini-3.8-flash-tts` | Gemini 3.8 Flash TTS | 8192 | 16384 |
| `gemini-3.8-flash-lite-tts` | Gemini 3.8 Flash Lite TTS | 8192 | 16384 |
| `gemini-robotics-er-2-preview` | Gemini Robotics-ER 2 Preview | 131072 | 65536 |
| `gemini-2.5-computer-use-preview-10-2025` | Gemini 2.5 Computer Use Preview 10-2025 | 131072 | 65536 |
| `antigravity-preview-05-2026` | Antigravity Agent Preview | 1048576 | 65536 |
| `antigravity-preview-09-2026` | Antigravity Agent Preview | 1048576 | 65536 |
| `antigravity-preview-latest` | Antigravity Agent Preview Latest | 1048576 | 65536 |
| `deep-research-max-preview-04-2026` | Deep Research Max Preview (Apr-21-2026) | 131072 | 65536 |
| `deep-research-preview-04-2026` | Deep Research Preview (Apr-21-2026) | 131072 | 65536 |
| `deep-research-pro-preview-12-2025` | Deep Research Pro Preview (Dec-12-2025) | 131072 | 65536 |
