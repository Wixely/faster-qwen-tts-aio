# faster-qwen-tts-aio

An all-in-one TTS service that wraps [faster-qwen3-tts](https://pypi.org/project/faster-qwen3-tts/) and exposes it via:

- **OpenAI-compatible HTTP API** (`/v1/audio/speech`, `/v1/models`) for [Open WebUI](https://github.com/open-webui/open-webui) and any OpenAI-compatible client.
- **Wyoming protocol** TCP server for [Home Assistant](https://www.home-assistant.io/integrations/wyoming/).
- A small browser UI for trying voices, uploading reference audio, and managing custom voices.

> **GPU required.** `faster-qwen3-tts` is CUDA-only. There is no CPU fallback.

## Features

- Built-in named voices (Qwen3-TTS-CustomVoice model: `aiden`, `serena`, etc.) — optional.
- Zero-shot voice cloning (Qwen3-TTS-Base model) — upload a short reference clip + transcript, generate speech in that voice.
- WAV / FLAC / PCM output. MP3 / Opus / AAC if `ffmpeg` is on the PATH.
- Persistent custom voices and config in a single `data/` directory (mounted as a Docker volume).
- Single Docker image; one HTTP port (8080) and one Wyoming TCP port (10200).

## Quick start (Docker)

```bash
docker run --rm \
  --gpus all \
  -p 8080:8080 \
  -p 10200:10200 \
  -v $(pwd)/data:/data \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  ghcr.io/<owner>/faster-qwen-tts-aio:latest
```

Then open `http://localhost:8080`.

`docker-compose.yml` is included as a starting point.

### Volumes

| Path inside container | Purpose |
| --- | --- |
| `/data` | Custom voices, config, uploaded audio. Persist this. |
| `/root/.cache/huggingface` | HuggingFace model cache (multi-GB). Persist to avoid redownloads. |

## Configuration

All config is via environment variables. Defaults work out of the box.

| Var | Default | Notes |
| --- | --- | --- |
| `DATA_DIR` | `/data` | Custom voices, config, uploads. |
| `HTTP_HOST` | `0.0.0.0` | |
| `HTTP_PORT` | `8080` | OpenAI API + web UI. |
| `ENABLE_WYOMING` | `true` | |
| `WYOMING_URI` | `tcp://0.0.0.0:10200` | |
| `BASE_MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Used for voice cloning. Set to empty to disable. |
| `CUSTOM_VOICE_MODEL_ID` | *(empty)* | Set to e.g. `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` to enable built-in named voices. Doubles VRAM. |
| `BUILTIN_SPEAKERS` | `aiden,serena` | Comma-separated. Names exposed when CustomVoice model is loaded. |
| `DEFAULT_LANGUAGE` | `English` | Passed to faster-qwen3-tts. |
| `DEFAULT_VOICE` | `aiden` | Used when a request omits `voice`. |
| `DEFAULT_RESPONSE_FORMAT` | `wav` | One of `wav`, `flac`, `pcm`, `mp3`, `opus`, `aac` (`mp3`/`opus`/`aac` need `ffmpeg`). |
| `EAGER_LOAD` | `true` | If true, load the model(s) on startup. Otherwise lazy-load on first request. |
| `OPENAI_API_KEY` | *(empty)* | If set, `Authorization: Bearer <key>` is required on `/v1/*`. |
| `MAX_INPUT_CHARS` | `4096` | Per OpenAI's API. |
| `LOG_LEVEL` | `INFO` | |

## Open WebUI integration

In Open WebUI: **Admin Panel → Settings → Audio → TTS Engine → OpenAI**

- API Base URL: `http://faster-qwen-tts-aio:8080/v1` (or your host/port)
- API Key: any non-empty string (or the value of `OPENAI_API_KEY` if set)
- TTS Model: anything (e.g. `qwen-tts`) — the value is accepted but not validated.
- TTS Voice: a voice name shown on the test page. Either a built-in (`aiden`, `serena`, …) or one of your uploaded custom voices.

## Home Assistant integration

Install the official **Wyoming Protocol** integration. Add a service with:

- Host: the IP of the machine running this container
- Port: `10200`

The TTS will appear in HA's TTS service list. Voices match what is exposed by this server.

## Custom voice cloning

1. Open `http://localhost:8080`.
2. Go to the **Voices** tab.
3. Upload a reference clip (5–15 s of clean speech is plenty; WAV/FLAC/OGG/MP3).
4. Enter the **exact transcript** of the clip.
5. Give it a unique **name** (used as the voice identifier in OpenAI/Wyoming requests).
6. Save. The clip and transcript are stored under `data/voices/<name>/`.

You can then synthesise with `voice="<name>"` from any client.

## Local development

```bash
python -m venv .venv
. .venv/bin/activate         # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

You will need a CUDA-capable GPU and a working PyTorch install. PyTorch is **not** in `requirements.txt` (install it from the [official wheels](https://pytorch.org/get-started/locally/) for your CUDA version, or use the provided Docker image which is built on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`).

## License

MIT. See [LICENSE](LICENSE). The model weights are subject to their own licenses on HuggingFace.
