# AnomaVision Studio Web

The web UI for AnomaVision Studio.

## Stack

- Next.js
- React
- TypeScript
- lucide-react
- Local-first filesystem storage remains owned by the Python Studio services.

The web app is intentionally separate from the anomaly-detection engine. It will consume a small Python API rather than reimplementing training, inference, export, validation, or monitoring.

## Run locally

From the repository root:

```powershell
cd apps/studio-web
npm install
npm run dev
```

Open the local URL shown by Next.js.

In a second terminal, start the Python Studio API from the repository root:

```powershell
uv run uvicorn apps.studio.api.app:app --host 127.0.0.1 --port 8000
```

The web app defaults to `http://localhost:8000`. To use another API URL, set `NEXT_PUBLIC_STUDIO_API_URL` before starting Next.js.

## Architecture

```
Next.js / React
      |
      v
Studio API (Python / FastAPI)
      |
      +--> apps/studio/services
      |
      +--> anomavision core
```

Do not add ML logic to this directory.
