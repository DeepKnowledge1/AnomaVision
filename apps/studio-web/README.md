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
