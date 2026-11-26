# RallyLens dashboard (Next.js)

The coach-facing web app. Upload a single-stroke clip and see:
- the detected stroke with a confidence ring and top-3 probabilities;
- the Grad-CAM evidence;
- a low-confidence warning when needed;
- the predicted next shot;
- the rally timeline;
- a coaching cue.

It talks to the FastAPI service in [`../backend`](../backend).

## Run it

Requires Node.js 18 or later.

```bash
cd frontend
npm install
cp .env.local.example .env.local   # optional: set NEXT_PUBLIC_API_URL
npm run dev                        # http://localhost:3000
```

## Connect to the backend

Click **Settings** in the top bar, paste the backend URL and press **Save & connect**. The URL is saved in the browser (`localStorage`), and the status pill turns to "Model online" once `/health` reports the stroke model is loaded.

- **Backend on Colab:** use the printed `https://<random>.trycloudflare.com` URL. It changes every session.
- **Backend running locally:** use `http://localhost:8000`, which is also the default.

## Use it

1. Check that the status pill says **Model online**.
2. Drop a clip containing one stroke (MP4, MOV, AVI or MKV).
3. Upload the next clips in rally order to build the timeline and next-shot forecast. Use **Reset rally** to start a new rally.

## Stack

Next.js 14 (App Router), React 18, TypeScript (strict mode) and Tailwind CSS. The page is a light dashboard with a dark-green hero, Plus Jakarta Sans and Inter fonts, a custom SVG confidence ring and probability bars, and a responsive layout. There are no UI or chart libraries.
