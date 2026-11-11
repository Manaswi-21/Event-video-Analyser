# EventLens

EventLens is a React/Vite application with a Flask video-analysis API. It uses
Supabase Auth for accounts, YOLOv8 for object detection, an MLP for event
classification, and offline natural-language templates for reports.

## Deploy on Render

The root `render.yaml` defines two Render services:

- `eventlens-web`: the public React website.
- `eventlens-api`: the authenticated Flask API, with a persistent disk for
  reports and generated media.

To deploy:

1. Push this project to a GitHub repository you can access.
2. In Render, choose **New + > Blueprint** and connect that repository.
3. Review the services and costs before applying. The API uses a paid Standard
   instance and a 1 GB persistent disk so YOLO has memory and reports survive
   restarts. Render may ask you to add a payment method.
4. Enter the same Supabase project URL and publishable key for the backend
   (`SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`) and frontend
   (`VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`) when prompted.
   Never use a Supabase service-role/secret key in the frontend.
5. After the static site deploys, open its `onrender.com` URL and add it to
   Supabase Dashboard > Authentication > URL Configuration as the Site URL and
   an allowed Redirect URL (for example `https://your-site.onrender.com/**`).
6. Confirm email/password sign-in and the API health endpoint:
   `https://your-api.onrender.com/api/health`.

The frontend API host is wired to the backend host by the Render Blueprint.
The API validates Supabase access tokens and scopes reports/media to the
authenticated user.

## Local development

Frontend:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm ci
npm run dev
```

Set the Supabase project URL and publishable key in `frontend/.env.local`.

Backend:

```powershell
cd backend
python -m pip install -r requirements.txt
$env:SUPABASE_URL = "https://your-project.supabase.co"
$env:SUPABASE_PUBLISHABLE_KEY = "your-publishable-key"
python app.py
```

The browser sends the signed-in user's access token to the API. The backend
checks it with Supabase Auth. Reports and generated media are stored under
`DATA_DIR` (the default is the backend directory; Render mounts its persistent
disk at `/var/data`).

## MLP training

The checked-in demo classifier is trained from a tiny illustrative sample. Its
predictions are not production-quality and are explicitly marked `demo` in
reports. For meaningful event classification, train on representative,
accurately labeled samples from your own videos.

Create a CSV with the columns below, one labeled observation per row, then run:

```powershell
cd backend
python train_model.py path\to\labeled-events.csv
```

Required CSV columns:

```text
persons,laptops,monitors,microphones,chairs,phones,mean_confidence,event_type
```

The resulting `event_classifier.joblib` is intentionally ignored by Git. Keep
it private, or add it to a private model store; to deploy a trained model, set
`EVENT_MLP_MODEL_PATH` in Render to its mounted path. The default YOLOv8n model
detects general objects; it does not directly recognize every event category.

## Reports and limitations

Reports are generated offline from model output using natural-language
templates; this release does not call an external generative NLP service.
Predictions are estimates from up to eight sampled frames, not a verified
interpretation of an entire video. Uploaded source videos are deleted after
analysis; annotated media and reports are kept on the Render persistent disk.

The API accepts videos up to 250 MB by default. Change `MAX_UPLOAD_BYTES` in
Render if needed, while considering processing time and disk capacity.
