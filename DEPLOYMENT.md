# Deploying CreditSense for Free

Two pieces to deploy:
1. **Backend** (Flask API + model) → Render.com (free tier)
2. **Frontend** (React UI, single HTML file) → Netlify (free, drag-and-drop)

Total cost: **$0**. No credit card required for either.

---

## Part 1 — Put the code on GitHub

Render and Netlify both deploy from a Git repository.

```bash
cd credit-sense
git init
git add .
git commit -m "CreditSense: ML loan approval predictor"
```

Create a new empty repo on GitHub (e.g. `credit-sense`), then:
```bash
git remote add origin https://github.com/<your-username>/credit-sense.git
git branch -M main
git push -u origin main
```

---

## Part 2 — Deploy the backend (Flask API) on Render

1. Go to **render.com** → sign up free (GitHub login is fastest).
2. Click **New +** → **Web Service**.
3. Connect your `credit-sense` GitHub repo.
4. Fill in the settings:
   - **Root Directory:** `backend`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Instance Type:** Free
5. Click **Create Web Service**. Wait ~2–3 minutes for the first deploy.
6. Once live, Render gives you a URL like:
   `https://creditsense-api.onrender.com`
7. Test it: open `https://creditsense-api.onrender.com/api/health` in your
   browser — you should see `{"status": "ok", "model_loaded": true}`.

**Free-tier note:** the service "sleeps" after 15 minutes with no traffic.
The next request after that takes ~30–50 seconds to wake up — totally fine
for a portfolio demo, just give it a moment if it feels slow at first.

---

## Part 3 — Point the frontend at your live backend

Open `frontend/index.html`, find this line near the top of the `<script>`:
```js
const API_BASE = "http://localhost:5000";
```
Replace it with your Render URL:
```js
const API_BASE = "https://creditsense-api.onrender.com";
```
Save, commit, and push:
```bash
git add frontend/index.html
git commit -m "Point frontend to deployed API"
git push
```

---

## Part 4 — Deploy the frontend on Netlify

**Easiest way (no account needed to try it):**
1. Go to **app.netlify.com/drop**
2. Drag your `frontend` folder onto the page.
3. Netlify instantly gives you a live URL like `https://random-name-123.netlify.app`.

**Better way (stays in sync with GitHub):**
1. Go to **netlify.com** → sign up free with GitHub.
2. **Add new site** → **Import an existing project** → pick your `credit-sense` repo.
3. **Base directory:** `frontend`
4. **Build command:** (leave empty — it's a static file, no build needed)
5. **Publish directory:** `frontend`
6. Deploy. You'll get a live URL in under a minute.

---

## Part 5 — Test the full live app

Open your Netlify URL, fill in the loan application form, and click
**Predict approval**. It calls your live Render API and shows a real
prediction — the same pipeline you built locally, now live on the internet
with a shareable link.

---

## Optional: custom domain / nicer URL

Both Render and Netlify let you rename the auto-generated subdomain for
free (e.g. `creditsense.netlify.app` instead of a random string) under
**Site settings → Change site name** (Netlify) or **Settings → Name**
(Render). A real custom domain (e.g. `creditsense.com`) costs money for
the domain itself, but isn't required.

## Troubleshooting

- **"Failed to fetch" / CORS error in the browser console:** double-check
  `API_BASE` in `index.html` exactly matches your Render URL (https, no
  trailing slash).
- **Render build fails:** check the build logs — it's almost always a
  typo in the Root Directory or a missing package in `requirements.txt`.
- **First request is very slow:** that's the free-tier "cold start" —
  normal, not a bug.
