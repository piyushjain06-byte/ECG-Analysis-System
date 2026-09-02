# Deployment — Streamlit Community Cloud

This gets you a free, public URL for the app. Streamlit Community Cloud
builds directly from a GitHub repo, so step 1 is getting this project
onto GitHub.

## 1. Push this project to GitHub

```bash
cd CardioDSP-Lab
git init
git add .
git commit -m "CardioDSP Lab - ECG DSP/ML prototype"
```
Create an **empty** repository on GitHub (no README/license, since this
project already has them), then:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

## 2. Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **New app**.
3. Pick your repo, branch `main`, and main file path `app.py`.
4. Click **Deploy**.

The build installs everything in `requirements.txt` automatically. First
build takes a few minutes (xgboost/scipy/matplotlib are the slow ones).

## 3. What ships with the deployed app

The repo includes pre-trained models (`models/trained/*.pkl`) trained on
the synthetic generator, so the deployed app works immediately — demo
button, CSV upload, and DSP/ML pages all functional with zero extra setup
on the server.

**Real MIT-BIH-trained models do not auto-deploy.** If you've retrained
locally with `python -m src.ml_training --source both` (see
[RUN_GUIDE.md](RUN_GUIDE.md)), commit the new `models/trained/*.pkl` and
`models/metadata/ml_performance.json` and push — Streamlit Cloud
redeploys automatically on every push to `main`. The raw MIT-BIH
recordings themselves (`data/mitbih/`) are gitignored and never need to
be uploaded to GitHub or the server; only the trained model files do.

## 4. Config already in place

- `.streamlit/config.toml` sets the dark theme — no extra configuration
  needed on Streamlit Cloud's side.
- `requirements.txt` is version-pinned (`~=`) for a reproducible build.
- `data/raw/`, `data/mitbih/`, `data/history.db`, and `reports/*.pdf` are
  gitignored — the deployed app writes to these at runtime; they reset on
  every redeploy (fine, since they're just cache/output, not source).

## 5. After deploying

- The disclaimer banner and About page ship as-is — do not remove them if
  you share the public link, since the academic-prototype framing is
  what makes the link appropriate to share at all.
- Community Cloud free tier apps sleep after inactivity and wake on the
  next visit (a ~30s cold start) — normal behavior, not a bug.

## Alternative: run it yourself without Streamlit Cloud

If you'd rather not use a third-party host, any machine with Python can
run it directly (see [RUN_GUIDE.md](RUN_GUIDE.md)) — `streamlit run app.py`
serves on `localhost:8501`, or add `--server.address 0.0.0.0` to expose it
on your local network.
