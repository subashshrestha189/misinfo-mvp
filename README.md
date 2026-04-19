# MisInfo Guard — Bot & Profile Risk Detection

Detects social media bots and assesses profile image risk using ML models, served via a FastAPI backend, a Streamlit dashboard, and a Chrome extension for live analysis on X (Twitter).

This project was completed as part of the **AIDI1003 Capstone Project** at **Durham College**.

---

## Features

- **Bot Detection** — Random Forest classifier trained on TwiBot-22 metadata (700K accounts, 92.2% accuracy)
- **Profile Image Risk Scoring** — MediaPipe + OpenCV face analysis pipeline
- **Ensemble Scoring** — Blends ML bot score (80%) with image risk (20%) into a combined probability
- **Chrome Extension** — Live badge overlay on X (Twitter) profiles
- **Streamlit Dashboard** — Analytics, user analyzer, diagnostics, and admin panel
- **FastAPI Backend** — REST API with `/analyze/user`, `/analyze/profile-image`, `/privacy/blur-on-demand`

---

## Project Structure

```
misinfo-mvp/
├── src/
│   ├── app.py                  # FastAPI application
│   ├── dashboard.py            # Streamlit dashboard
│   ├── ensemble.py             # Score blending logic
│   ├── twibot_features.py      # Feature engineering
│   ├── analytics.py            # SQLite analytics logging
│   ├── config.py               # API base URL config
│   └── cv/                     # Computer vision modules
│       ├── face_detect.py
│       ├── profile_risk.py
│       ├── privacy_blur.py
│       └── io_utils.py
├── extension/                  # Chrome MV3 extension
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   ├── popup.html / popup.js
│   └── styles.css
├── models/
│   └── bot_tuned/              # Trained model files (not in git)
│       ├── twibot_rf_calibrated.joblib
│       ├── feature_schema.joblib
│       └── summary.json
├── requirements.txt            # Streamlit Cloud (lightweight)
├── requirements_full.txt       # EC2 / local (full, includes opencv + mediapipe)
└── start.sh                    # EC2 startup script
```

---

## Setup

### Prerequisites

- Python 3.11
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/subashshrestha189/misinfo-mvp.git
cd misinfo-mvp
```

---

### 2. Local Development

**Create and activate a virtual environment:**

```bash
# Windows (PowerShell)
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1

# macOS / Linux
python3.11 -m venv .venv311
source .venv311/bin/activate
```

**Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements_full.txt
```

**Place trained model files** in `models/bot_tuned/`:
- `twibot_rf_calibrated.joblib`
- `feature_schema.joblib`
- `summary.json`

**Start the API:**

```bash
python -m uvicorn src.app:app --reload
```

API docs available at: `http://localhost:8000/docs`

**Start the dashboard (separate terminal):**

```bash
streamlit run src/dashboard.py
```

---

### 3. EC2 Deployment

**Requirements:** Ubuntu EC2 instance, port 8000 open in Security Group.

**SSH into the instance:**

```bash
ssh -i ~/Downloads/misinfo-key.pem ubuntu@<your-ec2-ip>
```

**On the EC2 instance:**

```bash
git clone https://github.com/subashshrestha189/misinfo-mvp.git
cd misinfo-mvp
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_full.txt
```

**Copy trained models from local machine** (run on your local machine):

```bash
scp -i ~/Downloads/misinfo-key.pem \
    models/bot_tuned/twibot_rf_calibrated.joblib \
    models/bot_tuned/feature_schema.joblib \
    ubuntu@<your-ec2-ip>:/home/ubuntu/misinfo-mvp/models/bot_tuned/
```

**Start the API on EC2:**

```bash
bash start.sh
```

---

### 4. Streamlit Cloud Dashboard

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**:
   - Repository: `subashshrestha189/misinfo-mvp`
   - Branch: `main`
   - Main file path: `src/dashboard.py`
3. Click **Advanced settings → Secrets** and add:
   ```toml
   API_BASE = "http://<your-ec2-ip>:8000"
   ```
4. Under **General**, set Python version to **3.11**
5. Click **Deploy**

> Note: `requirements.txt` in this repo is the lightweight version used by Streamlit Cloud. Full dependencies (opencv, mediapipe) are in `requirements_full.txt`.

---

### 5. Chrome Extension

1. Open Chrome and go to `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked** and select the `extension/` folder
4. Click the MisInfo Guard icon → set **FastAPI Base URL** to `http://<your-ec2-ip>:8000`
5. Visit any profile on X (Twitter) to see live bot risk scores

---

## Deployment Architecture

| Component | Platform | Specification | Rationale |
|-----------|----------|---------------|-----------|
| FastAPI backend | AWS EC2 | t3.medium — Ubuntu 22.04 LTS, 4GB RAM | Minimum viable spec for Random Forest + MediaPipe concurrent load (~1.15GB). Ubuntu 22.04 required for mediapipe 0.10.x compatibility. |
| Model storage | EC2 local disk | Copied via `scp` into `models/bot_tuned/` | Models are ~50MB total — no S3 needed at this scale. EC2 instance holds both the API and model files. |
| Streamlit dashboard | Streamlit Community Cloud | Free tier, GitHub-connected | Zero infrastructure cost. Auto-deploys on git push. `API_BASE` read from `st.secrets` for environment separation. |
| Chrome extension | Manifest V3 (unpacked) | Content script on x.com | MV3 is the current Chrome standard. Service worker proxies HTTP API calls to bypass mixed-content restrictions on HTTPS pages. |
| Training (one-time) | Local machine | Python 3.11, venv311 | TwiBot-22 feature extraction and RF training run locally. Trained model files are then copied to EC2 via `scp`. |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/analyze/user` | Bot probability from user metadata |
| `POST` | `/analyze/profile-image` | Profile image risk score |
| `POST` | `/privacy/blur-on-demand` | Blur faces in high-risk images |
| `POST` | `/ping` | Extension ping (analytics) |

Interactive docs: `http://<your-host>:8000/docs`

---

## Model Performance

### Bot Detection (TwiBot-22)
- Training samples: 700,000 accounts
- Accuracy: 70.9%
- ROC-AUC: 0.77
- Bot F1-score: 0.60
- Human Precision: 81.2% / Bot Precision: 55.3%
- Features: 25 metadata features

### Profile Image Risk
- Face detection via MediaPipe
- Signals: face presence, image quality, metadata anomalies

---

## Team

| Name | Role |
|------|------|
| Subash Shrestha | Model integration, API deployment, Chrome extension, GitHub |
| Baburam Panta | Bot feature engineering and dataset preprocessing |
| Aman Bansal | Transformer fine-tuning |
| Laxman Neupane | Backend architecture |
| Sisam Kafle | Streamlit UX design |
| Aditya Sharma | Baseline modeling & evaluation |

---

## License

Academic use only — Durham College Capstone Project
