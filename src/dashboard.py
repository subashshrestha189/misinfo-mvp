import json
import time
from typing import Dict, Any

import requests
import pandas as pd
import streamlit as st

from config import API_BASE

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Bot & Profile Risk Detector",
    page_icon="🛡️",
    layout="wide")


# ---------- SIDEBAR / SETTINGS ----------
with st.sidebar:
    st.title("⚙️ Settings")
    api_base = st.text_input("API Base URL", value=API_BASE, help="Your FastAPI base URL")
    if st.button("Test API Connection"):
        try:
            r = requests.get(f"{api_base}/")
            if r.ok:
                st.success(f"Connected: {r.json().get('message','OK')}")
            else:
                st.error(f"Status {r.status_code}: {r.text[:200]}")
        except Exception as e:
            st.error(f"Connection error: {e}")

    st.markdown("---")
    st.caption("Tip: Keep your FastAPI running in another terminal.")

st.title("🛡️ Bot & Profile Risk Detection")
st.write("Detect bots from account metadata and assess profile image risk.")

tabs = st.tabs(["👤 User Analyzer", "🖼️ Profile Image Privacy", "📈 Diagnostics"])


# ---------- HELPERS ----------
def call_api_json(method: str, url: str, payload: Dict[str, Any] | None = None, timeout: int = 30):
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)

def risk_badge(prob: float) -> str:
    if prob > 0.60:  return "🔴 High"
    if prob > 0.30:  return "🟠 Medium"
    return "🟢 Low"


# ---------- TAB 1: USER ANALYZER ----------
with tabs[0]:
    st.subheader("👤 User Bot Detection")
    st.caption("Enter account metadata to score bot probability and overall trust.")

    with st.form("user_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            followers_count  = st.number_input("Followers",         min_value=0,  value=120,  step=1)
            listed_count     = st.number_input("Listed Count",       min_value=0,  value=10,   step=1)
        with c2:
            following_count  = st.number_input("Following",          min_value=0,  value=200,  step=1)
            account_age_days = st.number_input("Account Age (days)", min_value=1,  value=800,  step=1)
        with c3:
            tweet_count      = st.number_input("Tweet Count",        min_value=0,  value=1500, step=1)

        st.markdown("#### Profile Flags")
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1: has_profile_image = st.checkbox("Has Profile Image", value=True)
        with d2: has_description   = st.checkbox("Has Description",   value=True)
        with d3: verified          = st.checkbox("Verified",          value=False)
        with d4: has_location      = st.checkbox("Has Location",      value=True)
        with d5: has_url           = st.checkbox("Has URL",           value=False)

        submit_user = st.form_submit_button("Analyze User")

    if submit_user:
        payload = {
            "followers_count":  float(followers_count),
            "following_count":  float(following_count),
            "tweet_count":      float(tweet_count),
            "listed_count":     float(listed_count),
            "account_age_days": float(account_age_days),
            "has_profile_image": int(has_profile_image),
            "has_description":   int(has_description),
            "verified":          int(verified),
            "has_location":      int(has_location),
            "has_url":           int(has_url),
        }
        with st.spinner("Scoring bot probability…"):
            data, err = call_api_json("POST", f"{api_base}/analyze/user", payload)
            time.sleep(0.1)

        if err:
            st.error(f"API error: {err}")
        elif not data:
            st.error("No response from API.")
        else:
            st.success("Analysis complete.")

            user     = data.get("user", {})
            ensemble = data.get("ensemble", {})

            # --- Top-level metrics ---
            m1, m2, m3 = st.columns(3)
            with m1:
                bot_prob = float(user.get("bot_probability", 0.0))
                st.metric("Bot Probability", f"{bot_prob*100:.1f}%")
                st.write(f"**Risk Level:** {risk_badge(bot_prob)}")
                st.progress(min(max(bot_prob, 0.0), 1.0))
            with m2:
                trust = ensemble.get("trust_score")
                if trust is not None:
                    st.metric("Trust Score", f"{float(trust)*100:.1f}%")
                st.write(f"**{ensemble.get('trust_level', '—')}**")
            with m3:
                heur = float(data.get("heuristics", {}).get("heuristic_score", 0.0))
                st.metric("Heuristic Score", f"{heur*100:.1f}%")

            # --- Ensemble breakdown chart ---
            trust_val = float(ensemble.get("trust_score", 0.0))
            bot_trust = 1.0 - bot_prob
            heur_val  = float(data.get("heuristics", {}).get("heuristic_score", 0.0))

            st.markdown("#### Trust Score Breakdown")
            df_ensemble = pd.DataFrame([
                {"Component": "Bot Trust / 1−P(bot) (w=0.6)", "Score": bot_trust},
                {"Component": "Heuristic Score (w=0.4)",      "Score": heur_val},
                {"Component": "▶ Final Trust Score",          "Score": trust_val},
            ])
            st.bar_chart(df_ensemble.set_index("Component"))
            st.dataframe(
                df_ensemble.style.highlight_max(subset=["Score"], color="#d4edda")
                                 .format({"Score": "{:.3f}"}),
                use_container_width=True,
            )

            with st.expander("Raw API response"):
                st.json(data)


# ---------- TAB 2: PROFILE IMAGE PRIVACY ----------
with tabs[1]:
    st.subheader("🖼️ Profile Image Risk & Privacy")

    col_left, col_right = st.columns(2)

    # --- Risk scoring ---
    with col_left:
        st.markdown("#### Risk Scoring (`/analyze/profile-image`)")
        uploaded_risk = st.file_uploader("Upload profile image", type=["png","jpg","jpeg"], key="risk_upload")
        if st.button("Analyze Image Risk") and uploaded_risk:
            with st.spinner("Scoring image risk…"):
                try:
                    resp = requests.post(
                        f"{api_base}/analyze/profile-image",
                        files={"file": (uploaded_risk.name, uploaded_risk.getvalue(), uploaded_risk.type)},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.image(uploaded_risk, caption="Uploaded image", use_column_width=True)
                    st.metric("Risk Score", f"{result.get('profile_image_risk_score', 0)*100:.1f}%")
                    st.write(f"**Risk Level:** {result.get('risk_level','—').capitalize()}")
                    with st.expander("Signals"):
                        st.json(result.get("signals", {}))
                    if result.get("notes"):
                        for note in result["notes"]:
                            st.caption(f"• {note}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API error: {e}")
        elif not uploaded_risk:
            st.info("Upload an image to score its risk.")

    # --- Blur on demand ---
    with col_right:
        st.markdown("#### Blur on Demand (`/privacy/blur-on-demand`)")
        uploaded_blur = st.file_uploader("Upload profile image", type=["png","jpg","jpeg"], key="blur_upload")
        blur_strength = st.slider("Blur strength", min_value=5, max_value=99, value=35, step=2)
        if st.button("Apply Privacy Blur") and uploaded_blur:
            with st.spinner("Applying blur…"):
                try:
                    resp = requests.post(
                        f"{api_base}/privacy/blur-on-demand",
                        files={"file": (uploaded_blur.name, uploaded_blur.getvalue(), uploaded_blur.type)},
                        params={"blur_strength": blur_strength},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    risk_score      = resp.headers.get("X-Risk-Score", "—")
                    risk_level      = resp.headers.get("X-Risk-Level", "—")
                    privacy_applied = resp.headers.get("X-Privacy-Applied", "False")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(uploaded_blur, caption="Original", use_column_width=True)
                    with col_b:
                        st.image(resp.content, caption="Processed", use_column_width=True)
                    st.metric("Risk Score", f"{float(risk_score)*100:.1f}%" if risk_score != "—" else "—")
                    st.write(f"**Risk Level:** {risk_level.capitalize()}")
                    st.write(f"**Blur Applied:** {'Yes' if privacy_applied == 'True' else 'No'}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API error: {e}")
        elif not uploaded_blur:
            st.info("Upload an image to apply blur.")


# ---------- TAB 3: DIAGNOSTICS ----------
with tabs[2]:
    st.subheader("📈 Diagnostics & Health")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**API Ping**")
        ping_data, ping_err = call_api_json("GET", f"{api_base}/")
        if ping_err:
            st.error(f"Ping failed: {ping_err}")
        else:
            st.success(ping_data)
    with c2:
        st.write("**Config**")
        st.code(json.dumps({"API_BASE": api_base}, indent=2))
    st.caption("If API ping fails, verify FastAPI is running and the base URL is correct.")
