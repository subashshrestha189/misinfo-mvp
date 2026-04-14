// background.js — MisInfo Guard service worker
// Proxies API calls from content.js to http://localhost because
// MV3 content scripts on HTTPS pages cannot fetch HTTP (mixed content).

const API_BASE = "http://localhost:8000";

// Ping the server on startup so the admin dashboard can count active installs.
chrome.runtime.onStartup.addListener(() => {
  fetch(`${API_BASE}/ping`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Extension-Id": chrome.runtime.id,
      "X-Extension-Version": chrome.runtime.getManifest().version,
    },
    body: JSON.stringify({}),
  }).catch(() => {}); // silently ignore if server is offline
});

// Also ping on first install / update.
chrome.runtime.onInstalled.addListener(() => {
  fetch(`${API_BASE}/ping`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Extension-Id": chrome.runtime.id,
      "X-Extension-Version": chrome.runtime.getManifest().version,
    },
    body: JSON.stringify({}),
  }).catch(() => {});
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {

  if (message.type === "ANALYZE_USER") {
    fetch(`${API_BASE}/analyze/user`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.payload),
    })
      .then(r => r.ok ? r.json() : null)
      .catch(() => null)
      .then(data => sendResponse({ data }));
    return true; // keep channel open for async response
  }

  if (message.type === "ANALYZE_PROFILE_IMAGE") {
    fetch(message.imgUrl)
      .then(r => r.ok ? r.blob() : null)
      .then(blob => {
        if (!blob) return null;
        const fd = new FormData();
        fd.append("file", blob, "profile.jpg");
        return fetch(`${API_BASE}/analyze/profile-image`, {
          method: "POST",
          body: fd,
        });
      })
      .then(r => (r && r.ok) ? r.json() : null)
      .catch(() => null)
      .then(data => sendResponse({ data }));
    return true;
  }
});
