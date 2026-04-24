// background.js — MisInfo Guard service worker
// Proxies API calls from content.js to http://localhost because
// MV3 content scripts on HTTPS pages cannot fetch HTTP (mixed content).

const DEFAULT_API_BASE = "http://35.168.16.102:8000";

function getApiBase() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["apiBase"], (result) => {
      resolve((result.apiBase || DEFAULT_API_BASE).replace(/\/$/, ""));
    });
  });
}

// Ping the server on startup so the admin dashboard can count active installs.
chrome.runtime.onStartup.addListener(async () => {
  const API_BASE = await getApiBase();
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

// Also ping on first install / update.
chrome.runtime.onInstalled.addListener(async () => {
  const API_BASE = await getApiBase();
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
    getApiBase().then(API_BASE =>
      fetch(`${API_BASE}/analyze/user`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(message.payload),
      })
        .then(r => r.ok ? r.json() : null)
        .catch(() => null)
        .then(data => sendResponse({ data }))
    );
    return true; // keep channel open for async response
  }

  if (message.type === "ANALYZE_PROFILE_IMAGE") {
    getApiBase().then(API_BASE =>
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
        .then(data => sendResponse({ data }))
    );
    return true;
  }
});
