// background.js — MisInfo Guard service worker
// Proxies API calls from content.js to http://localhost because
// MV3 content scripts on HTTPS pages cannot fetch HTTP (mixed content).

const API_BASE = "http://localhost:8000";

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
