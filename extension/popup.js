// popup.js
const DEFAULT_API_BASE = "http://localhost:8000";

const apiInput = document.getElementById("apiBase");
const testBtn  = document.getElementById("testBtn");
const status   = document.getElementById("status");

// Load saved base URL
chrome.storage.local.get(["apiBase"], (result) => {
  apiInput.value = result.apiBase || DEFAULT_API_BASE;
});

// Save on change
apiInput.addEventListener("change", () => {
  chrome.storage.local.set({ apiBase: apiInput.value.trim() });
});

// Test connection
testBtn.addEventListener("click", async () => {
  const base = apiInput.value.trim() || DEFAULT_API_BASE;
  chrome.storage.local.set({ apiBase: base });

  status.className = "";
  status.textContent = "Connecting…";
  status.style.display = "block";

  try {
    const resp = await fetch(`${base}/`);
    if (resp.ok) {
      const data = await resp.json();
      status.className = "ok";
      status.textContent = `✓ ${data.message || "Connected"}`;
    } else {
      status.className = "error";
      status.textContent = `✗ HTTP ${resp.status}`;
    }
  } catch (err) {
    status.className = "error";
    status.textContent = `✗ ${err.message}`;
  }
});
