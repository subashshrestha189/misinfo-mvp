// content.js — MisInfo Guard
// Analyzes Twitter/X profile pages and shows a bot-risk panel with real account data.

// ─── API HELPERS ─────────────────────────────────────────────────────────────
// Requests go via background.js (service worker) to bypass mixed-content
// restrictions — content scripts on HTTPS cannot fetch HTTP localhost directly.

function sendToBackground(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => {
      resolve(response?.data ?? null);
    });
  });
}

function analyzeUser(userData) {
  return sendToBackground({ type: "ANALYZE_USER", payload: userData });
}

function analyzeProfileImage(imgUrl) {
  return sendToBackground({ type: "ANALYZE_PROFILE_IMAGE", imgUrl });
}

// ─── UTILITIES ────────────────────────────────────────────────────────────────

/** Parse "8.7K", "2M", "1,234" → integer */
function parseCount(text) {
  if (!text) return 0;
  const clean = String(text).replace(/,/g, "").trim();
  const m = clean.match(/^([\d.]+)\s*([KkMm]?)$/);
  if (!m) return 0;
  let n = parseFloat(m[1]);
  const s = m[2].toUpperCase();
  if (s === "K") n *= 1000;
  if (s === "M") n *= 1_000_000;
  return Math.round(n);
}

/** Pull a number from the first numeric leaf-span inside el */
function profileCount(el) {
  if (!el) return 0;
  for (const s of el.querySelectorAll("span")) {
    if (s.children.length > 0) continue;
    const n = parseCount(s.textContent);
    if (n > 0) return n;
  }
  return 0;
}

/** Poll until selector matches or timeout expires */
function waitForEl(selector, maxMs = 6000, intervalMs = 300) {
  return new Promise((resolve) => {
    const el = document.querySelector(selector);
    if (el) { resolve(el); return; }
    const start = Date.now();
    const timer = setInterval(() => {
      const found = document.querySelector(selector);
      if (found || Date.now() - start > maxMs) {
        clearInterval(timer);
        resolve(found || null);
      }
    }, intervalMs);
  });
}

function riskColor(prob) {
  if (prob >= 0.60) return "red";
  if (prob >= 0.30) return "orange";
  return "green";
}

function riskEmoji(prob) {
  if (prob >= 0.60) return "🔴";
  if (prob >= 0.30) return "🟠";
  return "🟢";
}

// ─── PROFILE PAGE ANALYSIS ───────────────────────────────────────────────────

// URL path segments that are NOT user profile pages
const PROFILE_EXCLUDED = new Set([
  "home", "explore", "notifications", "messages", "search", "settings",
  "compose", "about", "tos", "privacy", "rules", "i", "who_to_follow",
  "trending", "hashtag", "following", "followers",
]);

function isProfilePage() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  return (
    parts.length === 1 &&
    !PROFILE_EXCLUDED.has(parts[0].toLowerCase())
  );
}

function getProfileHandle() {
  return window.location.pathname.split("/").filter(Boolean)[0] || "";
}

function extractProfilePageData() {
  const ownerHandle = getProfileHandle();

  // Profile image — use the page owner's handle to find THEIR photo link,
  // not the logged-in user's avatar which appears first in the DOM.
  const avatarEl = (
    document.querySelector(`a[href="/${ownerHandle}/photo"] img`) ||
    document.querySelector(`a[href*="/${ownerHandle}/photo"] img`)
  );
  const profileImageUrl = avatarEl?.src
    ? avatarEl.src.replace(/_(normal|bigger|mini)\./, "_400x400.")
    : null;

  // Real profile images are served from pbs.twimg.com.
  // Default/placeholder avatars come from abs.twimg.com.
  const has_profile_image =
    profileImageUrl && profileImageUrl.includes("pbs.twimg.com") ? 1 : 0;

  // Verified badge
  const verified = document.querySelector(
    '[data-testid="UserName"] [data-testid="icon-verified"], [data-testid="verificationBadge"]'
  ) ? 1 : 0;

  // Bio / description
  const bioEl = document.querySelector('[data-testid="UserDescription"]');
  const has_description = (bioEl?.innerText?.trim().length ?? 0) > 0 ? 1 : 0;

  // Location
  const locationEl = document.querySelector('[data-testid="UserLocation"]');
  const has_location = locationEl ? 1 : 0;
  const locationText = locationEl?.innerText?.trim() || "";

  // Website URL
  const urlEl = document.querySelector('[data-testid="UserUrl"]');
  const has_url = urlEl ? 1 : 0;

  // Join date → account age in days
  const joinEl = document.querySelector('[data-testid="UserJoinDate"]');
  const joinText = joinEl?.innerText || "";
  const joinMatch = joinText.match(/Joined\s+(\w+)\s+(\d{4})/i);
  let account_age_days = 365;
  let joinDisplay = "";
  if (joinMatch) {
    const d = new Date(`${joinMatch[1]} 1, ${joinMatch[2]}`);
    if (!isNaN(d.getTime())) {
      account_age_days = Math.max(1, Math.round((Date.now() - d.getTime()) / 86_400_000));
      const yrs = Math.floor(account_age_days / 365);
      joinDisplay = yrs > 0
        ? `${joinMatch[1]} ${joinMatch[2]} (${yrs}yr${yrs > 1 ? "s" : ""})`
        : `${joinMatch[1]} ${joinMatch[2]}`;
    }
  }

  // Follower / Following counts — real values directly from the DOM
  const followers_count = profileCount(document.querySelector('a[href$="/followers"]'));
  const following_count = profileCount(document.querySelector('a[href$="/following"]'));

  // default_profile: 1 = account has never set a custom header banner (bot signal).
  // If a profile banner image exists, the user has customised their profile.
  const bannerEl = document.querySelector('[data-testid="UserBanner"] img, [data-testid="UserProfileHeader_Items"] img');
  const default_profile = bannerEl ? 0 : 1;

  // Tweet count — scan aria-labels for "X posts"
  let tweet_count = 0;
  document.querySelectorAll('[data-testid="primaryColumn"] [aria-label]').forEach((el) => {
    if (tweet_count) return;
    const m = (el.getAttribute("aria-label") || "").match(/([\d,.]+[KkMm]?)\s+post/i);
    if (m) tweet_count = parseCount(m[1]);
  });
  if (!tweet_count) {
    tweet_count = followers_count > 10_000 ? 50_000
                : followers_count > 1_000  ? 5_000
                : followers_count > 100    ? 500
                : 50;
  }

  const listed_count = Math.round(followers_count / 100);

  return {
    handle: ownerHandle,
    joinDisplay,
    locationText,
    followers_count,
    following_count,
    has_description,
    has_location,
    has_url,
    profileImageUrl,
    userData: {
      followers_count,
      following_count,
      tweet_count,
      listed_count,
      account_age_days,
      has_profile_image,
      default_profile,
      has_description,
      verified,
      has_location,
      has_url,
      favourites_count: 0,  // not shown on profile page DOM; model uses 0 as default
    },
  };
}

// ─── PANEL UI ─────────────────────────────────────────────────────────────────

function createProfilePanel() {
  const old = document.getElementById("mg-profile-panel");
  if (old) old.remove();

  const panel = document.createElement("div");
  panel.id = "mg-profile-panel";
  panel.innerHTML = `
    <div class="mg-panel-header">
      <span class="mg-panel-logo">MisInfo Guard</span>
      <button class="mg-panel-close" title="Close">&#x2715;</button>
    </div>
    <div class="mg-panel-body">
      <div class="mg-panel-loading">Analyzing profile&#8230;</div>
    </div>
  `;
  panel.querySelector(".mg-panel-close").addEventListener("click", () => panel.remove());
  document.body.appendChild(panel);
  return panel;
}

function updateProfilePanel(panel, profileData, userResult, imgResult) {
  const {
    handle, joinDisplay, locationText,
    followers_count, following_count,
    has_description, has_location, has_url,
    userData,
  } = profileData;

  // Use combined_bot_probability (blends ML + image risk) when available.
  const botProb  = userResult?.user?.combined_bot_probability ?? userResult?.user?.bot_probability ?? null;
  const trust    = userResult?.ensemble?.trust_score        ?? null;
  const imgScore = imgResult?.profile_image_risk_score      ?? null;
  const imgLevel = imgResult?.risk_level                    ?? null;

  const botColor = botProb  !== null ? riskColor(botProb)  : "grey";
  const botEmoji = botProb  !== null ? riskEmoji(botProb)  : "⚪";
  const botLabel = botProb  !== null ? `${(botProb * 100).toFixed(0)}%` : "?";
  const imgColor = imgScore !== null ? riskColor(imgScore) : "grey";
  const imgEmoji = imgScore !== null ? riskEmoji(imgScore) : "⚪";

  const fmt = (n) =>
    n >= 1_000_000 ? (n / 1_000_000).toFixed(1) + "M"
    : n >= 1_000   ? (n / 1_000).toFixed(1) + "K"
    : String(n);

  const sig = (ok, label) =>
    `<div class="mg-signal mg-signal--${ok ? "ok" : "warn"}">${ok ? "&#x2713;" : "&#x2717;"} ${label}</div>`;

  panel.querySelector(".mg-panel-body").innerHTML = `
    <div class="mg-panel-handle">@${handle}</div>

    <div class="mg-panel-scores">
      <div class="mg-score-block">
        <div class="mg-score-label">Bot Risk</div>
        <span class="mg-pill mg-pill--${botColor}">${botEmoji} ${botLabel}</span>
      </div>
      ${imgScore !== null ? `
      <div class="mg-score-block">
        <div class="mg-score-label">Image Risk</div>
        <span class="mg-pill mg-pill--${imgColor}">${imgEmoji} ${imgLevel ?? "?"}</span>
      </div>` : ""}
    </div>

    <div class="mg-panel-divider"></div>

    <div class="mg-panel-signals">
      ${sig(userData.verified,          "Verified account")}
      ${sig(has_description,            "Has bio")}
      ${sig(has_location,               "Has location")}
      ${sig(has_url,                    "Has website")}
      ${sig(userData.has_profile_image, "Has profile image")}
    </div>

    <div class="mg-panel-divider"></div>

    <div class="mg-panel-stats">
      ${joinDisplay     ? `<div>&#128197; Joined: ${joinDisplay}</div>`             : ""}
      ${followers_count ? `<div>&#128101; Followers: ${fmt(followers_count)}</div>` : ""}
      ${following_count ? `<div>&#10145;&#65039; Following: ${fmt(following_count)}</div>` : ""}
      ${locationText    ? `<div>&#128205; ${locationText}</div>`                    : ""}
      ${trust !== null  ? `<div>&#128737;&#65039; Trust Score: ${(trust * 100).toFixed(0)}%</div>` : ""}
      ${userResult === null ? `<div class="mg-warn">API unavailable</div>`          : ""}
    </div>
  `;
}

// ─── SPA NAVIGATION + MAIN ENTRY ─────────────────────────────────────────────
// Twitter/X is a React SPA — URL changes don't reload the page.
// We patch history.pushState and listen to popstate to detect navigation.

let _currentProfileHandle = null;

async function analyzeProfilePage() {
  if (!isProfilePage()) return;

  const handle = getProfileHandle();
  if (handle === _currentProfileHandle) return;
  _currentProfileHandle = handle;

  // Wait for the profile header to finish rendering
  const sentinel = await waitForEl('[data-testid="UserName"]', 6000);
  if (!sentinel || !isProfilePage()) return; // navigated away during wait

  const panel       = createProfilePanel();
  const profileData = extractProfilePageData();

  // Get image risk first — we pass it to /analyze/user so the backend
  // can blend it into combined_bot_probability at the ensemble layer.
  const imgResult = profileData.profileImageUrl
    ? await analyzeProfileImage(profileData.profileImageUrl)
    : null;

  const userPayload = { ...profileData.userData };
  if (imgResult?.profile_image_risk_score != null) {
    userPayload.profile_image_risk_score = imgResult.profile_image_risk_score;
  }
  const userResult = await analyzeUser(userPayload);

  // Only update if the user hasn't navigated away
  if (document.getElementById("mg-profile-panel") === panel) {
    updateProfilePanel(panel, profileData, userResult, imgResult);
  }
}

;(function patchHistory() {
  const orig = history.pushState.bind(history);
  history.pushState = function (...args) {
    orig(...args);
    _currentProfileHandle = null;
    setTimeout(analyzeProfilePage, 900);
  };
  window.addEventListener("popstate", () => {
    _currentProfileHandle = null;
    setTimeout(analyzeProfilePage, 900);
  });
})();

// Run on initial load (user may open a profile URL directly)
setTimeout(analyzeProfilePage, 1200);
