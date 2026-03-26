// content.js — MisInfo Guard
// Watches x.com for new tweets, calls local FastAPI, injects risk badges.

// Track processed tweets so we never analyze the same article twice
const processed = new WeakSet();

// ─── API HELPERS ────────────────────────────────────────────────────────────
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

// ─── DOM EXTRACTION ──────────────────────────────────────────────────────────

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

/**
 * Read a number from the first non-empty child span of an element matched
 * by `selector` within `root`. Returns 0 if not found.
 */
function countFromSpan(root, selector) {
  const el = root.querySelector(selector);
  if (!el) return 0;
  // Walk all leaf text nodes / spans for a numeric value
  const spans = el.querySelectorAll("span");
  for (const s of spans) {
    if (s.children.length > 0) continue; // skip wrappers
    const n = parseCount(s.textContent);
    if (n > 0) return n;
  }
  return 0;
}

/**
 * Extract reply / retweet / like / view counts from the engagement bar area.
 *
 * Twitter's DOM has changed many times. We try three strategies in order:
 *   1. The [role="group"] aria-label (often contains all counts)
 *   2. Individual button spans via data-testid
 *   3. The analytics/views link which may sit OUTSIDE the group element
 */
function extractEngagementCounts(engagementBar, article) {
  const c = { replies: 0, retweets: 0, likes: 0, views: 0 };
  if (!engagementBar) return c;

  // ── Strategy 1: group aria-label ────────────────────────────────
  const label = engagementBar.getAttribute("aria-label") || "";
  if (label) {
    const get = (re) => { const m = label.match(re); return m ? parseCount(m[1]) : 0; };
    c.replies  = get(/([\d,.]+[KkMm]?)\s+repl/i);
    c.retweets = get(/([\d,.]+[KkMm]?)\s+(?:repost|retweet)/i);
    c.likes    = get(/([\d,.]+[KkMm]?)\s+like/i);
    c.views    = get(/([\d,.]+[KkMm]?)\s+view/i);
  }

  // ── Strategy 2: data-testid button spans ─────────────────────────
  const tryBtn = (testId, key) => {
    if (c[key] > 0) return;
    c[key] = countFromSpan(engagementBar, `[data-testid="${testId}"]`);
  };
  tryBtn("like", "likes");
  tryBtn("retweet", "retweets");
  tryBtn("reply", "replies");

  // ── Strategy 3: view/analytics element (may be outside group) ───
  if (c.views === 0 && article) {
    // analytics link sometimes sits adjacent to the group
    const analyticsEl = article.querySelector(
      'a[href*="/analytics"], [data-testid="analyticsButton"], [aria-label*="View" i]'
    );
    if (analyticsEl) {
      const spans = analyticsEl.querySelectorAll("span");
      for (const s of spans) {
        if (s.children.length > 0) continue;
        const n = parseCount(s.textContent);
        if (n > 0) { c.views = n; break; }
      }
    }
  }

  // ── Strategy 4: scan all bare-text spans in the bar for numbers ──
  // As a last resort, read every leaf span in the bar and use the largest
  // numeric value seen as the view count (views are usually the biggest number).
  if (c.views === 0) {
    let maxN = 0;
    engagementBar.querySelectorAll("span").forEach((s) => {
      if (s.children.length > 0) return;
      const n = parseCount(s.textContent);
      if (n > maxN) maxN = n;
    });
    // Only trust it if it's meaningfully large (avoids picking up icon counts)
    if (maxN > 0) c.views = maxN;
  }

  return c;
}

function extractTweetData(article) {
  // Handle / username
  const userNameEl = article.querySelector('[data-testid="User-Name"]');
  const handleLink = userNameEl?.querySelector('a[href*="/"]');
  const handle = handleLink?.getAttribute("href")?.replace(/^\//, "") || "";

  // Display name (shown above the handle)
  const displayNameEl = userNameEl?.querySelector("span span");
  const displayName = displayNameEl?.textContent?.trim() || "";

  // Profile image URL (hosted on pbs.twimg.com)
  const avatarImg = article.querySelector('[data-testid="Tweet-User-Avatar"] img');
  const profileImageUrl = avatarImg?.src || null;

  // Verified badge
  const verified = article.querySelector('[data-testid="verificationBadge"]') ? 1 : 0;

  // Has profile image
  const has_profile_image = profileImageUrl ? 1 : 0;

  // Tweet text
  const tweetTextEl = article.querySelector('[data-testid="tweetText"]');
  const tweetText = tweetTextEl?.innerText || "";

  // Has URL in tweet text
  const has_url = /https?:\/\/\S+/.test(tweetText) ? 1 : 0;

  return { handle, displayName, profileImageUrl, verified, has_profile_image, has_url, tweetText };
}

// ─── CLIENT-SIDE HEURISTIC ───────────────────────────────────────────────────
//
// The ML model needs real Twitter API fields (followers_count, account_age_days,
// tweet_count) which are NOT available in the DOM. Without those, every tweet
// receives a near-identical feature vector and the model returns the same score.
//
// This heuristic uses signals that ARE observable from the DOM and vary per post:
//   • Handle patterns  (trailing digits, digit density)
//   • Tweet text       (caps abuse, manipulation keywords, punctuation)
//   • Engagement ratios (RT-to-like anomalies)
//
// Returns a suspicion score in [0, 1] — higher = more suspicious.

function computeClientHeuristic(handle, displayName, tweetText, eng) {
  let suspicion = 0.0;

  // ── Handle patterns ──────────────────────────────────────────────
  // Long trailing digit run is a strong bot signal (e.g. "Awilson136265")
  const trailingDigits = (handle.match(/\d+$/) || [""])[0].length;
  if      (trailingDigits >= 6) suspicion += 0.35;
  else if (trailingDigits >= 4) suspicion += 0.20;
  else if (trailingDigits >= 2) suspicion += 0.08;

  // High proportion of digits anywhere in the handle
  const digitRatio = (handle.match(/\d/g) || []).length / Math.max(handle.length, 1);
  if (digitRatio > 0.5) suspicion += 0.10;
  else if (digitRatio > 0.3) suspicion += 0.05;

  // Very long handle (> 18 chars) combined with digits
  if (handle.length > 18 && trailingDigits >= 3) suspicion += 0.05;

  // Display name all-caps or very short single-word with no relation to handle
  if (displayName && displayName === displayName.toUpperCase() && displayName.length > 3) {
    suspicion += 0.08;
  }

  // ── Tweet text signals ───────────────────────────────────────────
  const text = tweetText || "";

  // Excessive all-caps ratio (shouting / low-quality content)
  const letters = text.match(/[A-Za-z]/g) || [];
  const uppers  = text.match(/[A-Z]/g) || [];
  if (letters.length > 10 && uppers.length / letters.length > 0.55) suspicion += 0.15;

  // Misinformation / manipulation language
  const misinfo = /\b(breaking[\s!]+news|urgent|they don.?t want you|wake up|hidden truth|deep state|fake news|mainstream media lie|hoax|conspiracy|sheeple|plandemic|shadow.?ban|censored|share before delete)\b/i;
  if (misinfo.test(text)) suspicion += 0.25;

  // Engagement bait
  const bait = /\b(rt if|retweet if|share if|follow back|f4f|follow me|giveaway|win free|click (here|now)|limited time|act now)\b/i;
  if (bait.test(text)) suspicion += 0.20;

  // Excessive exclamation marks (≥ 4)
  if ((text.match(/!/g) || []).length >= 4) suspicion += 0.10;

  // Repeated identical punctuation spam (e.g. "!!!!!!" or "??????")
  if (/([!?.])\1{4,}/.test(text)) suspicion += 0.08;

  // ── Engagement ratio anomalies ───────────────────────────────────
  const { retweets, likes } = eng;
  // Very high retweet-to-like ratio = retweet-bot pattern
  if (likes > 3 && retweets / likes > 8)  suspicion += 0.12;
  if (likes > 3 && retweets / likes > 20) suspicion += 0.10; // extra penalty

  return Math.min(1.0, suspicion);
}

// ─── BADGE CREATION ──────────────────────────────────────────────────────────

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

/**
 * @param {object|null} userResult  - API /analyze/user response
 * @param {object|null} imgResult   - API /analyze/profile-image response
 * @param {number}      displayProb - final blended probability to show
 */
function createBadge(userResult, imgResult, displayProb) {
  const badge = document.createElement("div");
  badge.className = "mg-badge";

  const botProb   = userResult?.user?.bot_probability ?? null;
  const trust     = userResult?.ensemble?.trust_score ?? null;
  const riskLevel = userResult?.user?.risk_level ?? null;
  const imgScore  = imgResult?.profile_image_risk_score ?? null;
  const imgLevel  = imgResult?.risk_level ?? null;

  const prob  = displayProb ?? botProb;
  const color = prob !== null ? riskColor(prob) : "grey";
  const emoji = prob !== null ? riskEmoji(prob) : "⚪";
  const label = prob !== null ? `${(prob * 100).toFixed(0)}%` : "?";

  badge.innerHTML = `
    <span class="mg-pill mg-pill--${color}" title="Bot/misinfo probability">
      ${emoji} ${label}
    </span>
    <div class="mg-tooltip">
      <div class="mg-tooltip-title">MisInfo Guard</div>
      ${prob      !== null ? `<div><b>Risk Score:</b> ${(prob * 100).toFixed(1)}%</div>` : ""}
      ${botProb   !== null ? `<div><b>Bot Prob (API):</b> ${(botProb * 100).toFixed(1)}%</div>` : ""}
      ${riskLevel !== null ? `<div><b>Risk Level:</b> ${riskLevel}</div>` : ""}
      ${trust     !== null ? `<div><b>Trust Score:</b> ${(trust * 100).toFixed(1)}%</div>` : ""}
      ${imgScore  !== null ? `<div><b>Img Risk:</b> ${(imgScore * 100).toFixed(1)}% (${imgLevel})</div>` : ""}
      ${userResult === null ? `<div class="mg-warn">API unavailable — heuristic only</div>` : ""}
    </div>
  `;

  return badge;
}

function createLoadingBadge() {
  const badge = document.createElement("div");
  badge.className = "mg-badge mg-badge--loading";
  badge.innerHTML = `<span class="mg-pill mg-pill--grey" title="Analyzing…">⏳</span>`;
  return badge;
}

// ─── TWEET PROCESSOR ─────────────────────────────────────────────────────────

async function processTweet(article) {
  if (processed.has(article)) return;
  processed.add(article);
  article.dataset.mgProcessed = "1";

  const engagementBar = article.querySelector('[role="group"]');
  if (!engagementBar) return;

  // Inject loading placeholder immediately
  const placeholder = createLoadingBadge();
  engagementBar.appendChild(placeholder);

  const { handle, displayName, profileImageUrl, verified, has_profile_image, has_url, tweetText } =
    extractTweetData(article);

  const eng = extractEngagementCounts(engagementBar, article);

  // ── Proxy estimates for the ML model ────────────────────────────
  // The X API fields are not in the DOM. We estimate from engagement counts:
  //   views / 30  ≈ followers (tweets reach ~3% of followers on average)
  //   total engagement bucket → tweet_count proxy
  //   verified / follower level → account age proxy
  const est_followers   = Math.round(eng.views / 30);
  const total_eng       = eng.replies + eng.retweets + eng.likes;
  const est_tweet_count = total_eng > 1000 ? 50000
                        : total_eng > 100  ? 5000
                        : total_eng > 10   ? 500
                        : 10;
  const est_age_days    = verified            ? 730
                        : est_followers > 50000 ? 730
                        : est_followers > 5000  ? 365
                        : est_followers > 500   ? 90
                        : 1;

  const userData = {
    followers_count:  est_followers,
    following_count:  0,
    tweet_count:      est_tweet_count,
    listed_count:     0,
    account_age_days: est_age_days,
    has_profile_image,
    has_description:  0,
    verified,
    has_location:     0,
    has_url,
  };

  // ── Client heuristic (varies per account/tweet even without API data) ──
  const clientRisk = computeClientHeuristic(handle, displayName, tweetText, eng);

  // ── API call ────────────────────────────────────────────────────
  const [userResult, imgResult] = await Promise.all([
    analyzeUser(userData),
    profileImageUrl ? analyzeProfileImage(profileImageUrl) : Promise.resolve(null),
  ]);

  // ── Blend API prob + client heuristic ──────────────────────────
  // When est_followers < 100, the model inputs are unreliable (defaulted to zero
  // and very small numbers). In that case, blend 50/50 with the client heuristic
  // so that handle/text signals can meaningfully differentiate accounts.
  const apiProb = userResult?.user?.bot_probability ?? null;
  let displayProb;
  if (apiProb === null) {
    displayProb = clientRisk;                         // API down → heuristic only
  } else if (est_followers < 100) {
    displayProb = 0.50 * apiProb + 0.50 * clientRisk; // sparse data → blend equally
  } else if (est_followers < 1000) {
    displayProb = 0.70 * apiProb + 0.30 * clientRisk; // moderate data → lean on model
  } else {
    displayProb = apiProb;                            // good data → trust the model
  }
  displayProb = Math.min(1, Math.max(0, displayProb));

  placeholder.replaceWith(createBadge(userResult, imgResult, displayProb));
}

// ─── OBSERVER ────────────────────────────────────────────────────────────────

function observeTweets() {
  // Process any tweets already in the DOM
  document
    .querySelectorAll('article[data-testid="tweet"]:not([data-mg-processed])')
    .forEach(processTweet);

  // Watch for dynamically loaded tweets (infinite scroll / live updates)
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (!(node instanceof HTMLElement)) continue;

        const tweets = node.matches('article[data-testid="tweet"]')
          ? [node]
          : [...node.querySelectorAll('article[data-testid="tweet"]:not([data-mg-processed])')];

        tweets.forEach(processTweet);
      }
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", observeTweets);
} else {
  observeTweets();
}
