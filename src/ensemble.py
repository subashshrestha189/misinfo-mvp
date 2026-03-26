# src/ensemble.py
from __future__ import annotations
from typing import Dict, Any


def compute_heuristic_score(user: Dict[str, Any]) -> float:
    """
    Simple heuristic score in [0,1].
    Higher = more trustworthy / less risky.
    Uses basic profile signals.
    """
    score = 0.5

    followers = user.get("followers_count", 0) or 0
    account_age_days = user.get("account_age_days", 0) or 0
    has_profile_image = bool(user.get("has_profile_image", 0))
    has_description = bool(user.get("has_description", 0))
    verified = bool(user.get("verified", 0))

    # Older accounts are slightly more trusted
    if account_age_days > 365:
        score += 0.10
    if account_age_days > 730:
        score += 0.05  # > 2 years

    # Basic completeness
    if has_profile_image:
        score += 0.05
    if has_description:
        score += 0.05

    # Verified badge gives a bigger bump
    if verified:
        score += 0.15

    # Very new + very low follower accounts → more suspicious
    if account_age_days < 30 and followers < 20:
        score -= 0.15

    return float(max(0.0, min(1.0, score)))


def combine_scores(bot_probability: float, heuristic_score: float) -> Dict[str, Any]:
    """
    Combine bot probability and heuristic score into a single trust score.
    bot_probability:  [0,1] – probability that account is a bot
    heuristic_score:  [0,1] – hand-crafted trust score (higher = more legitimate)
    """
    w_bot = 0.6
    w_heur = 0.4

    bot_trust = 1.0 - bot_probability  # invert: 1 = human-like, 0 = strong bot

    trust_score = w_bot * float(bot_trust) + w_heur * float(heuristic_score)
    trust_score = max(0.0, min(1.0, trust_score))

    if trust_score >= 0.75:
        trust_level = "High Trust"
    elif trust_score >= 0.5:
        trust_level = "Moderate Trust"
    else:
        trust_level = "Low Trust"

    return {
        "trust_score": round(trust_score, 3),
        "trust_level": trust_level,
        "weights": {"bot": w_bot, "heuristics": w_heur},
    }
