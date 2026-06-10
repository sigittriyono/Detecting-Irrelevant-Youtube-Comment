import re
from datetime import datetime


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def format_number(n: int) -> str:
    """Format large numbers with dot separator (Indonesian style)."""
    return f"{n:,}".replace(",", ".")


def format_date(iso_date: str) -> str:
    """Format ISO date string to readable Indonesian-style date."""
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return dt.strftime("%d %B %Y")
    except Exception:
        return iso_date


def truncate_text(text: str, max_len: int = 120) -> str:
    """Truncate long text with ellipsis."""
    return text if len(text) <= max_len else text[:max_len].rstrip() + "…"


def get_moderation_status(irrelevant_ratio: float) -> dict:
    """Return moderation status based on irrelevant comment ratio."""
    if irrelevant_ratio < 0.10:
        return {
            "emoji": "🟢",
            "label": "Low",
            "status": "Low level of irrelevant comments detected.",
            "recommendation": (
                "This video has a healthy comment section with very few unrelated comments. "
                "No immediate moderation action is required."
            ),
            "color": "#22c55e",
            "bg": "#f0fdf4",
        }
    elif irrelevant_ratio < 0.30:
        return {
            "emoji": "🟡",
            "label": "Moderate",
            "status": "Moderate level of irrelevant comments detected.",
            "recommendation": (
                "This video contains a noticeable number of unrelated comments. "
                "Manual moderation is recommended to improve discussion quality."
            ),
            "color": "#f59e0b",
            "bg": "#fffbeb",
        }
    else:
        return {
            "emoji": "🔴",
            "label": "High",
            "status": "High level of irrelevant comments detected.",
            "recommendation": (
                "This video has a significant amount of spam or off-topic comments. "
                "Immediate moderation is strongly recommended to maintain content quality."
            ),
            "color": "#ef4444",
            "bg": "#fef2f2",
        }
