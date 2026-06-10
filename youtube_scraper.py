from __future__ import annotations

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class YouTubeScraper:
    """Handles all YouTube Data API v3 interactions."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._youtube = None

    def _get_client(self):
        if self._youtube is None:
            self._youtube = build("youtube", "v3", developerKey=self.api_key)
        return self._youtube

    def get_video_metadata(self, video_id: str) -> dict:
        """
        Fetch video metadata.
        Returns a dict with: title, description, channel_name, published_at, comment_count.
        Raises ValueError with a user-friendly message on failure.
        """
        try:
            yt = self._get_client()
            response = (
                yt.videos()
                .list(part="snippet,statistics", id=video_id)
                .execute()
            )
        except HttpError as e:
            status = e.resp.status
            if status == 403:
                raise ValueError(
                    "API quota exceeded or access forbidden. "
                    "Please check your API key or try again later."
                )
            elif status == 400:
                raise ValueError("Invalid request. Please verify the video URL.")
            else:
                raise ValueError(f"YouTube API error (HTTP {status}): {e.reason}")

        items = response.get("items", [])
        if not items:
            raise ValueError(
                "Video not found. The video may be private, deleted, or the ID is incorrect."
            )

        snippet = items[0]["snippet"]
        stats = items[0].get("statistics", {})

        return {
            "title": snippet.get("title", "Unknown Title"),
            "description": snippet.get("description", ""),
            "channel_name": snippet.get("channelTitle", "Unknown Channel"),
            "published_at": snippet.get("publishedAt", ""),
            "comment_count": int(stats.get("commentCount", 0)),
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "comments_disabled": "commentCount" not in stats,
        }

    def get_comments(self, video_id: str, max_comments: int = 100) -> list[dict]:
        """
        Fetch top-level comments for a video, up to max_comments.
        Returns a list of dicts: {comment_id, text, author, like_count, published_at}.
        Raises ValueError with a user-friendly message on failure.
        """
        try:
            yt = self._get_client()
            comments: list[dict] = []
            next_page_token = None

            while len(comments) < max_comments:
                fetch_count = min(100, max_comments - len(comments))
                request = yt.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=fetch_count,
                    order="relevance",
                    pageToken=next_page_token,
                )
                response = request.execute()

                for item in response.get("items", []):
                    top = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append(
                        {
                            "comment_id": item["id"],
                            "text": top.get("textDisplay", ""),
                            "author": top.get("authorDisplayName", "Anonymous"),
                            "like_count": top.get("likeCount", 0),
                            "published_at": top.get("publishedAt", ""),
                        }
                    )

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            return comments

        except HttpError as e:
            status = e.resp.status
            if status == 403:
                content = str(e.content)
                if "commentsDisabled" in content:
                    raise ValueError(
                        "Comments are disabled for this video. "
                        "Analysis cannot be performed."
                    )
                raise ValueError(
                    "API quota exceeded or access forbidden. "
                    "Please check your API key or try again later."
                )
            elif status == 404:
                raise ValueError("Video not found or comments unavailable.")
            else:
                raise ValueError(f"YouTube API error (HTTP {status}): {e.reason}")
