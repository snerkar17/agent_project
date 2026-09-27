"""A small MCP server that looks up iTunes preview links."""
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("music-previews")


@mcp.tool()
async def find_preview(track: str, artist: str) -> dict:
    """Find an iTunes preview for a song. Returns links, not downloaded audio."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://itunes.apple.com/search",
                params={"term": f"{artist} {track}", "entity": "song", "limit": 5},
            )
            response.raise_for_status()
            results = response.json().get("results", [])
    except (httpx.HTTPError, ValueError):
        return {"found": False, "reason": "Preview search is temporarily unavailable."}

    for song in results:
        # Do not attach a cover version or a different song returned by search.
        if (song.get("trackName", "").strip().casefold() == track.strip().casefold()
                and song.get("artistName", "").strip().casefold() == artist.strip().casefold()
                and song.get("previewUrl")):
            return {
                "found": True,
                "track": song["trackName"],
                "artist": song["artistName"],
                "preview_url": song["previewUrl"],
                "credit": "Preview provided courtesy of iTunes",
            }
    return {"found": False}


if __name__ == "__main__":
    mcp.run(transport="stdio")
