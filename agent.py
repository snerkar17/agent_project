from dotenv import load_dotenv
from dataclasses import dataclass
from collections import Counter

load_dotenv(override=True)

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_community.utilities import SQLDatabase


# --------------------------------------------------
# Database
# --------------------------------------------------

# Live backend dependency.
# Keep this in Python, not in Studio runtime context.
db = SQLDatabase.from_uri("sqlite:///Chinook.db")


# --------------------------------------------------
# Runtime Context
# --------------------------------------------------

@dataclass
class RuntimeContext:
    customer_id: int = 1


# --------------------------------------------------
# Helper 1: Get customer's purchased tracks
# --------------------------------------------------

def get_customer_purchases(
    customer_id: int,
) -> list[dict]:

    query = """
    SELECT
        t.TrackId,
        t.Name AS Track,
        ar.Name AS Artist,
        g.Name AS Genre,
        i.InvoiceDate AS PurchaseDate
    FROM Invoice i
    JOIN InvoiceLine il
        ON i.InvoiceId = il.InvoiceId
    JOIN Track t
        ON il.TrackId = t.TrackId
    JOIN Album al
        ON t.AlbumId = al.AlbumId
    JOIN Artist ar
        ON al.ArtistId = ar.ArtistId
    JOIN Genre g
        ON t.GenreId = g.GenreId
    WHERE i.CustomerId = :customer_id
    ORDER BY i.InvoiceDate DESC, il.InvoiceLineId DESC
    """

    return list(
        db._execute(
            query,
            fetch="all",
            parameters={
                "customer_id": customer_id,
            },
        )
    )


# --------------------------------------------------
# Helper 2: Summarize preferences
# --------------------------------------------------

def summarize_preferences(
    purchases: list[dict],
) -> dict:

    artist_counts = Counter()
    genre_counts = Counter()

    for purchase in purchases:
        artist_counts[purchase["Artist"]] += 1
        genre_counts[purchase["Genre"]] += 1

    return {
        "preferred_artists": dict(artist_counts),
        "preferred_genres": dict(genre_counts),
    }


# --------------------------------------------------
# Helper 3: Get full track catalog
# --------------------------------------------------

def get_track_catalog() -> list[dict]:

    query = """
    SELECT
        t.TrackId,
        t.Name AS Track,
        ar.Name AS Artist,
        g.Name AS Genre
    FROM Track t
    JOIN Album al
        ON t.AlbumId = al.AlbumId
    JOIN Artist ar
        ON al.ArtistId = ar.ArtistId
    JOIN Genre g
        ON t.GenreId = g.GenreId
    """

    return list(
        db._execute(
            query,
            fetch="all",
        )
    )


# --------------------------------------------------
# Helper 4: Find unseen matching tracks
# --------------------------------------------------

def find_candidate_tracks(
    catalog: list[dict],
    purchases: list[dict],
    preferences: dict,
) -> list[dict]:

    owned_track_ids = {
        purchase["TrackId"]
        for purchase in purchases
    }

    preferred_artists = preferences["preferred_artists"]
    preferred_genres = preferences["preferred_genres"]

    candidates = []

    for track in catalog:

        # Never recommend tracks already purchased
        if track["TrackId"] in owned_track_ids:
            continue

        artist_match = track["Artist"] in preferred_artists
        genre_match = track["Genre"] in preferred_genres

        if artist_match or genre_match:
            candidates.append(track)

    return candidates


# --------------------------------------------------
# Helper 5: Rank candidates
# --------------------------------------------------

def rank_candidates(
    candidates: list[dict],
    preferences: dict,
    limit: int = 5,
) -> list[dict]:

    artist_counts = preferences["preferred_artists"]
    genre_counts = preferences["preferred_genres"]

    ranked = []

    for track in candidates:

        artist = track["Artist"]
        genre = track["Genre"]

        score = 0

        # Artist match matters more
        if artist in artist_counts:
            score += 2 * artist_counts[artist]

        # Genre match also contributes
        if genre in genre_counts:
            score += genre_counts[genre]

        ranked.append({
            **track,
            "score": score,
        })

    ranked.sort(
        key=lambda track: track["score"],
        reverse=True,
    )

    return ranked[:limit]


# --------------------------------------------------
# Purchase History Tool
# --------------------------------------------------

@tool
def get_purchase_history(
    runtime: ToolRuntime[RuntimeContext],
) -> list[dict]:
    """
    Get the authenticated customer's purchased tracks,
    including artist, genre, and purchase date.
    Results are ordered from most recent purchase to oldest.
    """

    customer_id = runtime.context.customer_id

    return get_customer_purchases(customer_id)


# --------------------------------------------------
# Recommendation Tool
# --------------------------------------------------

@tool
def recommend_tracks(
    runtime: ToolRuntime[RuntimeContext],
    limit: int = 5,
) -> list[dict]:
    """
    Recommend unseen tracks to the authenticated customer
    based on artists and genres from their purchase history.
    """

    # Trusted customer identity comes from runtime context
    customer_id = runtime.context.customer_id

    purchases = get_customer_purchases(
        customer_id,
    )

    preferences = summarize_preferences(
        purchases,
    )

    catalog = get_track_catalog()

    candidates = find_candidate_tracks(
        catalog,
        purchases,
        preferences,
    )

    recommendations = rank_candidates(
        candidates,
        preferences,
        limit,
    )

    return recommendations


# --------------------------------------------------
# Agent
# --------------------------------------------------

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_purchase_history, recommend_tracks],
    system_prompt=(
        "You are a customer support agent for a music store. "
        "Help the authenticated customer discover music based on "
        "their previous purchases. "
        "Only recommend tracks returned by the recommendation tool. "
        "Briefly explain why each recommendation fits their preferences."
    ),
    context_schema=RuntimeContext,
)


# --------------------------------------------------
# Test Run
# --------------------------------------------------

if __name__ == "__main__":

    question = "Recommend 5 tracks for me."

    for step in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        context=RuntimeContext(
            customer_id=1,
        ),
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()