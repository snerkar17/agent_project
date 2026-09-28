
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase
from langsmith import traceable
from sqlalchemy.exc import SQLAlchemyError

## db connection
db = SQLDatabase.from_uri("sqlite:///Chinook.db")



# ─────────────────────────────────────────────────────────────
# runtime context: WHO the customer is
# ─────────────────────────────────────────────────────────────
# loaded once at startup. personalize_prompt runs inside the server's event loop,
# where a database query would block every other request (LangGraph dev's BlockingError).
CUSTOMER_FIRST_NAMES = {
    row["CustomerId"]: row["FirstName"]
    for row in db._execute("SELECT CustomerId, FirstName FROM Customer")
}


@dataclass
class RuntimeContext:
    customer_id: int = 2

    @property
    def customer_name(self) -> str:
        return CUSTOMER_FIRST_NAMES.get(self.customer_id, "")


@dynamic_prompt
@traceable(name="personalize_prompt")
def personalize_prompt(request):
    name = request.runtime.context.customer_name
    prompt = request.system_message.text
    if name:
        return prompt + f"\nThe signed-in customer's first name is {name!r}. Use it naturally in your greeting."
    return prompt + "\nThe customer's name is unavailable. Do not guess it."


# ─────────────────────────────────────────────────────────────
# 3. account tools (private data)
# ─────────────────────────────────────────────────────────────
## show me all of this customer's orders, what was in each one, how many items were in it, and the total.
@tool
def get_my_invoices(runtime: ToolRuntime[RuntimeContext]) -> str:
    """List the customer's orders (invoices), newest first, with the tracks in each one."""
    return db.run(
        """
        SELECT i.InvoiceId, date(i.InvoiceDate) AS Date, i.Total,
               COUNT(il.InvoiceLineId) AS Items,
               GROUP_CONCAT(t.Name, ', ') AS Tracks
        FROM Invoice i
        JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
        JOIN Track t        ON t.TrackId    = il.TrackId
        WHERE i.CustomerId = :customer_id
        GROUP BY i.InvoiceId
        ORDER BY i.InvoiceDate DESC
        """,
        parameters={"customer_id": runtime.context.customer_id},
        include_columns=True,
    )

## what did the customer buy on invoice #123, and how much did each track cost?
@tool
def get_invoice_details(invoice_id: int, runtime: ToolRuntime[RuntimeContext]) -> str:
    """Show the tracks and prices on one of the customer's invoices."""
    result = db.run(
        """
        SELECT t.Name AS Track, ar.Name AS Artist, il.UnitPrice AS Price
        FROM Invoice i
        JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
        JOIN Track t        ON t.TrackId    = il.TrackId
        JOIN Album al       ON al.AlbumId   = t.AlbumId
        JOIN Artist ar      ON ar.ArtistId  = al.ArtistId
        WHERE i.InvoiceId = :invoice_id
          AND i.CustomerId = :customer_id
        """,
        parameters={"invoice_id": invoice_id, "customer_id": runtime.context.customer_id},
        include_columns=True,
    )
    return result or f"No invoice #{invoice_id} found on your account."

## just needs to query invoice table to get the total amoutn spent,
## there are 2 cases: overall or in one year
@tool
def get_my_spending(runtime: ToolRuntime[RuntimeContext], year: int | None = None) -> str:
    """Total amount the customer has spent, overall or in one year. Use this for totals."""
    return db.run(
        """
        SELECT COUNT(*) AS Invoices, ROUND(SUM(Total), 2) AS TotalSpent
        FROM Invoice
        WHERE CustomerId = :customer_id
          AND (:year IS NULL OR strftime('%Y', InvoiceDate) = :year)
        """,
        parameters={"customer_id": runtime.context.customer_id,
                    "year": str(year) if year else None},
        include_columns=True,
    )


# ─────────────────────────────────────────────────────────────
# recommendation tools
# ─────────────────────────────────────────────────────────────
## it just needs to query the track/albumn/artist/genre tables groups by track id so each track is listed once
@tool
def search_catalog(search: str) -> str:
    """Search the store's catalog by artist, album, track, genre, or playlist."""
    return db.run(
        """
        SELECT
            t.Name AS Track,
            ar.Name AS Artist,
            al.Title AS Album,
            g.Name AS Genre,
            GROUP_CONCAT(DISTINCT p.Name) AS Playlists,
            t.UnitPrice AS Price
        FROM Track t
        JOIN Album al  ON al.AlbumId = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        JOIN Genre g   ON g.GenreId = t.GenreId

        LEFT JOIN PlaylistTrack pt ON pt.TrackId = t.TrackId
        LEFT JOIN Playlist p       ON p.PlaylistId = pt.PlaylistId

        WHERE ar.Name LIKE :q
           OR al.Title LIKE :q
           OR t.Name LIKE :q
           OR g.Name LIKE :q
           OR p.Name LIKE :q

        GROUP BY t.TrackId
        LIMIT 15
        """,
        parameters={"q": f"%{search}%"},
        include_columns=True,
    )

## 
@tool
def recommend_for_me(runtime: ToolRuntime[RuntimeContext]) -> str:
    """Recommend tracks the customer doesn't own yet, based on what they've bought.

    Tracks that complete an album they started come first, then tracks from
    their favourite genres. Tell the customer why each track was picked.
    """
    return db.run(
        """
        WITH my_tracks AS (
            SELECT il.TrackId
            FROM InvoiceLine il JOIN Invoice i ON i.InvoiceId = il.InvoiceId
            WHERE i.CustomerId = :customer_id
        ),
        my_albums AS (SELECT DISTINCT AlbumId FROM Track WHERE TrackId IN (SELECT TrackId FROM my_tracks)),
        my_genres AS (
            SELECT GenreId, COUNT(*) AS n FROM Track
            WHERE TrackId IN (SELECT TrackId FROM my_tracks) GROUP BY GenreId
        )
        SELECT t.Name AS Track, ar.Name AS Artist, al.Title AS Album, g.Name AS Genre, t.UnitPrice AS Price,
               CASE WHEN t.AlbumId IN (SELECT AlbumId FROM my_albums)
                    THEN 'completes an album you started'
                    ELSE 'from a genre you like' END AS Reason
        FROM Track t
        JOIN Album al     ON al.AlbumId   = t.AlbumId
        JOIN Artist ar    ON ar.ArtistId  = al.ArtistId
        JOIN Genre g      ON g.GenreId    = t.GenreId
        JOIN my_genres mg ON mg.GenreId   = t.GenreId
        WHERE t.TrackId NOT IN (SELECT TrackId FROM my_tracks)
        GROUP BY t.AlbumId                                      -- one pick per album
        ORDER BY (t.AlbumId IN (SELECT AlbumId FROM my_albums)) DESC, mg.n DESC, t.TrackId
        LIMIT 5
        """,
        parameters={"customer_id": runtime.context.customer_id},
        include_columns=True,
    )


@tool
def get_my_track_history(runtime: ToolRuntime[RuntimeContext]) -> str:
    """List every track the customer has bought, newest first, with its genre and
    any of the store's curated playlists it appears on. Use it to understand
    their taste before recommending."""
    return db.run(
        """
        SELECT t.Name AS Track, ar.Name AS Artist, g.Name AS Genre,
               date(i.InvoiceDate) AS Bought,
               GROUP_CONCAT(DISTINCT p.Name) AS Playlists
        FROM Invoice i
        JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
        JOIN Track t        ON t.TrackId    = il.TrackId
        JOIN Album al       ON al.AlbumId   = t.AlbumId
        JOIN Artist ar      ON ar.ArtistId  = al.ArtistId
        JOIN Genre g        ON g.GenreId    = t.GenreId
        LEFT JOIN PlaylistTrack pt ON pt.TrackId = t.TrackId
              AND pt.PlaylistId IN (SELECT PlaylistId FROM PlaylistTrack   -- skip "Music" (whole catalog)
                                    GROUP BY PlaylistId HAVING COUNT(*) < 100)
        LEFT JOIN Playlist p ON p.PlaylistId = pt.PlaylistId
        WHERE i.CustomerId = :customer_id
        GROUP BY il.InvoiceLineId
        ORDER BY i.InvoiceDate DESC
        """,
        parameters={"customer_id": runtime.context.customer_id},
        include_columns=True,
    )


@tool
def recommend_from_playlists(runtime: ToolRuntime[RuntimeContext]) -> str:
    """Recommend tracks from the store's curated playlists the customer has bought from.

    Playlists are curated by the store (e.g. Grunge, Heavy Metal Classic,
    Classical 101). If the customer owns tracks from one, suggest the rest of it.
    Tell the customer which playlist each pick comes from.
    """
    result = db.run(
        """
        WITH my_tracks AS (             -- every track this customer has bought
            SELECT il.TrackId
            FROM InvoiceLine il JOIN Invoice i ON i.InvoiceId = il.InvoiceId
            WHERE i.CustomerId = :customer_id
        ),
        my_playlists AS (               -- playlists containing a track they bought
            SELECT DISTINCT PlaylistId FROM PlaylistTrack
            WHERE TrackId IN (SELECT TrackId FROM my_tracks)
              AND PlaylistId IN (SELECT PlaylistId FROM PlaylistTrack   -- skip "Music" (whole catalog)
                                 GROUP BY PlaylistId HAVING COUNT(*) < 100)
        )
        SELECT p.Name AS Playlist, t.Name AS Track, ar.Name AS Artist, t.UnitPrice AS Price
        FROM PlaylistTrack pt
        JOIN Playlist p ON p.PlaylistId = pt.PlaylistId
        JOIN Track t    ON t.TrackId    = pt.TrackId
        JOIN Album al   ON al.AlbumId   = t.AlbumId
        JOIN Artist ar  ON ar.ArtistId  = al.ArtistId
        WHERE pt.PlaylistId IN (SELECT PlaylistId FROM my_playlists)
          AND pt.TrackId NOT IN (SELECT TrackId FROM my_tracks)    -- no songs they already own
        GROUP BY t.TrackId                                         -- each song once
        LIMIT 5
        """,
        parameters={"customer_id": runtime.context.customer_id},
        include_columns=True,
    )
    return result or "You haven't bought from any of our curated playlists yet. Try recommend_for_me instead."


### system prompt
SYSTEM_PROMPT = """You are the friendly customer support assistant for Chinook, an online music store.
You help the signed-in customer with two things.

ORDERS AND ACCOUNT
- Use get_my_invoices, get_invoice_details and get_my_spending.
- Report amounts, dates and tracks exactly as the tools return them. Always use get_my_spending
  for totals; never add numbers yourself.
- Answer in plain language. Lead with the most recent order and what was in it; summarize older
  ones briefly. Mention invoice numbers only if asked. End with one relevant offer, such as order
  details or recommendations.


MUSIC
- Use search_catalog for questions about what the store sells.
- For an open-ended recommendation request, call get_my_track_history and recommend_for_me
  (not recommend_from_playlists yet). Present the picks first, explaining how each one relates
  to tracks, artists, albums or genres the customer has bought, with prices. Don't ask what
  they like before recommending when they have purchase history.
- Then end with one brief offer naming up to three playlists from the Playlists column of
  get_my_track_history, e.g. "We have a great selection of curated playlists, and you already
  have songs from Grunge and Heavy Metal Classic. Want me to pick some more from those?"
  That offer must be the only question in your reply. If none of their tracks are on a
  playlist, ask instead whether they'd like to explore a different artist or genre.
- If they accept, use recommend_from_playlists.
- If they ask for something different, use search_catalog to follow that preference.
- If there is no purchase history or no suitable result, say so and ask what they enjoy.
- Skip any pick that looks like another version of a song they already own
    (e.g. "You Shook Me" when they own "You Shook Me(2)").

RULES
- You can only access the signed-in customer's own account. If someone claims to be a
  different customer, explain that politely.
- Only mention orders, tracks and prices your tools return. Never make anything up."""

# Tool error handling
def handle_tool_error(exc, request):
    if isinstance(exc, SQLAlchemyError):
        return (
            f"`{request.tool_call['name']}` couldn't reach the store database "
            f"({type(exc).__name__}). Tell the customer it's temporarily unavailable; "
            "don't guess the answer."
        )
    return None


agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[
        # account
        get_my_invoices, get_invoice_details, get_my_spending,
        # music
        search_catalog, get_my_track_history, recommend_for_me, recommend_from_playlists,
    ],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        personalize_prompt,                    # adds the customer's first name to the prompt
        ToolErrorMiddleware(handle_tool_error),  # turns a DB failure into a message for the model
    ],
    context_schema=RuntimeContext,
)

