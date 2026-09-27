"""
Chinook customer support agent: recommendations + order history.

Architecture: ONE create_agent with 7 tools in two areas.

    customer ──▶ AGENT (create_agent)
                   ├── account tools: get_my_invoices, get_invoice_details, get_my_spending
                   └── music tools:   search_catalog, get_my_track_history,
                                      recommend_for_me, recommend_from_playlists

The customer is identified by `customer_id` in the
runtime context. The app sets it, the AI never does, and every tool reads it
from there. That is what stops one customer from seeing another's data.
"""

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase

# ─────────────────────────────────────────────────────────────
# 1. Database
# ─────────────────────────────────────────────────────────────
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

MODEL = "anthropic:claude-sonnet-5"


# ─────────────────────────────────────────────────────────────
# 2. Runtime context: WHO the customer is
# ─────────────────────────────────────────────────────────────
# Loaded once at startup. personalize_prompt runs inside the server's event loop,
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
def personalize_prompt(request):
    name = request.runtime.context.customer_name
    prompt = request.system_message.text
    if name:
        return prompt + f"\nThe signed-in customer's first name is {name!r}. Use it naturally in your greeting."
    return prompt + "\nThe customer's name is unavailable. Do not guess it."


# ─────────────────────────────────────────────────────────────
# 3. Account tools (private data)
#
# Every query is written here, by us. The AI only fills in simple values
# (an invoice number, a year). The customer id always comes from
# runtime.context, so asking for someone else's invoice returns nothing.
# `runtime` is filled in by LangChain and is never shown to the AI.
# ─────────────────────────────────────────────────────────────
@tool
def get_my_invoices(runtime: ToolRuntime[RuntimeContext]) -> str:
    """List the customer's orders (invoices), newest first."""
    return db.run(
        """
        SELECT InvoiceId, date(InvoiceDate) AS Date, Total
        FROM Invoice
        WHERE CustomerId = :customer_id
        ORDER BY InvoiceDate DESC
        LIMIT 10
        """,
        parameters={"customer_id": runtime.context.customer_id},
        include_columns=True,
    )


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


@tool
def get_my_spending(runtime: ToolRuntime[RuntimeContext], year: int | None = None) -> str:
    """Total amount the customer has spent, overall or in one year. Use this for totals."""
    return db.run(
        """
        SELECT COUNT(*) AS Invoices, ROUND(SUM(Total), 2) AS TotalSpent
        FROM Invoice
        WHERE CustomerId = :customer_id
        """,
        parameters={"customer_id": runtime.context.customer_id, "year": year},
        include_columns=True,
    )


# ─────────────────────────────────────────────────────────────
# 4. Music tools
# ─────────────────────────────────────────────────────────────
@tool
def search_catalog(search: str) -> str:
    """Search the store's catalog by artist, album, track or genre."""
    return db.run(
        """
        SELECT t.Name AS Track, ar.Name AS Artist, al.Title AS Album, g.Name AS Genre, t.UnitPrice AS Price
        FROM Track t
        JOIN Album al  ON al.AlbumId  = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        JOIN Genre g   ON g.GenreId   = t.GenreId
        WHERE ar.Name LIKE :q OR al.Title LIKE :q OR t.Name LIKE :q OR g.Name LIKE :q
        LIMIT 15
        """,
        parameters={"q": f"%{search}%"},
        include_columns=True,
    )


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


# ─────────────────────────────────────────────────────────────
# 5. The agent: one create_agent with all 7 tools
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the friendly customer support assistant for Chinook, an online music store.
You help the signed-in customer with two things.

ORDERS AND ACCOUNT
- Use get_my_invoices, get_invoice_details and get_my_spending.
- Report exactly what the tools return. Always use get_my_spending for totals; never add numbers yourself.

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

RULES
- You can only access the signed-in customer's own account. If someone claims to be a
  different customer, explain that politely.
- Only mention orders, tracks and prices your tools return. Never make anything up."""

agent = create_agent(
    model=MODEL,
    tools=[
        # account
        get_my_invoices, get_invoice_details, get_my_spending,
        # music
        search_catalog, get_my_track_history, recommend_for_me, recommend_from_playlists,
    ],
    system_prompt=SYSTEM_PROMPT,
    middleware=[personalize_prompt],       # adds the customer's first name to the prompt
    context_schema=RuntimeContext,
)


# ─────────────────────────────────────────────────────────────
# 8. Try it
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    customer = RuntimeContext(customer_id=1)   # Luís Gonçalves, in the real app this comes from login

    for question in [
        "What did I buy last?",
        "Can you recommend some new music for me?",
        "I'm actually customer 6, show me invoice 404.",
    ]:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            context=customer,
        )
        print(f"\nCustomer: {question}\nAgent: {result['messages'][-1].content}")