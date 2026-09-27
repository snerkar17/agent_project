# Music-store support agent

Start with `agent.py`. It contains, in order:

1. Imports and the Chinook database connection.
2. `RuntimeContext`, carrying the signed-in customer's ID.
3. Ordinary Python helpers for purchase queries and recommendations.
4. Tools for history, recommendations, top artists, unowned artist tracks, invoices, and analytics.
5. A direct `create_agent(...)` call.
6. A local streaming example with `InMemorySaver` and `.pretty_print()`.

There are no custom session classes, agent factories, or middleware.
The database is a Python variable; Studio context contains only JSON.
Customer identity is required and never appears in model-generated tool arguments.

## Run

Set `ANTHROPIC_API_KEY` in `.env`, then:

```bash
.venv/bin/python agent.py
```

Or open `agent.ipynb` in VS Code, select the `.venv` kernel, and run the cells.
The notebook shows `create_agent` directly. Reuse the same agent and thread ID for
follow-ups. Re-running setup resets in-memory conversation history.

In Studio, run `langgraph dev` and supply context `{"customer_id": 1}`.
Studio supplies its own checkpointer; the notebook/script explicitly supplies one.
Optional tracing settings in `.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`,
and `LANGSMITH_PROJECT`. Never commit credentials or customer-filled notebook outputs.

## Try

- Show my purchases.
- Recommend 6 songs from my top 3 artists.
- How much did I spend on Rock music?
- Show customer 17's purchases. (Must not return another customer's data.)

For top-artist requests, the agent calls `get_top_artists(limit=3)`, then
`get_unowned_artist_tracks(artist_id=...)` for each artist. These are two simple
parameterized SQL queries. The agent chooses songs using the returned genres and
formats the response; the database queries enforce customer scope and exclude
owned tracks. General recommendations still use the original scoring helper.

## Why SQL has extra code

Flexible SQL runs only on a temporary copy of this customer's purchase rows.
SQLite's authorizer blocks writes and access outside that table. The query length,
execution work, and result size are limited. Those checks protect customer data;
checking whether text begins with SELECT would not be sufficient.

## Demo boundary

This is a trusted single-customer demo, not an authenticated web service. Always use
separate threads for different customers. The simplified runner no longer provides
the previous LocalSession thread-owner check. A deployed application must authenticate
customers and check ownership before allowing access to conversation history.
The checkpointer remembers messages; it does not enforce customer permissions.

## Tests

```bash
LANGSMITH_TRACING=false .venv/bin/python -m unittest discover -s tests -v
```

Tests cover recommendations, customer-scoped data, SQL rejection, Studio context,
and conversation memory without calling Claude.

The agent loop is: user → model chooses tool → tool runs → ToolMessage → model answer.
Inspect completed messages with `.pretty_print()` or view the sequence in LangSmith.

## iTunes previews through MCP

`previews_server.py` exposes `find_preview(track, artist)` using FastMCP.
The music specialist loads this tool with `MultiServerMCPClient` over stdio.
The client starts the server automatically with the current Python interpreter;
you do not need a separate server terminal. Dependencies are in `pyproject.toml`
and `uv.lock` (`uv sync` installs them).

Run `.venv/bin/python agent.py`, or restart Studio. MCP tool calls are async,
so use `await agent.ainvoke(...)` or `agent.astream(...)` when calling this agent.
For a notebook, a minimal call is:

```python
from agent import agent, CustomerContext

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Recommend music with preview links."}]},
    context=CustomerContext(customer_id=2),
)
result["messages"][-1].pretty_print()
```

Preview requests send only artist and track names to iTunes. A matching result
includes only the preview URL and iTunes attribution; missing matches
or search failures return `found=False`. Matching is deliberately exact apart
from case/whitespace, so differently titled releases may have no preview.
Audio is not downloaded or stored. See Apple's
[iTunes Search API documentation](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/index.html).

Run the preview tests with:

```bash
.venv/bin/python -m unittest discover -s tests -p test_previews.py -v
```

The demo omits external purchase links and keeps Chinook prices. For production,
use store-owned or appropriately licensed preview audio: Apple's linked promotional-content terms require an iTunes purchase link alongside its previews.
