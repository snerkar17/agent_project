import unittest
from types import SimpleNamespace
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain.messages import AIMessage
from agent import RuntimeContext, db, SYSTEM_PROMPT
from agent import get_customer_purchases
from agent import query_purchases
from agent import recommend_tracks, get_invoice_details, tools
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


class TestModel(BaseChatModel):
    @property
    def _llm_type(self):
        return 'test-model'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='Hello'))])


class SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = db

    def setUp(self):
        self.context = RuntimeContext(customer_id=1)
        self.runtime = SimpleNamespace(context=self.context)
        self.invoice = get_customer_purchases(1)[0]['InvoiceId']
        self.foreign = self.db._execute('SELECT InvoiceId FROM Invoice WHERE CustomerId=2')[0]['InvoiceId']

    def test_identity_and_schema(self):
        for value in (0, -1, True, '1', None):
            with self.assertRaises(ValueError):
                RuntimeContext(customer_id=value)
        for tool in tools:
            properties = tool.tool_call_schema.model_json_schema()['properties']
            self.assertNotIn('customer_id', properties)
            self.assertNotIn('runtime', properties)

    def test_history_and_recommendations(self):
        purchases = get_customer_purchases(1)
        self.assertEqual(len({p['InvoiceId'] for p in purchases}), 7)
        self.assertEqual(purchases, sorted(purchases, key=lambda p: (p['PurchaseDate'], p['InvoiceLineId']), reverse=True))
        owned = {p['TrackId'] for p in purchases}
        recommendations = recommend_tracks.func(self.runtime, limit=3)
        self.assertEqual(len(recommendations), 3)
        self.assertFalse(owned.intersection(r['TrackId'] for r in recommendations))
        self.assertEqual(recommendations, recommend_tracks.func(self.runtime, limit=3))
        other = RuntimeContext(customer_id=99999)
        self.assertEqual(recommend_tracks.func(SimpleNamespace(context=other)), [])
        from collections import Counter
        artists = Counter(p['Artist'] for p in purchases)
        genres = Counter(p['Genre'] for p in purchases)
        for r in recommendations:
            self.assertEqual(r['score'], 2 * artists[r['Artist']] + genres[r['Genre']])
        from pydantic import ValidationError
        for limit in (0, -1, 51, True):
            with self.assertRaises(ValidationError):
                recommend_tracks.tool_call_schema.model_validate({'limit': limit})

    def test_invoice_isolation(self):
        self.assertIn('error', get_invoice_details.func(self.foreign, self.runtime))
        self.assertEqual(get_invoice_details.func(self.invoice, self.runtime)['invoice']['InvoiceId'], self.invoice)

    def test_analytics(self):
        result = query_purchases(self.context.customer_id, 'WITH totals AS (SELECT SUM(UnitPrice*Quantity) AS total FROM purchases) SELECT ROUND(total, 2) FROM totals')
        self.assertEqual(result['rows'], [(39.62,)])
        for query in (
            'SELECT * FROM Customer', 'SELECT * FROM Invoice',
            'SELECT * FROM sqlite_master', 'SELECT * FROM purchases WHERE CustomerId=17',
            'DELETE FROM purchases', 'DROP TABLE purchases',
            "ATTACH DATABASE 'Chinook.db' AS secret", 'PRAGMA database_list',
            "SELECT load_extension('anything')", 'SELECT 1; SELECT 2',
            'CREATE TABLE bad(x)', 'UPDATE purchases SET Track=1',
            "INSERT INTO purchases(Track) VALUES ('bad')",
            'WITH RECURSIVE x(a) AS (SELECT 1 UNION ALL SELECT a+1 FROM x) SELECT * FROM x',
        ):
            with self.subTest(query=query):
                self.assertIn('error', query_purchases(self.context.customer_id, query))
        self.assertEqual(query_purchases(self.context.customer_id, 'SELECT COUNT(*) FROM purchases')['rows'],
                         [(len(get_customer_purchases(1)),)])
        self.assertTrue(query_purchases(self.context.customer_id, 'SELECT a.Track FROM purchases a CROSS JOIN purchases b')['truncated'])
        self.assertIn('error', query_purchases(self.context.customer_id, 'SELECT COUNT(*) FROM purchases a CROSS JOIN purchases b CROSS JOIN purchases c CROSS JOIN purchases d'))

    def test_conversation_memory(self):
        agent = create_agent(model=TestModel(), tools=tools, system_prompt=SYSTEM_PROMPT,
                             context_schema=RuntimeContext, checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': 'test-thread'}}
        inputs = {'messages': [{'role': 'user', 'content': 'Hello'}]}
        agent.invoke(inputs, config=config, context=self.context)
        result = agent.invoke(inputs, config=config, context=self.context)
        self.assertEqual(len(result['messages']), 4)
        other = agent.invoke(inputs, config={'configurable': {'thread_id': 'new-thread'}},
                               context=self.context)
        self.assertEqual(len(other['messages']), 2)

    def test_studio_schema(self):
        from agent import RuntimeContext
        context = RuntimeContext(customer_id=1, thread_id='extra')
        self.assertEqual(context.customer_id, 1)
        self.assertNotIn('db', context.model_json_schema()['properties'])


class TopArtistTests(unittest.TestCase):
    def test_top_artists_are_customer_scoped(self):
        from agent import get_top_artists
        from collections import Counter
        for customer_id in (1, 2):
            runtime = SimpleNamespace(context=RuntimeContext(customer_id=customer_id))
            result = get_top_artists.func(runtime, limit=3)
            purchases = get_customer_purchases(customer_id)
            counts = Counter(p['Artist'] for p in purchases)
            expected = sorted(counts, key=lambda name: (-counts[name], name))[:3]
            self.assertEqual([row['Artist'] for row in result], expected)
            for row in result:
                self.assertEqual(row['PurchaseCount'], counts[row['Artist']])
                self.assertTrue(row['PurchasedGenres'])
        self.assertEqual(get_top_artists.func(
            SimpleNamespace(context=RuntimeContext(customer_id=99999))), [])

    def test_artist_tracks_exclude_owned_tracks(self):
        from agent import get_top_artists, get_unowned_artist_tracks
        for customer_id in (1, 2):
            runtime = SimpleNamespace(context=RuntimeContext(customer_id=customer_id))
            owned = {p['TrackId'] for p in get_customer_purchases(customer_id)}
            for artist in get_top_artists.func(runtime):
                tracks = get_unowned_artist_tracks.func(artist['ArtistId'], runtime)
                self.assertTrue(all(t['Artist'] == artist['Artist'] for t in tracks))
                self.assertFalse(owned.intersection(t['TrackId'] for t in tracks))
        self.assertEqual(get_unowned_artist_tracks.func(-1, runtime), [])


if __name__ == '__main__':
    unittest.main()
