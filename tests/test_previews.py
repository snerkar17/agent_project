import unittest
from unittest.mock import patch
import httpx
from previews_server import find_preview


class PreviewTests(unittest.IsolatedAsyncioTestCase):
    async def search(self, payload=None, status=200, fail=False):
        def respond(request):
            if fail:
                raise httpx.ConnectError('offline', request=request)
            return httpx.Response(status, json=payload, request=request)
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch('previews_server.httpx.AsyncClient', return_value=client):
            return await find_preview('One', 'Metallica')

    async def test_match(self):
        song = {'trackName': 'One', 'artistName': 'Metallica',
                'previewUrl': 'https://example.com/preview',
                'trackViewUrl': 'https://example.com/buy'}
        result = await self.search({'results': [song]})
        self.assertTrue(result['found'])
        self.assertEqual(result['preview_url'], song['previewUrl'])
        self.assertNotIn('buy_url', result)
        self.assertNotIn(song['trackViewUrl'], str(result))
        self.assertEqual(result['credit'], 'Preview provided courtesy of iTunes')

    async def test_preview_without_purchase_url(self):
        result = await self.search({'results': [{
            'trackName': 'One', 'artistName': 'Metallica',
            'previewUrl': 'https://example.com/preview',
        }]})
        self.assertTrue(result['found'])
        self.assertNotIn('buy_url', result)

    async def test_no_match_or_missing_preview(self):
        for results in ([], [{'trackName': 'One', 'artistName': 'Metallica'}],
                        [{'trackName': 'One', 'artistName': 'Cover Band',
                          'previewUrl': 'https://example.com/preview',
                          'trackViewUrl': 'https://example.com/buy'}]):
            self.assertFalse((await self.search({'results': results}))['found'])

    async def test_failures(self):
        self.assertFalse((await self.search({}, status=429))['found'])
        self.assertFalse((await self.search(fail=True))['found'])


if __name__ == '__main__':
    unittest.main()
