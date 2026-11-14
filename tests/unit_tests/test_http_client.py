# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generation requests through the shared inference HTTP client."""

import asyncio
import unittest

from aiohttp import web

from rlinf.utils.http_client import InferenceHTTPClient


class TestInferenceHTTPClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []

        async def generate(http_request):
            self.requests.append(await http_request.json())
            return web.json_response({"output_ids": [2]})

        app = web.Application()
        app.router.add_post("/generate", generate)
        runner = web.AppRunner(app)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        self.client = InferenceHTTPClient(f"http://127.0.0.1:{runner.addresses[0][1]}")
        self.addAsyncCleanup(self.client.aclose)

    async def test_sync_and_async_generation_forward_top_logprobs(self):
        for asynchronous in (False, True):
            for topk in (0, 2, [0, 2]):
                with self.subTest(asynchronous=asynchronous, topk=topk):
                    batched = isinstance(topk, list)
                    body = {
                        "input_ids": [[1], [3]] if batched else [1],
                        "sampling_params": {"max_new_tokens": 1},
                        "return_logprob": [False, True] if batched else bool(topk),
                        "top_logprobs_num": topk,
                        "image_data": ["image"],
                    }
                    if asynchronous:
                        result = await self.client.async_generate(**body)
                    else:
                        result = await asyncio.to_thread(self.client.generate, **body)
                    self.assertEqual(result, {"output_ids": [2]})
                    self.assertEqual(self.requests[-1], body)

    async def test_existing_positional_image_argument_keeps_default_topk(self):
        args = ("prompt", None, {"max_new_tokens": 1}, False, ["image"])
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                if asynchronous:
                    await self.client.async_generate(*args)
                else:
                    await asyncio.to_thread(self.client.generate, *args)
                self.assertEqual(
                    self.requests[-1],
                    {
                        "text": "prompt",
                        "sampling_params": {"max_new_tokens": 1},
                        "return_logprob": False,
                        "image_data": ["image"],
                        "top_logprobs_num": 0,
                    },
                )

    async def test_sglang_server_backend_uses_real_http_client(self):
        try:
            from rlinf.workers.rollout.sglang.backends import SGLangServerBackend
        except ImportError:
            self.skipTest("SGLang is required")

        backend = object.__new__(SGLangServerBackend)
        backend._client = self.client
        for topk in (0, 2):
            with self.subTest(topk=topk):
                options = {} if topk == 0 else {"top_logprobs_num": topk}
                result = await backend.async_generate(
                    input_ids=[1],
                    sampling_params={"max_new_tokens": 1},
                    return_logprob=bool(topk),
                    image_data=None,
                    **options,
                )
                self.assertEqual(result, {"output_ids": [2]})
                self.assertEqual(
                    self.requests[-1],
                    {
                        "input_ids": [1],
                        "sampling_params": {"max_new_tokens": 1},
                        "return_logprob": bool(topk),
                        "top_logprobs_num": topk,
                    },
                )
