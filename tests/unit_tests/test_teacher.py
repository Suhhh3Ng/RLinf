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

"""Teacher score contracts and optional real inference-engine comparison."""

import asyncio
import os
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from rlinf.algorithms.distillation.config import TeacherModelConfig
from rlinf.data.schema.distillation import TeacherScoreRequest, teacher_targets_to_batch
from rlinf.workers.teacher.backend import TeacherModelInfo, tokenizer_fingerprint
from rlinf.workers.teacher.client import TeacherClient, TeacherEndpoint
from rlinf.workers.teacher.vllm_backend import VLLMTeacherBackend


def request(k=None, selected=None):
    return TeacherScoreRequest(
        "request",
        "sample",
        "group",
        "default",
        (1, 3, 2),
        (False, True, True),
        k,
        "tokenizer",
        selected,
        "step-0" if selected is not None else None,
    )


class TestTeacher(unittest.TestCase):
    def setUp(self):
        # No inference engine is initialized; exercise the real protocol adapter.
        self.backend = VLLMTeacherBackend(TeacherModelConfig("unused"), "default")
        self.backend.info = TeacherModelInfo("model", "tokenizer", 5, 32)
        self.probs = torch.tensor(
            [[0.1, 0.2, 0.3, 0.25, 0.15], [0.3, 0.2, 0.1, 0.15, 0.25]]
        )
        self.rows = [None] + [
            {i: float(value.log()) for i, value in enumerate(row)} for row in self.probs
        ]

    def test_sampled_positions_include_last_token(self):
        target = self.backend.normalize(request(), self.rows)
        torch.testing.assert_close(
            target.logprobs[1:, 0],
            torch.stack((self.probs[0, 3].log(), self.probs[1, 2].log())),
        )
        self.assertEqual(target.valid_mask.tolist(), [False, True, True])
        self.assertEqual(target.token_ids[1:].tolist(), [[3], [2]])

    def test_teacher_topk_discards_extra_sampled_token(self):
        target = self.backend.normalize(request(2), self.rows)
        self.assertEqual(target.token_ids[1:].tolist(), [[2, 3], [0, 4]])
        self.assertLess(target.logprobs[1].exp().sum().item(), 1)

    def test_student_ids_need_not_be_teacher_topk(self):
        ids = ((0, 0), (4, 0), (2, 3))
        target = self.backend.normalize(request(2, ids), self.rows)
        self.assertEqual(target.token_ids[1:].tolist(), [[4, 0], [2, 3]])
        expected = torch.stack((self.probs[0, [4, 0]], self.probs[1, [2, 3]])).log()
        torch.testing.assert_close(target.logprobs[1:], expected)

    def test_full_targets_order_and_response_padding(self):
        target = self.backend.normalize(request(-1), self.rows)
        self.assertIsNone(target.token_ids)
        torch.testing.assert_close(target.logprobs[1:], self.probs.log())
        batch = teacher_targets_to_batch([target], [1], [2], 4, response_only=True)
        self.assertNotIn("teacher_token_ids", batch)
        self.assertEqual(
            batch["teacher_valid_mask"].tolist(), [[True, True, False, False]]
        )
        torch.testing.assert_close(batch["teacher_logprobs"][0, :2], self.probs.log())

    def test_malformed_or_mismatched_response_rejected(self):
        for bad in (
            replace(request(2), topk=0),
            replace(request(2), topk=True),
            replace(request(2), tokenizer_fingerprint="different"),
            request(2, ((0, 0), (1, 1), (2, 3))),
        ):
            with self.subTest(request=bad):
                with self.assertRaises(ValueError):
                    self.backend.normalize(bad, self.rows)
        with self.assertRaises(ValueError):
            self.backend.normalize(request(-1), self.rows[:-1])
        incomplete = [None, {0: -1.0}, self.rows[2]]
        with self.assertRaises(ValueError):
            self.backend.normalize(request(-1), incomplete)


class TestTeacherClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web

        self.delay = 0
        self.wrong_model = False
        self.active = 0
        self.peak = 0

        async def handler(http_request):
            payload = await http_request.json()
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(self.delay)
                return web.json_response(
                    {
                        "request_id": payload["request_id"],
                        "teacher_key": "default",
                        "model_fingerprint": "wrong" if self.wrong_model else "model",
                        "target_format": "sampled",
                        "student_policy_version": None,
                        "token_ids": [[0], [3], [2]],
                        "logprobs": [[0.0], [-1.0], [-2.0]],
                        "valid_mask": [False, True, True],
                    }
                )
            finally:
                self.active -= 1

        app = web.Application()
        app.router.add_post("/score", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self.endpoint = TeacherEndpoint(
            f"http://127.0.0.1:{port}",
            "default",
            TeacherModelInfo("model", "tokenizer", 5, 32),
        )

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_matching_bounded_concurrency_and_close(self):
        self.delay = 0.01
        client = TeacherClient([self.endpoint], max_concurrency=2)
        try:
            results = await asyncio.gather(
                *(
                    client.score(replace(request(), request_id=f"r{i}"))
                    for i in range(5)
                )
            )
            self.assertEqual(
                [r.request_id for r in results], [f"r{i}" for i in range(5)]
            )
            self.assertLessEqual(self.peak, 2)
        finally:
            await client.close()
            await client.close()
        with self.assertRaises(RuntimeError):
            await client.score(request())

    async def test_timeout_and_model_mismatch_fail_request(self):
        self.delay = 0.1
        client = TeacherClient([self.endpoint], timeout=0.01)
        try:
            with self.assertRaises(TimeoutError):
                await client.score(request())
        finally:
            await client.close()
        self.delay = 0
        self.wrong_model = True
        client = TeacherClient([self.endpoint])
        try:
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                await client.score(request())
        finally:
            await client.close()


@unittest.skipUnless(
    os.environ.get("RLINF_TEST_TEACHER_MODEL"),
    "Set RLINF_TEST_TEACHER_MODEL to enable a real Teacher engine test",
)
class TestTeacherEngine(unittest.TestCase):
    def test_prompt_scores_match_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = os.environ["RLINF_TEST_TEACHER_MODEL"]
        tokenizer = AutoTokenizer.from_pretrained(path)
        engine = os.environ.get("RLINF_TEST_TEACHER_BACKEND", "vllm")
        dtype = torch.float16 if engine == "sglang" else torch.float32
        device = "cuda" if engine == "sglang" else "cpu"
        tolerance = 2e-3 if engine == "sglang" else 2e-4
        model = (
            AutoModelForCausalLM.from_pretrained(path)
            .to(device=device, dtype=dtype)
            .eval()
        )
        tokens = (1, 3, 4, 5, 2)
        with torch.no_grad():
            expected = (
                model(torch.tensor([tokens], device=device))
                .logits[0, :-1]
                .float()
                .log_softmax(-1)
                .cpu()
            )
        del model

        if engine == "sglang":
            from rlinf.workers.teacher.sglang_backend import SGLangTeacherBackend

            backend_cls = SGLangTeacherBackend
            options = {
                "dtype": "float16",
                "disable_cuda_graph": True,
                "disable_piecewise_cuda_graph": True,
                "mem_fraction_static": 0.15,
                "max_running_requests": 1,
                "attention_backend": "triton",
                "sampling_backend": "pytorch",
            }
        else:
            backend_cls = VLLMTeacherBackend
            options = {
                "dtype": "float32",
                "enforce_eager": True,
                "gpu_memory_utilization": 0.15,
                "max_num_seqs": 1,
                "enable_prefix_caching": False,
            }
        backend = backend_cls(
            TeacherModelConfig(
                path,
                inference={
                    "backend": engine,
                    "max_model_len": 64,
                    "engine_kwargs": options,
                },
            ),
            "default",
        )

        async def run():
            try:
                await backend.initialize()
                for k in (None, 4, -1):
                    req = TeacherScoreRequest(
                        f"engine-{k}",
                        "sample",
                        "group",
                        "default",
                        tokens,
                        (False, True, True, True, True),
                        k,
                        tokenizer_fingerprint(tokenizer),
                    )
                    target = await backend.score(req)
                    reference = (
                        expected
                        if k == -1
                        else expected.gather(-1, target.token_ids[1:])
                    )
                    torch.testing.assert_close(
                        target.logprobs[1:], reference, atol=tolerance, rtol=2e-4
                    )
                selected = ((0, 0), (7, 8), (9, 10), (11, 12), (13, 14))
                req = TeacherScoreRequest(
                    "engine-selected",
                    "sample",
                    "group",
                    "default",
                    tokens,
                    (False, True, True, True, True),
                    2,
                    tokenizer_fingerprint(tokenizer),
                    selected,
                    "step-0",
                )
                target = await backend.score(req)
                torch.testing.assert_close(
                    target.logprobs[1:],
                    expected.gather(-1, torch.tensor(selected[1:])),
                    atol=tolerance,
                    rtol=2e-4,
                )
            finally:
                await backend.close()

        asyncio.run(run())


def create_tiny_teacher(path, architecture="gpt2"):
    """Save a seeded random GPT-2 and local tokenizer, requiring no model download."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(21)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="t0",
        bos_token="t1",
        eos_token="t2",
        pad_token="t0",
    )
    wrapped.save_pretrained(path)
    if architecture == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=64,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        ).save_pretrained(path)
    else:
        GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_positions=64,
                n_embd=32,
                n_layer=1,
                n_head=2,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        ).save_pretrained(path)


if __name__ == "__main__":
    unittest.main()
