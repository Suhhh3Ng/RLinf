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

"""Bounded HTTP connections to immutable Teacher replicas."""

import asyncio
from dataclasses import asdict, dataclass

import torch

from rlinf.data.schema.distillation import TeacherScoreRequest, TeacherTargets

from .backend import TeacherModelInfo


@dataclass(frozen=True)
class TeacherEndpoint:
    """Serializable service address and model identity."""

    url: str
    teacher_key: str
    model_info: TeacherModelInfo


class TeacherClient:
    """Limit in-flight scoring and match every response to its request."""

    def __init__(
        self,
        endpoints: list[TeacherEndpoint],
        *,
        max_concurrency: int = 8,
        timeout: float = 300,
        max_payload_bytes: int = 256 * 1024 * 1024,
    ):
        if (
            not endpoints
            or max_concurrency < 1
            or timeout <= 0
            or max_payload_bytes < 1
        ):
            raise ValueError(
                "TeacherClient requires endpoints and positive resource limits"
            )
        if len({(e.teacher_key, e.model_info) for e in endpoints}) != 1:
            raise ValueError(
                "Stage-one replicas must have the same immutable model identity"
            )
        self.endpoints = tuple(endpoints)
        self._limit = asyncio.Semaphore(max_concurrency)
        self._timeout = timeout
        self._max_payload_bytes = max_payload_bytes
        self._session = None
        self._next = 0
        self._closed = False

    async def score(self, request: TeacherScoreRequest) -> TeacherTargets:
        """Score one request; errors fail the caller instead of dropping samples."""
        import aiohttp

        if self._closed:
            raise RuntimeError("TeacherClient is closed")
        async with self._limit:
            endpoint = self.endpoints[self._next % len(self.endpoints)]
            self._next += 1
            request.validate(endpoint.model_info.vocab_size)
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                )
            async with self._session.post(
                endpoint.url + "/score", json=asdict(request)
            ) as response:
                response.raise_for_status()
                chunks, total = [], 0
                async for chunk in response.content.iter_chunked(64 * 1024):
                    total += len(chunk)
                    if total > self._max_payload_bytes:
                        raise ValueError(
                            f"Teacher request {request.request_id}: response exceeds byte budget"
                        )
                    chunks.append(chunk)
            import json

            data = json.loads(b"".join(chunks))
            result = TeacherTargets(
                request_id=data["request_id"],
                teacher_key=data["teacher_key"],
                model_fingerprint=data["model_fingerprint"],
                target_format=data["target_format"],
                student_policy_version=data["student_policy_version"],
                token_ids=None
                if data["token_ids"] is None
                else torch.tensor(data["token_ids"], dtype=torch.long),
                logprobs=torch.tensor(data["logprobs"], dtype=torch.float32),
                valid_mask=torch.tensor(data["valid_mask"], dtype=torch.bool),
            )
            if result.model_fingerprint != endpoint.model_info.model_fingerprint:
                raise ValueError(
                    f"Teacher request {request.request_id}: model fingerprint changed"
                )
            result.validate(request, endpoint.model_info.vocab_size)
            return result

    async def close(self) -> None:
        """Close local connections, leaving the remote model owned by its runtime."""
        self._closed = True
        if self._session is not None:
            session, self._session = self._session, None
            await session.close()
