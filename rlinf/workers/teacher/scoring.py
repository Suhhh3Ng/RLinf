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

"""Attach Teacher targets to original reasoning and Agent trajectories."""

import asyncio
from uuid import uuid4

from rlinf.algorithms.distillation.config import build_distillation_plan
from rlinf.data.schema.distillation import TeacherScoreRequest

from .backend import tokenizer_fingerprint
from .client import TeacherClient


class RolloutTeacherScorer:
    """One connection owner per rollout or Agent worker."""

    def __init__(self, cfg, endpoints, tokenizer):
        self.plan = build_distillation_plan(cfg)
        self.client = TeacherClient(endpoints)
        self.fingerprint = tokenizer_fingerprint(tokenizer)
        self.teacher_key = endpoints[0].teacher_key
        if self.fingerprint != endpoints[0].model_info.tokenizer_fingerprint:
            raise ValueError("Student and Teacher tokenizer mappings must be identical")
        self.student_policy_version = "0"

    def request(
        self, input_ids, response_mask, *, selected_token_ids=None, sample_id=None
    ):
        """Build a position-aligned request without decoding or re-tokenizing."""
        return TeacherScoreRequest(
            request_id=uuid4().hex,
            sample_id=sample_id or uuid4().hex,
            group_id=sample_id or "",
            teacher_key=self.teacher_key,
            input_ids=tuple(input_ids),
            response_mask=tuple(bool(v) for v in response_mask),
            topk=self.plan.topk,
            tokenizer_fingerprint=self.fingerprint,
            selected_token_ids=selected_token_ids,
            student_policy_version=self.student_policy_version
            if self.plan.need_student_topk_selection
            else None,
        )

    async def score_agent_output(self, output):
        """Score trainable turns independently using their actual private context."""
        turns = getattr(output, "single_turn_outputs", [output])

        async def score_turn(turn):
            if turn.extra_fields.get("not_training", False):
                return
            mask = [False] * len(turn.prompt_ids) + list(
                turn.response_mask
                if turn.response_mask is not None
                else [True] * len(turn.response_ids)
            )
            request = self.request(turn.prompt_ids + turn.response_ids, mask)
            if self.plan.need_student_topk_selection:
                turn.teacher_request = request
                turn.student_policy_version = self.student_policy_version
            else:
                turn.teacher_targets = await self.client.score(request)

        await asyncio.gather(*(score_turn(turn) for turn in turns))
        return output

    async def score_sequence(self, prompt_ids, response_ids, sample_id):
        """Score a completed reasoning sequence."""
        request = self.request(
            prompt_ids + response_ids,
            [False] * len(prompt_ids) + [True] * len(response_ids),
            sample_id=sample_id,
        )
        if self.plan.need_student_topk_selection:
            return request
        return await self.client.score(request)

    async def close(self):
        """Release local client connections."""
        await self.client.close()
