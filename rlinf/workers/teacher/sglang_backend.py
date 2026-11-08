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

"""SGLang native prompt scoring for a frozen Teacher."""

from rlinf.data.schema.distillation import TeacherScoreRequest, TeacherTargets

from .backend import TeacherModelInfo, TeacherScoringBackend


class SGLangTeacherBackend(TeacherScoringBackend):
    """Use input logprobs rather than generated-token logprobs."""

    async def initialize(self) -> TeacherModelInfo:
        from sglang import Engine

        info = self.load_model_info()
        options = dict(self.config.inference.get("engine_kwargs", {}))
        options.update(
            model_path=self.config.model_path,
            tp_size=self.config.inference.get("tensor_parallel_size", 1),
            pp_size=self.config.inference.get("pipeline_parallel_size", 1),
            context_length=info.max_context_length,
        )
        self.engine = Engine(**options)
        return info

    async def score(self, request: TeacherScoreRequest) -> TeacherTargets:
        self.validate_request(request)
        full = request.topk == -1 or request.selected_token_ids is not None
        count = self.info.vocab_size if full else request.topk or 0
        output = await self.engine.async_generate(
            input_ids=list(request.input_ids),
            sampling_params={"temperature": 1.0, "max_new_tokens": 1},
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=count,
        )
        meta = output["meta_info"]
        if request.topk is None:
            rows = [
                None if item[0] is None else {int(item[1]): float(item[0])}
                for item in meta["input_token_logprobs"]
            ]
        else:
            rows = [
                None if not row else {int(item[1]): float(item[0]) for item in row}
                for row in meta["input_top_logprobs"]
            ]
        return self.normalize(request, rows)

    async def close(self) -> None:
        if self.engine is not None:
            engine, self.engine = self.engine, None
            engine.shutdown()
