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

"""Driver ownership of Teacher worker groups."""

from dataclasses import dataclass, field

from rlinf.algorithms.distillation.config import (
    DistillationLossConfig,
    TeacherModelConfig,
)
from rlinf.utils.logging import get_logger

from .client import TeacherEndpoint
from .teacher_worker import TeacherWorker


@dataclass
class TeacherRuntime:
    """Driver-only handles; remote workers receive only endpoints."""

    worker_groups: list = field(default_factory=list)
    endpoints: list[TeacherEndpoint] = field(default_factory=list)

    def close(self) -> None:
        """Attempt all cleanup operations without masking a training exception."""
        groups, self.worker_groups = self.worker_groups, []
        self.endpoints = []
        for group in reversed(groups):
            try:
                group.stop().wait()
            except Exception:
                get_logger().exception("Failed to stop a Teacher worker group")


def launch_teacher_workers(cfg, cluster, placement) -> TeacherRuntime:
    """Launch frozen replicas, rolling back partially initialized groups."""
    runtime = TeacherRuntime()
    loss = DistillationLossConfig(**cfg.distillation.get("distillation_loss", {}))
    try:
        for key, values in cfg.distillation.teacher_models.items():
            model = TeacherModelConfig(**values)
            for replica_id in range(model.num_replicas):
                group = TeacherWorker.create_group(key, model, loss).launch(
                    cluster,
                    name=f"teacher_{key}_{replica_id}",
                    placement_strategy=placement.get_strategy(key, replica_id),
                )
                runtime.worker_groups.append(group)
                runtime.endpoints.extend(group.init_worker().wait())
        return runtime
    except BaseException:
        runtime.close()
        raise
