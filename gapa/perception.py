"""Perception provider interfaces.

The MVP uses oracle simulation poses. VLM perception is intentionally left as a
replaceable stub.
"""

from __future__ import annotations

from typing import Any


class OraclePerception:
    def locate(self, env: Any, object_name: str) -> dict[str, Any]:
        actor = env.get_actor(object_name)
        pose = actor.get_pose()
        return {
            "object_name": object_name,
            "pose": pose.p.tolist() + pose.q.tolist(),
            "source": "oracle",
        }


class VLMPerception:
    def locate(self, env: Any, object_name: str) -> dict[str, Any]:
        return {
            "object_name": object_name,
            "pose": None,
            "source": "vlm",
            "status": "not_implemented",
        }
