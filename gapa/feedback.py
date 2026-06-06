"""Execution stage feedback and failure reporting for GAPA."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


@dataclass
class StageEvent:
    run_id: str
    attempt_id: int
    program_id: str
    stage: str
    api_call: str
    object_name: str
    target_name: Optional[str] = None
    relation: Optional[str] = None
    arm: str = "left"
    args: dict[str, Any] = field(default_factory=dict)
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SuggestedAction = Literal[
    "none", "parameter_adjust", "perception_reestimate",
    "strategy_switch", "code_regeneration",
]

FailureType = Literal[
    "object_not_found", "low_confidence_pose", "object_not_grasped",
    "wrong_object_grasped", "object_slipped", "missed_target",
    "relation_not_satisfied", "drawer_not_opened",
    "collision_or_stuck", "program_exception",
]


@dataclass
class FailureReport:
    status: str
    failed_stage: Optional[str] = None
    failure_type: Optional[FailureType] = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    suggested_action: SuggestedAction = "none"
    llm_feedback: Optional[str] = None
    perception_requests: list[dict[str, Any]] = field(default_factory=list)
    retry_policy: dict[str, Any] = field(default_factory=lambda: {"should_replan": False, "max_extra_attempts": 0})
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def ok(cls, confidence: float = 0.9) -> "FailureReport":
        return cls(status="ok", confidence=confidence)

    @classmethod
    def failed(
        cls,
        failed_stage: str,
        failure_type: FailureType,
        confidence: float,
        evidence: list[str],
        suggested_action: SuggestedAction,
        llm_feedback: str,
        retry_policy: Optional[dict[str, Any]] = None,
    ) -> "FailureReport":
        return cls(
            status="failed",
            failed_stage=failed_stage,
            failure_type=failure_type,
            confidence=confidence,
            evidence=evidence,
            suggested_action=suggested_action,
            llm_feedback=llm_feedback,
            retry_policy=retry_policy or {"should_replan": True, "max_extra_attempts": 1},
        )


class FeedbackProvider:
    def evaluate(self, event: StageEvent, context: dict[str, Any]) -> FailureReport:
        raise NotImplementedError


class FakeFeedbackProvider(FeedbackProvider):
    def evaluate(self, event: StageEvent, context: dict[str, Any]) -> FailureReport:
        if event.exception:
            return FailureReport.failed(
                failed_stage=event.stage,
                failure_type="program_exception",
                confidence=1.0,
                evidence=[f"Exception: {event.exception}"],
                suggested_action="code_regeneration",
                llm_feedback=f"Program threw exception at stage {event.stage}. Regenerate play_once.",
            )
        return FailureReport.ok(confidence=0.95)


class RuleBasedFeedbackProvider(FeedbackProvider):
    GRASP_Z_THRESHOLD = 0.02
    PLACE_XY_THRESHOLD = 0.05

    def evaluate(self, event: StageEvent, context: dict[str, Any]) -> FailureReport:
        if event.exception:
            return FailureReport.failed(
                failed_stage=event.stage,
                failure_type="program_exception",
                confidence=1.0,
                evidence=[f"Exception: {event.exception}"],
                suggested_action="code_regeneration",
                llm_feedback=f"Program exception at {event.stage}: {event.exception}. Regenerate play_once.",
            )

        if event.stage == "grasp":
            return self._check_grasp(event)
        elif event.stage == "lift":
            return self._check_lift(event)
        elif event.stage in ("place", "final_success"):
            if event.stage == "final_success" and event.exception:
                return FailureReport.failed(
                    failed_stage="final_success",
                    failure_type="relation_not_satisfied",
                    confidence=0.95,
                    evidence=[event.exception],
                    suggested_action="code_regeneration",
                    llm_feedback=f"Task failed: {event.exception}. Regenerate play_once with a different approach.",
                )
            return self._check_placement(event, context)
        elif event.stage == "open_drawer":
            return self._check_drawer(event)
        else:
            return FailureReport.ok(0.8)

    def _check_grasp(self, event: StageEvent) -> FailureReport:
        before_pose = event.before.get("object_pose")
        after_pose = event.after.get("object_pose")
        if before_pose and after_pose:
            import numpy as np
            z_diff = abs(np.array(after_pose)[2] - np.array(before_pose)[2])
            if z_diff > self.GRASP_Z_THRESHOLD:
                return FailureReport.ok(0.85)
        return FailureReport.ok(0.7)

    def _check_lift(self, event: StageEvent) -> FailureReport:
        before_pose = event.before.get("object_pose")
        after_pose = event.after.get("object_pose")
        if before_pose and after_pose:
            import numpy as np
            z_diff = abs(np.array(after_pose)[2] - np.array(before_pose)[2])
            if z_diff < self.GRASP_Z_THRESHOLD:
                return FailureReport.failed(
                    failed_stage="lift",
                    failure_type="object_not_grasped",
                    confidence=0.85,
                    evidence=[f"Object z only moved {z_diff:.3f}m"],
                    suggested_action="parameter_adjust",
                    llm_feedback="Object did not lift. Try larger pre_grasp_dis or different arm.",
                )
        return FailureReport.ok(0.8)

    def _check_placement(self, event: StageEvent, context: dict[str, Any]) -> FailureReport:
        after_pose = event.after.get("object_pose")
        target_pose = event.after.get("target_pose")
        if after_pose and target_pose:
            import numpy as np
            dist = np.linalg.norm(np.array(after_pose)[:2] - np.array(target_pose)[:2])
            if dist < self.PLACE_XY_THRESHOLD:
                return FailureReport.ok(0.9)
            return FailureReport.failed(
                failed_stage="place",
                failure_type="missed_target",
                confidence=0.8,
                evidence=[f"Object {dist:.3f}m from target"],
                suggested_action="parameter_adjust",
                llm_feedback=f"Object is {dist:.3f}m from target. Adjust place position.",
            )
        return FailureReport.ok(0.7)

    def _check_drawer(self, event: StageEvent) -> FailureReport:
        return FailureReport.ok(0.75)
