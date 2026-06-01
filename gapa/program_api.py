"""Safe runtime API exposed to generated GAPA play_once programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from envs.utils import ArmTag
except Exception:  # pragma: no cover - used when simulator deps are unavailable in unit tests.
    class ArmTag:
        def __init__(self, value):
            if isinstance(value, ArmTag):
                value = value.arm
            if value not in ("left", "right"):
                raise ValueError(f"Invalid arm tag: {value}")
            self.arm = value

        @property
        def opposite(self):
            return ArmTag("right" if self.arm == "left" else "left")

        def __eq__(self, other):
            return self.arm == (other.arm if isinstance(other, ArmTag) else other)

        def __hash__(self):
            return hash(self.arm)

        def __str__(self):
            return self.arm

from .program_safety import validate_program_source
from .task_dsl import FailureReport, TaskDSL


class ProgramExecutionError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


@dataclass
class ProgramCandidate:
    program_id: str
    source: str
    description: str = ""
    metadata: dict[str, Any] | None = None
    safety: dict[str, Any] | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "description": self.description,
            "source": self.source,
            "metadata": self.metadata or {},
            "safety": self.safety or {},
            "path": self.path,
        }


def _choose_arm_for_actor(actor: Any) -> ArmTag:
    return ArmTag("left" if actor.get_pose().p[0] < 0 else "right")


def _default_contact_point(env: Any, object_name: str, arm_tag: ArmTag, requested: Any = None) -> Any:
    if requested is not None:
        return requested
    spec = getattr(env, "gapa_specs", {}).get(object_name)
    if spec is not None and spec.modelname in {"002_bowl", "021_cup"}:
        return [0, 2][int(arm_tag == "left")]
    return None


class SafeSkillAPI:
    def __init__(self, env: Any, run_dir: str | None = None, generate_id: str = "current", attempt_id: int = 1):
        self.env = env
        self.run_dir = run_dir
        self.generate_id = generate_id
        self.attempt_id = attempt_id
        self.held: dict[str, ArmTag] = {}
        self.last_gripper: ArmTag | None = None
        self.step_index = 0

    def pose(self, name: str) -> list[float]:
        actor = self.env.get_actor(name)
        pose = actor.get_pose()
        return pose.p.tolist() + pose.q.tolist()

    def choose_arm(self, name: str) -> str:
        return str(_choose_arm_for_actor(self.env.get_actor(name)))

    def grasp(
        self,
        name: str,
        arm: str | None = None,
        pre_grasp_dis: float = 0.09,
        grasp_dis: float = 0.0,
        gripper_pos: float = 0.0,
        contact_point_id: int | list[int] | None = None,
    ) -> None:
        actor = self.env.get_actor(name)
        arm_tag = ArmTag(arm) if arm else _choose_arm_for_actor(actor)
        grasp_action = self.env.grasp_actor(
            actor,
            arm_tag=arm_tag,
            pre_grasp_dis=float(pre_grasp_dis),
            grasp_dis=float(grasp_dis),
            gripper_pos=float(gripper_pos),
            contact_point_id=_default_contact_point(self.env, name, arm_tag, contact_point_id),
        )
        if self.last_gripper is not None and self.last_gripper != arm_tag:
            moved = self.env.move(grasp_action, self.env.back_to_origin(arm_tag=arm_tag.opposite))
        else:
            moved = self.env.move(grasp_action)
        self._require_moved(moved, "grasp", f"grasp({name}) motion failed.")
        self.held[name] = arm_tag
        self.last_gripper = arm_tag
        self._snapshot(f"grasp_{name}")

    def move_up(self, arm: str, z: float = 0.08, move_axis: str = "world") -> None:
        arm_tag = ArmTag(arm)
        moved = self.env.move(self.env.move_by_displacement(arm_tag=arm_tag, z=float(z), move_axis=move_axis))
        self._require_moved(moved, "move_up", f"move_up({arm}) failed.")
        self._snapshot(f"move_up_{arm}")

    def place_on(
        self,
        name: str,
        target: str,
        arm: str | None = None,
        functional_point_id: int | None = 0,
        pre_dis: float = 0.08,
        dis: float = 0.02,
        constrain: str = "auto",
        pre_dis_axis: str = "grasp",
    ) -> None:
        self._place(name, target, "on", arm, functional_point_id, pre_dis, dis, constrain, pre_dis_axis)

    def place_in(
        self,
        name: str,
        target: str,
        arm: str | None = None,
        functional_point_id: int | None = 0,
        pre_dis: float = 0.08,
        dis: float = 0.02,
        constrain: str = "auto",
        pre_dis_axis: str = "grasp",
    ) -> None:
        self._place(name, target, "in", arm, functional_point_id, pre_dis, dis, constrain, pre_dis_axis)

    def back_to_origin(self, arm: str) -> None:
        arm_tag = ArmTag(arm)
        moved = self.env.move(self.env.back_to_origin(arm_tag=arm_tag))
        self._require_moved(moved, "back_to_origin", f"back_to_origin({arm}) failed.")
        self._snapshot(f"back_to_origin_{arm}")

    def _place(
        self,
        name: str,
        target: str,
        relation: str,
        arm: str | None,
        functional_point_id: int | None,
        pre_dis: float,
        dis: float,
        constrain: str,
        pre_dis_axis: str,
    ) -> None:
        actor = self.env.get_actor(name)
        arm_tag = ArmTag(arm) if arm else self.held.get(name) or _choose_arm_for_actor(actor)
        target_pose = self.env.get_target_pose(target, relation=relation)
        moved = self.env.move(
            self.env.place_actor(
                actor,
                arm_tag=arm_tag,
                target_pose=target_pose,
                functional_point_id=functional_point_id,
                pre_dis=float(pre_dis),
                dis=float(dis),
                is_open=True,
                constrain=constrain,
                pre_dis_axis=pre_dis_axis,
            )
        )
        self._require_moved(moved, f"place_{relation}", f"place_{relation}({name}, {target}) failed.")
        self.held.pop(name, None)
        self.last_gripper = arm_tag
        self._snapshot(f"place_{relation}_{name}_{target}")

    def _require_moved(self, moved: Any, stage: str, message: str) -> None:
        if not moved or not self.env.plan_success:
            self.env.plan_success = True
            raise ProgramExecutionError(stage, message)

    def _snapshot(self, label: str) -> None:
        self.step_index += 1
        self._record_video_frames(1)
        if not self.run_dir:
            return
        self.env.save_camera_images(
            task_name="gapa",
            step_name=f"attempt{self.attempt_id}_step{self.step_index}_{label}",
            generate_num_id=self.generate_id,
            save_dir=self.run_dir,
        )

    def _record_video_frames(self, frame_count: int) -> None:
        if not getattr(self.env, "save_data", False) or not hasattr(self.env, "_take_picture"):
            return
        for _ in range(frame_count):
            self.env._take_picture()
            self.env.scene.step()
            self.env._update_render()


def execute_program_candidate(
    candidate: ProgramCandidate,
    env: Any,
    task: TaskDSL,
    run_dir: str | None = None,
    attempt_id: int = 1,
    generate_id: str = "current",
) -> FailureReport | None:
    env.active_task = task
    env.active_plan = None
    env.plan_success = True
    api = SafeSkillAPI(env, run_dir=run_dir, generate_id=generate_id, attempt_id=attempt_id)
    try:
        validate_program_source(candidate.source)
        namespace: dict[str, Any] = {}
        exec(compile(candidate.source, f"<{candidate.program_id}>", "exec"), {"__builtins__": {}}, namespace)
        play_once = namespace.get("play_once")
        if not callable(play_once):
            raise ProgramExecutionError("program", "Generated program did not define play_once(api).")
        api._snapshot("initial")
        play_once(api)
    except ProgramExecutionError as exc:
        return FailureReport(
            attempt_id=attempt_id,
            stage=exc.stage,
            message=exc.message,
            action="none",
            details={"program_id": candidate.program_id},
        )
    except Exception as exc:
        return FailureReport(
            attempt_id=attempt_id,
            stage="program_exception",
            message=str(exc),
            action="none",
            details={"program_id": candidate.program_id},
        )

    if not env.check_success():
        return FailureReport(
            attempt_id=attempt_id,
            stage="success_check",
            message="Program executed but task success condition failed.",
            action="none",
            details={"program_id": candidate.program_id},
        )
    return None
