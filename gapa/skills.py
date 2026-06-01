"""SkillLibrary interpreter for GAPA SkillPlans."""

from __future__ import annotations

from typing import Any

from envs.utils import ArmTag

from .task_dsl import FailureReport, SkillPlan, SkillStep


class SkillExecutionError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


def _choose_arm(actor: Any) -> ArmTag:
    return ArmTag("left" if actor.get_pose().p[0] < 0 else "right")


def _spec_for(env: Any, object_name: str) -> Any | None:
    return getattr(env, "gapa_specs", {}).get(object_name)


def _default_contact_point(env: Any, object_name: str, arm_tag: ArmTag, requested: Any = None) -> Any:
    if requested is not None:
        return requested
    spec = _spec_for(env, object_name)
    if spec is not None and spec.modelname in {"002_bowl", "021_cup"}:
        return [0, 2][int(arm_tag == "left")]
    return None


def _default_place_params(env: Any, object_name: str, params: dict[str, Any]) -> dict[str, Any]:
    spec = _spec_for(env, object_name)
    is_box = spec is not None and spec.kind == "box"
    defaults = {
        "functional_point_id": 0,
        "pre_dis": 0.05 if is_box else 0.12,
        "dis": 0.0 if is_box else 0.03,
        "constrain": "auto" if is_box else "free",
        "pre_dis_axis": "fp" if is_box else "grasp",
        "retreat_z": 0.07 if is_box else 0.08,
        "retreat_axis": "arm",
    }
    return {**defaults, **params}


class SkillLibrary:
    def __init__(self, env: Any, run_dir: str | None = None, generate_id: str = "current"):
        self.env = env
        self.run_dir = run_dir
        self.generate_id = generate_id
        self.held: dict[str, ArmTag] = {}
        self.last_gripper: ArmTag | None = None

    def execute_plan(self, plan: SkillPlan, attempt_id: int = 1) -> FailureReport | None:
        self.env.active_task = plan.task
        self.env.active_plan = plan
        self.env.plan_success = True
        self._snapshot(f"attempt{attempt_id}_step0_initial")

        for index, step in enumerate(plan.steps, start=1):
            try:
                self.execute_step(step)
                self._snapshot(f"attempt{attempt_id}_step{index}_{step.skill}")
            except SkillExecutionError as exc:
                return FailureReport(
                    attempt_id=attempt_id,
                    stage=exc.stage,
                    message=exc.message,
                    action="adjust_parameters",
                    details={"step": step.to_dict(), "plan_id": plan.plan_id},
                )

        if not self.env.check_success():
            return FailureReport(
                attempt_id=attempt_id,
                stage="success_check",
                message="Plan executed but task success condition failed.",
                action="adjust_parameters",
                details={"plan_id": plan.plan_id},
            )
        return None

    def execute_step(self, step: SkillStep) -> None:
        if step.skill == "grasp_object":
            self._grasp(step.object_name, step.params)
        elif step.skill in ("place_in", "place_on"):
            if not step.target_name:
                raise SkillExecutionError(step.skill, "Missing target_name for placement.")
            self._place(step.object_name, step.target_name, step.skill, step.params)
        else:
            raise SkillExecutionError(step.skill, f"Unsupported skill: {step.skill}")

    def _grasp(self, object_name: str, params: dict[str, Any]) -> None:
        actor = self.env.get_actor(object_name)
        arm_tag = _choose_arm(actor)
        grasp_action = self.env.grasp_actor(
            actor,
            arm_tag=arm_tag,
            pre_grasp_dis=params.get("pre_grasp_dis", 0.09),
            grasp_dis=params.get("grasp_dis", 0.0),
            gripper_pos=params.get("gripper_pos", 0.0),
            contact_point_id=_default_contact_point(self.env, object_name, arm_tag, params.get("contact_point_id")),
        )
        if self.last_gripper is not None and self.last_gripper != arm_tag:
            moved = self.env.move(grasp_action, self.env.back_to_origin(arm_tag=arm_tag.opposite))
        else:
            moved = self.env.move(grasp_action)
        if not moved or not self.env.plan_success:
            self.env.plan_success = True
            raise SkillExecutionError("grasp_object", "grasp_object motion failed.")
        self.held[object_name] = arm_tag
        lift_z = params.get("lift_z", 0.08)
        if lift_z:
            moved = self.env.move(self.env.move_by_displacement(arm_tag=arm_tag, z=lift_z))
            if not moved or not self.env.plan_success:
                self.env.plan_success = True
                raise SkillExecutionError("grasp_object", "grasp_object lift failed.")
        self.last_gripper = arm_tag

    def _place(self, object_name: str, target_name: str, skill: str, params: dict[str, Any]) -> None:
        params = _default_place_params(self.env, object_name, params)
        actor = self.env.get_actor(object_name)
        arm_tag = self.held.get(object_name) or _choose_arm(actor)
        target_pose = self.env.get_target_pose(target_name, relation="in" if skill == "place_in" else "on")
        moved = self.env.move(
            self.env.place_actor(
                actor,
                arm_tag=arm_tag,
                target_pose=target_pose,
                functional_point_id=params.get("functional_point_id"),
                pre_dis=params.get("pre_dis", 0.08),
                dis=params.get("dis", 0.02),
                is_open=params.get("is_open", True),
                constrain=params.get("constrain", "free"),
                pre_dis_axis=params.get("pre_dis_axis", "grasp"),
            )
        )
        if not moved or not self.env.plan_success:
            self.env.plan_success = True
            raise SkillExecutionError(skill, f"{skill} motion failed.")
        if params.get("retreat_z", 0.08):
            moved = self.env.move(
                self.env.move_by_displacement(
                    arm_tag=arm_tag,
                    z=params.get("retreat_z", 0.08),
                    move_axis=params.get("retreat_axis", "arm"),
                )
            )
            if not moved or not self.env.plan_success:
                self.env.plan_success = True
                raise SkillExecutionError(skill, f"{skill} retreat failed.")
        self.held.pop(object_name, None)
        self.last_gripper = arm_tag

    def _snapshot(self, step_name: str) -> None:
        self._record_video_frames(1)
        if not self.run_dir:
            return
        self.env.save_camera_images(
            task_name="gapa",
            step_name=step_name,
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
