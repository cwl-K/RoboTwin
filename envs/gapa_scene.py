from __future__ import annotations

from typing import Any, Literal

import numpy as np

from gapa.object_registry import GapaObjectSpec, OBJECT_SPECS, validate_object_names

_GAPA_RUNTIME_IMPORT_ERROR: Exception | None = None

try:
    import sapien.core as sapien

    from ._base_task import Base_Task
    from .utils import create_actor, create_box, rand_create_sapien_urdf_obj, rand_pose
except Exception as exc:  # pragma: no cover - exercised only when simulator deps are unavailable.
    sapien = None
    Base_Task = object
    create_actor = None
    create_box = None
    rand_create_sapien_urdf_obj = None
    rand_pose = None
    _GAPA_RUNTIME_IMPORT_ERROR = exc


NON_OVERLAP_MARGIN = 0.02
PLACEMENT_ATTEMPTS = 50
SLOT_JITTER = 0.015
SOURCE_X_RANGE = (-0.28, 0.28)
SOURCE_Y_RANGE = (-0.10, 0.05)
OFFICIAL_CABINET_SOURCE_X_RANGE = (-0.32, 0.32)
OFFICIAL_CABINET_SOURCE_Y_RANGE = (-0.20, -0.10)
TARGET_X_RANGE = (-0.08, 0.08)
TARGET_Y_RANGE = (-0.15, -0.10)
CABINET_X_RANGE = (-0.05, 0.05)
CABINET_Y_RANGE = (0.155, 0.155)
SOURCE_CENTER_X_EXCLUSION = 0.05
OFFICIAL_CABINET_SOURCE_CENTER_X_EXCLUSION = 0.20
SOURCE_LARGE_SAFE_SLOTS = (
    (-0.25, -0.095),
    (0.25, -0.095),
    (-0.25, 0.04),
    (0.25, 0.04),
)
SOURCE_SMALL_SAFE_SLOTS = (
    (-0.19, -0.095),
    (0.19, -0.095),
    (-0.09, 0.05),
    (0.09, 0.05),
    (-0.25, 0.04),
    (0.25, 0.04),
)
OFFICIAL_CABINET_SOURCE_SAFE_SLOTS = (
    (-0.25, -0.15),
    (0.25, -0.15),
    (-0.305, -0.185),
    (0.305, -0.185),
    (-0.215, -0.115),
    (0.215, -0.115),
)
TARGET_SAFE_SLOTS = ((0.0, -0.13),)
CABINET_SAFE_SLOTS = ((0.0, 0.155),)
TABLE_SAFE_SLOTS = SOURCE_SMALL_SAFE_SLOTS + TARGET_SAFE_SLOTS
PlacementRecord = tuple[str, float, float, float]
PlacementZone = Literal["source", "target", "cabinet", "cabinet_source"]


def _select_scene_specs(object_names: list[str] | tuple[str, ...]) -> list[tuple[str, GapaObjectSpec]]:
    selected = validate_object_names(object_names)
    return [(name, OBJECT_SPECS[name]) for name in selected]


def _is_non_overlapping(
    x: float,
    y: float,
    radius: float,
    accepted: list[PlacementRecord],
    margin: float = NON_OVERLAP_MARGIN,
) -> bool:
    for _, other_x, other_y, other_radius in accepted:
        distance = float(np.hypot(x - other_x, y - other_y))
        if distance <= radius + other_radius + margin:
            return False
    return True


def _placement_zone(spec: GapaObjectSpec) -> PlacementZone:
    if spec.kind == "urdf":
        return "cabinet"
    if spec.can_target and not spec.can_grasp:
        return "target"
    return "source"


def _source_slots_for_spec(spec: GapaObjectSpec, cabinet_mode: bool = False) -> tuple[tuple[float, float], ...]:
    if cabinet_mode:
        return OFFICIAL_CABINET_SOURCE_SAFE_SLOTS
    if spec.footprint_radius >= 0.06:
        return SOURCE_LARGE_SAFE_SLOTS
    return SOURCE_SMALL_SAFE_SLOTS


def _slots_for_spec(spec: GapaObjectSpec, cabinet_mode: bool = False) -> tuple[tuple[float, float], ...]:
    zone = _placement_zone(spec)
    if zone == "cabinet":
        return CABINET_SAFE_SLOTS
    if zone == "target":
        return TARGET_SAFE_SLOTS
    return _source_slots_for_spec(spec, cabinet_mode=cabinet_mode)


def _sampling_zone(spec: GapaObjectSpec, cabinet_mode: bool = False) -> PlacementZone:
    zone = _placement_zone(spec)
    if cabinet_mode and zone == "source":
        return "cabinet_source"
    return zone


def _is_in_spawn_zone(x: float, y: float, zone: PlacementZone) -> bool:
    if zone == "cabinet":
        return CABINET_X_RANGE[0] <= x <= CABINET_X_RANGE[1] and CABINET_Y_RANGE[0] <= y <= CABINET_Y_RANGE[1]
    if zone == "target":
        return TARGET_X_RANGE[0] <= x <= TARGET_X_RANGE[1] and TARGET_Y_RANGE[0] <= y <= TARGET_Y_RANGE[1]
    if zone == "cabinet_source":
        return (
            OFFICIAL_CABINET_SOURCE_X_RANGE[0] <= x <= OFFICIAL_CABINET_SOURCE_X_RANGE[1]
            and OFFICIAL_CABINET_SOURCE_Y_RANGE[0] <= y <= OFFICIAL_CABINET_SOURCE_Y_RANGE[1]
            and abs(x) >= OFFICIAL_CABINET_SOURCE_CENTER_X_EXCLUSION
        )
    return (
        SOURCE_X_RANGE[0] <= x <= SOURCE_X_RANGE[1]
        and SOURCE_Y_RANGE[0] <= y <= SOURCE_Y_RANGE[1]
        and abs(x) >= SOURCE_CENTER_X_EXCLUSION
    )


def _sample_non_overlapping_pose(
    slots: tuple[tuple[float, float], ...],
    spec: GapaObjectSpec,
    accepted: list[PlacementRecord],
    zone: PlacementZone | None = None,
    attempts: int = PLACEMENT_ATTEMPTS,
    jitter: float = SLOT_JITTER,
) -> tuple[float, float]:
    zone = zone or _placement_zone(spec)
    if zone == "cabinet_source":
        for _ in range(attempts * max(1, len(slots))):
            x = float(np.random.uniform(*OFFICIAL_CABINET_SOURCE_X_RANGE))
            y = float(np.random.uniform(*OFFICIAL_CABINET_SOURCE_Y_RANGE))
            if _is_in_spawn_zone(x, y, zone) and _is_non_overlapping(x, y, spec.footprint_radius, accepted):
                return x, y

    for slot_index in np.random.permutation(len(slots)):
        slot = slots[int(slot_index)]
        for _ in range(attempts):
            x = float(slot[0] + np.random.uniform(-jitter, jitter))
            y = float(slot[1] + np.random.uniform(-jitter, jitter))
            if _is_in_spawn_zone(x, y, zone) and _is_non_overlapping(x, y, spec.footprint_radius, accepted):
                return x, y

        x, y = float(slot[0]), float(slot[1])
        if _is_in_spawn_zone(x, y, zone) and _is_non_overlapping(x, y, spec.footprint_radius, accepted):
            return x, y

    raise RuntimeError(f"Could not place {spec.alias} without overlap in the {zone} spawn zone.")


def _sample_scene_layout(selected_specs: list[tuple[str, GapaObjectSpec]]) -> dict[str, tuple[float, float]]:
    source_specs = [(alias, spec) for alias, spec in selected_specs if _placement_zone(spec) == "source"]
    target_specs = [(alias, spec) for alias, spec in selected_specs if _placement_zone(spec) == "target"]
    cabinet_specs = [(alias, spec) for alias, spec in selected_specs if _placement_zone(spec) == "cabinet"]
    cabinet_mode = bool(cabinet_specs)
    source_slot_limit = len(OFFICIAL_CABINET_SOURCE_SAFE_SLOTS) if cabinet_mode else len(SOURCE_SMALL_SAFE_SLOTS)
    if len(source_specs) > source_slot_limit:
        raise ValueError(f"Select at most {source_slot_limit} graspable GAPA objects.")
    if len(target_specs) > len(TARGET_SAFE_SLOTS):
        raise ValueError(f"Select at most {len(TARGET_SAFE_SLOTS)} target-only GAPA object.")
    if len(cabinet_specs) > len(CABINET_SAFE_SLOTS):
        raise ValueError(f"Select at most {len(CABINET_SAFE_SLOTS)} cabinet GAPA object.")

    accepted: list[PlacementRecord] = []
    placements = {}
    placement_order = sorted(
        cabinet_specs + target_specs + source_specs,
        key=lambda item: (
            0 if _placement_zone(item[1]) == "cabinet" else 1 if _placement_zone(item[1]) == "target" else 2,
            -item[1].footprint_radius,
        ),
    )
    for alias, spec in placement_order:
        zone = _sampling_zone(spec, cabinet_mode=cabinet_mode)
        x, y = _sample_non_overlapping_pose(_slots_for_spec(spec, cabinet_mode=cabinet_mode), spec, accepted, zone=zone)
        accepted.append((alias, x, y, spec.footprint_radius))
        placements[alias] = (x, y)
    return placements


class GapaScene(Base_Task):
    """Generic fixed-pool scene for the GAPA MVP."""

    def __init__(self):
        if _GAPA_RUNTIME_IMPORT_ERROR is not None:
            raise RuntimeError("GapaScene runtime dependencies are unavailable.") from _GAPA_RUNTIME_IMPORT_ERROR
        super().__init__()
        self.gapa_objects: dict[str, Any] = {}
        self.gapa_specs: dict[str, GapaObjectSpec] = {}
        self.gapa_object_names: list[str] = []
        self.gapa_task_origin_z: float | None = None
        self.gapa_task_arm_tag: str | None = None
        self.active_task = None
        self.active_plan = None

    def setup_demo(self, is_test: bool = False, **kwags):
        self.gapa_object_names = validate_object_names(kwags.get("gapa_object_names"))
        super()._init_task_env_(**kwags)

    def check_stable(self):
        # The fixed GAPA pool intentionally includes lightweight graspable
        # objects; small pose drift should be handled by the oracle pose provider
        # instead of rejecting the whole random scene at initialization.
        return True, []

    def load_actors(self):
        self.gapa_objects = {}
        self.gapa_specs = {}
        self.gapa_task_origin_z = None
        self.gapa_task_arm_tag = None
        selected_specs = _select_scene_specs(self.gapa_object_names)
        placements = _sample_scene_layout(selected_specs)

        for alias, spec in selected_specs:
            x, y = placements[alias]
            pose = rand_pose(
                xlim=[x, x],
                ylim=[y, y],
                zlim=[spec.z],
                qpos=spec.qpos,
                rotate_rand=spec.rotate_rand,
                rotate_lim=list(spec.rotate_lim),
            )
            if spec.kind == "box":
                actor = create_box(
                    scene=self,
                    pose=pose,
                    half_size=spec.half_size,
                    color=spec.color,
                    name=alias,
                    is_static=spec.is_static,
                )
            elif spec.kind == "urdf":
                actor = rand_create_sapien_urdf_obj(
                    scene=self,
                    modelname=spec.modelname,
                    modelid=spec.model_id,
                    xlim=[x, x],
                    ylim=[y, y],
                    zlim=[spec.z],
                    rotate_rand=False,
                    qpos=spec.qpos,
                    fix_root_link=True,
                )
            else:
                actor = create_actor(
                    scene=self,
                    pose=pose,
                    modelname=spec.modelname,
                    convex=spec.convex,
                    is_static=spec.is_static,
                    model_id=spec.model_id,
                )
            if actor is None:
                raise RuntimeError(f"Failed to create GAPA actor: {alias} ({spec.modelname})")
            if hasattr(actor, "set_mass") and spec.mass:
                actor.set_mass(spec.mass)
            self.gapa_objects[alias] = actor
            self.gapa_specs[alias] = spec
            setattr(self, alias, actor)
            self.add_prohibit_area(actor, padding=max(0.04, spec.footprint_radius * 0.5))

    def play_once(self):
        if self.active_plan is None:
            return self.info
        from gapa.skills import SkillLibrary

        failure = SkillLibrary(self).execute_plan(self.active_plan)
        self.info["info"] = {
            "plan_id": self.active_plan.plan_id,
            "failure": None if failure is None else failure.to_dict(),
        }
        return self.info

    def get_actor(self, object_name: str):
        try:
            return self.gapa_objects[object_name]
        except KeyError as exc:
            raise KeyError(f"Unknown GAPA object: {object_name}") from exc

    def get_scene_description(self) -> dict[str, dict[str, Any]]:
        description = {}
        for alias, actor in self.gapa_objects.items():
            pose = actor.get_pose()
            spec = self.gapa_specs[alias]
            description[alias] = {
                "name": alias,
                "label": spec.label,
                "modelname": spec.modelname,
                "model_id": spec.model_id,
                "roles": list(spec.roles),
                "target_relations": list(spec.target_relations),
                "pose": pose.p.tolist() + pose.q.tolist(),
            }
        return description

    def get_target_pose(self, target_name: str, relation: str = "on"):
        target = self.get_actor(target_name)
        spec = self.gapa_specs[target_name]
        if target_name == "cabinet" and relation == "in":
            return target.get_functional_point(0)
        if spec.kind == "box":
            return target.get_functional_point(1, "pose")
        if target_name == "plate":
            return target.get_functional_point(0, "pose")
        if target_name in ("bowl", "cup"):
            pose = target.get_pose()
            return sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + spec.target_z_offset], pose.q)
        return target.get_pose()

    def check_success(self):
        if self.active_task is None:
            return False
        obj = self.get_actor(self.active_task.object_name)
        target = self.get_actor(self.active_task.target_name)
        obj_p = np.array(obj.get_pose().p)
        if self.active_task.target_name == "cabinet" and self.active_task.relation == "in":
            target_pose = self.get_target_pose("cabinet", relation="in")
            target_p = np.array(target_pose.p if hasattr(target_pose, "p") else target_pose[:3])
            origin_z = self.gapa_task_origin_z
            if origin_z is None:
                origin_z = obj_p[2]
            arm_tag = self.gapa_task_arm_tag
            if arm_tag == "left":
                gripper_open = self.robot.is_left_gripper_open()
            elif arm_tag == "right":
                gripper_open = self.robot.is_right_gripper_open()
            else:
                gripper_open = False
            tag = np.all(abs(obj_p[:2] - target_p[:2]) < np.array([0.05, 0.05]))
            height_delta = obj_p[2] - float(origin_z)
            return height_delta > 0.007 and height_delta < 0.12 and tag and gripper_open
        target_p = np.array(target.get_pose().p)
        xy_dist = np.linalg.norm(obj_p[:2] - target_p[:2])
        height_ok = obj_p[2] >= target_p[2] - 0.02
        if self.active_task.relation == "in":
            return xy_dist < 0.16 and height_ok
        return xy_dist < 0.12 and height_ok
