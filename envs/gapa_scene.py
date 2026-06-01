from __future__ import annotations

from typing import Any, Literal

import numpy as np

from gapa.object_registry import GapaObjectSpec, OBJECT_SPECS, validate_object_names

_GAPA_RUNTIME_IMPORT_ERROR: Exception | None = None

try:
    import sapien.core as sapien

    from ._base_task import Base_Task
    from .utils import create_actor, create_box, rand_pose
except Exception as exc:  # pragma: no cover - exercised only when simulator deps are unavailable.
    sapien = None
    Base_Task = object
    create_actor = None
    create_box = None
    rand_pose = None
    _GAPA_RUNTIME_IMPORT_ERROR = exc


NON_OVERLAP_MARGIN = 0.02
PLACEMENT_ATTEMPTS = 50
SLOT_JITTER = 0.015
SOURCE_X_RANGE = (-0.28, 0.28)
SOURCE_Y_RANGE = (-0.10, 0.05)
TARGET_X_RANGE = (-0.08, 0.08)
TARGET_Y_RANGE = (-0.15, -0.10)
SOURCE_CENTER_X_EXCLUSION = 0.05
SOURCE_LARGE_SAFE_SLOTS = (
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
TARGET_SAFE_SLOTS = ((0.0, -0.13),)
TABLE_SAFE_SLOTS = SOURCE_SMALL_SAFE_SLOTS + TARGET_SAFE_SLOTS
PlacementRecord = tuple[str, float, float, float]
PlacementZone = Literal["source", "target"]


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
    if spec.can_target and not spec.can_grasp:
        return "target"
    return "source"


def _source_slots_for_spec(spec: GapaObjectSpec) -> tuple[tuple[float, float], ...]:
    if spec.footprint_radius >= 0.06:
        return SOURCE_LARGE_SAFE_SLOTS
    return SOURCE_SMALL_SAFE_SLOTS


def _slots_for_spec(spec: GapaObjectSpec) -> tuple[tuple[float, float], ...]:
    if _placement_zone(spec) == "target":
        return TARGET_SAFE_SLOTS
    return _source_slots_for_spec(spec)


def _is_in_spawn_zone(x: float, y: float, zone: PlacementZone) -> bool:
    if zone == "target":
        return TARGET_X_RANGE[0] <= x <= TARGET_X_RANGE[1] and TARGET_Y_RANGE[0] <= y <= TARGET_Y_RANGE[1]
    return (
        SOURCE_X_RANGE[0] <= x <= SOURCE_X_RANGE[1]
        and SOURCE_Y_RANGE[0] <= y <= SOURCE_Y_RANGE[1]
        and abs(x) >= SOURCE_CENTER_X_EXCLUSION
    )


def _sample_non_overlapping_pose(
    slots: tuple[tuple[float, float], ...],
    spec: GapaObjectSpec,
    accepted: list[PlacementRecord],
    attempts: int = PLACEMENT_ATTEMPTS,
    jitter: float = SLOT_JITTER,
) -> tuple[float, float]:
    zone = _placement_zone(spec)
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
    if len(source_specs) > len(SOURCE_SMALL_SAFE_SLOTS):
        raise ValueError(f"Select at most {len(SOURCE_SMALL_SAFE_SLOTS)} graspable GAPA objects.")
    if len(target_specs) > len(TARGET_SAFE_SLOTS):
        raise ValueError(f"Select at most {len(TARGET_SAFE_SLOTS)} target-only GAPA object.")

    accepted: list[PlacementRecord] = []
    placements = {}
    placement_order = sorted(
        target_specs + source_specs,
        key=lambda item: (0 if _placement_zone(item[1]) == "target" else 1, -item[1].footprint_radius),
    )
    for alias, spec in placement_order:
        x, y = _sample_non_overlapping_pose(_slots_for_spec(spec), spec, accepted)
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
        selected_specs = _select_scene_specs(self.gapa_object_names)
        placements = _sample_scene_layout(selected_specs)

        for alias, spec in selected_specs:
            x, y = placements[alias]
            pose = rand_pose(
                xlim=[x, x],
                ylim=[y, y],
                zlim=[spec.z],
                qpos=spec.qpos,
                rotate_rand=spec.kind == "box",
                rotate_lim=[0, 0, 0.75],
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
        target_p = np.array(target.get_pose().p)
        xy_dist = np.linalg.norm(obj_p[:2] - target_p[:2])
        height_ok = obj_p[2] >= target_p[2] - 0.02
        if self.active_task.relation == "in":
            return xy_dist < 0.16 and height_ok
        return xy_dist < 0.12 and height_ok
