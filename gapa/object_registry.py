"""Selectable object registry for the GAPA MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ObjectRole = Literal["source", "target"]
TargetRelation = Literal["in", "on"]


@dataclass(frozen=True)
class GapaObjectSpec:
    alias: str
    label: str
    modelname: str
    model_id: int | None
    roles: tuple[ObjectRole, ...]
    qpos: list[float]
    footprint_radius: float
    aliases: tuple[str, ...]
    target_relations: tuple[TargetRelation, ...] = ()
    default_relation: TargetRelation = "on"
    convex: bool = True
    is_static: bool = False
    mass: float = 0.03
    kind: Literal["actor", "box"] = "actor"
    half_size: tuple[float, float, float] | None = None
    color: tuple[float, float, float] | None = None
    z: float = 0.741
    target_z_offset: float = 0.05

    @property
    def can_grasp(self) -> bool:
        return "source" in self.roles

    @property
    def can_target(self) -> bool:
        return "target" in self.roles


OBJECT_SPECS: dict[str, GapaObjectSpec] = {
    "cup": GapaObjectSpec(
        alias="cup",
        label="Cup",
        modelname="021_cup",
        model_id=1,
        roles=("source", "target"),
        qpos=[0.5, 0.5, 0.5, 0.5],
        footprint_radius=0.06,
        aliases=("cup", "杯子", "杯"),
        target_relations=("in", "on"),
        default_relation="in",
        mass=0.08,
        target_z_offset=0.08,
    ),
    "bowl": GapaObjectSpec(
        alias="bowl",
        label="Bowl",
        modelname="002_bowl",
        model_id=3,
        roles=("source", "target"),
        qpos=[0.5, 0.5, 0.5, 0.5],
        footprint_radius=0.09,
        aliases=("bowl", "碗"),
        target_relations=("in", "on"),
        default_relation="in",
        mass=0.08,
        target_z_offset=0.06,
    ),
    "plate": GapaObjectSpec(
        alias="plate",
        label="Plate",
        modelname="003_plate",
        model_id=0,
        roles=("target",),
        qpos=[0.5, 0.5, 0.5, 0.5],
        footprint_radius=0.12,
        aliases=("plate", "盘子", "盘"),
        target_relations=("on",),
        default_relation="on",
        is_static=True,
        mass=0.2,
    ),
    "red_block": GapaObjectSpec(
        alias="red_block",
        label="Red block",
        modelname="box",
        model_id=None,
        roles=("source", "target"),
        qpos=[1.0, 0.0, 0.0, 0.0],
        footprint_radius=0.04,
        aliases=("red block", "red_block", "red cube", "红色方块", "红方块", "红色积木", "红块"),
        target_relations=("on",),
        default_relation="on",
        kind="box",
        half_size=(0.025, 0.025, 0.025),
        color=(1.0, 0.0, 0.0),
        z=0.766,
        mass=0.02,
    ),
    "green_block": GapaObjectSpec(
        alias="green_block",
        label="Green block",
        modelname="box",
        model_id=None,
        roles=("source", "target"),
        qpos=[1.0, 0.0, 0.0, 0.0],
        footprint_radius=0.04,
        aliases=("green block", "green_block", "green cube", "绿色方块", "绿方块", "绿色积木", "绿块"),
        target_relations=("on",),
        default_relation="on",
        kind="box",
        half_size=(0.025, 0.025, 0.025),
        color=(0.0, 0.75, 0.1),
        z=0.766,
        mass=0.02,
    ),
    "blue_block": GapaObjectSpec(
        alias="blue_block",
        label="Blue block",
        modelname="box",
        model_id=None,
        roles=("source", "target"),
        qpos=[1.0, 0.0, 0.0, 0.0],
        footprint_radius=0.04,
        aliases=("blue block", "blue_block", "blue cube", "蓝色方块", "蓝方块", "蓝色积木", "蓝块"),
        target_relations=("on",),
        default_relation="on",
        kind="box",
        half_size=(0.025, 0.025, 0.025),
        color=(0.0, 0.2, 1.0),
        z=0.766,
        mass=0.02,
    ),
}


SELECTABLE_OBJECTS = tuple(OBJECT_SPECS)
SOURCE_OBJECTS = tuple(name for name, spec in OBJECT_SPECS.items() if spec.can_grasp)
TARGET_OBJECTS = tuple(name for name, spec in OBJECT_SPECS.items() if spec.can_target)
OBJECT_ALIASES = {name: spec.aliases for name, spec in OBJECT_SPECS.items()}
RELATION_DEFAULTS = {name: spec.default_relation for name, spec in OBJECT_SPECS.items() if spec.can_target}
MAX_SELECTED_OBJECTS = len(SELECTABLE_OBJECTS)


def get_object_spec(name: str) -> GapaObjectSpec:
    try:
        return OBJECT_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown GAPA object: {name}") from exc


def validate_object_names(names: list[str] | tuple[str, ...] | None) -> list[str]:
    selected = list(dict.fromkeys(names or []))
    if not selected:
        raise ValueError("Select at least one GAPA object before generating a scene.")
    if len(selected) > MAX_SELECTED_OBJECTS:
        raise ValueError(f"Select at most {MAX_SELECTED_OBJECTS} GAPA objects.")
    unknown = [name for name in selected if name not in OBJECT_SPECS]
    if unknown:
        raise ValueError(f"Unknown GAPA object(s): {', '.join(unknown)}.")
    return selected


def object_options() -> list[dict[str, object]]:
    return [
        {
            "name": spec.alias,
            "label": spec.label,
            "modelname": spec.modelname,
            "model_id": spec.model_id,
            "roles": list(spec.roles),
            "target_relations": list(spec.target_relations),
        }
        for spec in OBJECT_SPECS.values()
    ]
