"""Selectable object registry for the GAPA MVP."""

from __future__ import annotations

import math
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
    kind: Literal["actor", "box", "urdf"] = "actor"
    half_size: tuple[float, float, float] | None = None
    color: tuple[float, float, float] | None = None
    z: float = 0.741
    target_z_offset: float = 0.05
    rotate_rand: bool = False
    rotate_lim: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def can_grasp(self) -> bool:
        return "source" in self.roles

    @property
    def can_target(self) -> bool:
        return "target" in self.roles


OFFICIAL_CABINET_SOURCE_OBJECTS = (
    "mouse",
    "stapler",
    "toy_car",
    "rubiks_cube",
    "bread",
    "phone",
    "playing_cards",
    "tea_box",
    "coffee_box",
    "soap",
)


def _official_cabinet_source(
    *,
    alias: str,
    label: str,
    modelname: str,
    aliases: tuple[str, ...],
    footprint_radius: float = 0.06,
) -> GapaObjectSpec:
    return GapaObjectSpec(
        alias=alias,
        label=label,
        modelname=modelname,
        model_id=0,
        roles=("source",),
        qpos=[0.707, 0.707, 0.0, 0.0],
        footprint_radius=footprint_radius,
        aliases=aliases,
        mass=0.01,
        rotate_rand=True,
        rotate_lim=(0.0, math.pi / 3.0, 0.0),
    )


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
    "cabinet": GapaObjectSpec(
        alias="cabinet",
        label="Cabinet drawer",
        modelname="036_cabinet",
        model_id=46653,
        roles=("target",),
        qpos=[1.0, 0.0, 0.0, 1.0],
        footprint_radius=0.14,
        aliases=("cabinet", "drawer", "cabinet drawer", "抽屉", "柜子", "柜子的抽屉"),
        target_relations=("in",),
        default_relation="in",
        kind="urdf",
        mass=0.0,
    ),
    "mouse": _official_cabinet_source(
        alias="mouse",
        label="Mouse",
        modelname="047_mouse",
        aliases=("mouse", "computer mouse", "鼠标"),
        footprint_radius=0.055,
    ),
    "stapler": _official_cabinet_source(
        alias="stapler",
        label="Stapler",
        modelname="048_stapler",
        aliases=("stapler", "订书机"),
        footprint_radius=0.065,
    ),
    "toy_car": _official_cabinet_source(
        alias="toy_car",
        label="Toy car",
        modelname="057_toycar",
        aliases=("toy car", "toy_car", "toycar", "car", "玩具车", "小车"),
        footprint_radius=0.075,
    ),
    "rubiks_cube": _official_cabinet_source(
        alias="rubiks_cube",
        label="Rubik's cube",
        modelname="073_rubikscube",
        aliases=("rubik's cube", "rubiks cube", "rubik cube", "rubikscube", "magic cube", "魔方"),
        footprint_radius=0.055,
    ),
    "bread": _official_cabinet_source(
        alias="bread",
        label="Bread",
        modelname="075_bread",
        aliases=("bread", "面包"),
        footprint_radius=0.065,
    ),
    "phone": _official_cabinet_source(
        alias="phone",
        label="Phone",
        modelname="077_phone",
        aliases=("phone", "mobile phone", "cell phone", "手机", "电话"),
        footprint_radius=0.070,
    ),
    "playing_cards": _official_cabinet_source(
        alias="playing_cards",
        label="Playing cards",
        modelname="081_playingcards",
        aliases=("playing cards", "playing_cards", "playingcards", "cards", "扑克牌", "纸牌"),
        footprint_radius=0.060,
    ),
    "tea_box": _official_cabinet_source(
        alias="tea_box",
        label="Tea box",
        modelname="112_tea-box",
        aliases=("tea box", "tea_box", "tea-box", "茶盒", "茶叶盒"),
        footprint_radius=0.065,
    ),
    "coffee_box": _official_cabinet_source(
        alias="coffee_box",
        label="Coffee box",
        modelname="113_coffee-box",
        aliases=("coffee box", "coffee_box", "coffee-box", "咖啡盒"),
        footprint_radius=0.065,
    ),
    "soap": _official_cabinet_source(
        alias="soap",
        label="Soap",
        modelname="107_soap",
        aliases=("soap", "肥皂", "香皂"),
        footprint_radius=0.055,
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
        rotate_rand=True,
        rotate_lim=(0.0, 0.0, 0.75),
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
        rotate_rand=True,
        rotate_lim=(0.0, 0.0, 0.75),
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
        rotate_rand=True,
        rotate_lim=(0.0, 0.0, 0.75),
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


def canonical_object_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", " ")
    for object_name, spec in OBJECT_SPECS.items():
        candidates = {object_name, object_name.replace("_", " "), *(alias.lower() for alias in spec.aliases)}
        if normalized in candidates:
            return object_name
    return name


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
