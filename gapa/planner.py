"""LLM-backed task parsing for the GAPA MVP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm_client import LLMClient
from .object_registry import OBJECT_SPECS, SOURCE_OBJECTS, TARGET_OBJECTS
from .task_dsl import TaskDSL


@dataclass(frozen=True)
class ParseResult:
    dsl: TaskDSL
    source: str
    llm_attempted: bool = False


def _extract_json(raw: str) -> Any:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates = [(index, open_char) for index, open_char in ((text.find("{"), "{"), (text.find("["), "[")) if index >= 0]
    if not candidates:
        raise ValueError("LLM response did not contain JSON.")
    start, open_char = min(candidates, key=lambda item: item[0])
    close_char = "}" if open_char == "{" else "]"
    end = text.rfind(close_char)
    if end < start:
        raise ValueError("LLM response JSON was incomplete.")
    return json.loads(text[start:end + 1])


def _object_prompt_lines(scene_objects: dict[str, dict] | None = None) -> str:
    scene_names = set(scene_objects or {})
    lines = []
    for name, spec in OBJECT_SPECS.items():
        scene_marker = "present" if not scene_objects or name in scene_names else "not_in_current_scene"
        lines.append(
            f"- {name}: aliases={list(spec.aliases)}, roles={list(spec.roles)}, "
            f"target_relations={list(spec.target_relations)}, scene={scene_marker}"
        )
    return "\n".join(lines)


class TaskPlanner:
    def __init__(self, llm_client: LLMClient | None = None, use_llm: bool = False):
        self.llm_client = llm_client or LLMClient()
        self.use_llm = use_llm

    def parse(self, text: str, scene_objects: dict[str, dict] | None = None) -> ParseResult:
        if not self.use_llm:
            raise RuntimeError("GAPA planner requires LLM; rules fallback is disabled.")
        if not self.llm_client.is_configured:
            raise RuntimeError("GAPA LLM is not configured. Check gapa/gapa_api.env.")
        return self._parse_with_llm(text, scene_objects)

    def _parse_with_llm(self, text: str, scene_objects: dict[str, dict] | None) -> ParseResult:
        prompt = (
            "Parse the robot task into JSON with keys object_name, target_name, relation. "
            "Allowed relation values are in and on. Return only JSON.\n\n"
            f"Objects and aliases:\n{_object_prompt_lines(scene_objects)}\n\n"
            f"Graspable source objects: {SOURCE_OBJECTS}.\n"
            f"Placement target objects: {TARGET_OBJECTS}.\n"
            f"Task: {text}"
        )
        raw = self.llm_client.chat([
            {"role": "system", "content": "You parse simple grasp-and-place robot tasks."},
            {"role": "user", "content": prompt},
        ])
        data = _extract_json(raw)
        if not isinstance(data, dict):
            raise RuntimeError("LLM task parse response must be a JSON object.")
        dsl = TaskDSL(
            raw_text=text,
            object_name=data["object_name"],
            target_name=data["target_name"],
            relation=data["relation"],
        )
        return ParseResult(self._validate(dsl, scene_objects), "llm", True)

    def _validate(self, dsl: TaskDSL, scene_objects: dict[str, dict] | None) -> TaskDSL:
        if dsl.object_name not in SOURCE_OBJECTS:
            dsl.feasible = False
            dsl.reason = f"Unsupported or missing source object. Supported: {', '.join(SOURCE_OBJECTS)}."
            return dsl
        if dsl.target_name not in TARGET_OBJECTS:
            dsl.feasible = False
            dsl.reason = f"Unsupported or missing target. Supported: {', '.join(TARGET_OBJECTS)}."
            return dsl
        if dsl.object_name == dsl.target_name:
            dsl.feasible = False
            dsl.reason = "Source object and target must be different."
            return dsl
        target_spec = OBJECT_SPECS[dsl.target_name]
        if dsl.relation not in target_spec.target_relations:
            dsl.feasible = False
            dsl.reason = f"Target {dsl.target_name} does not support relation '{dsl.relation}'."
            return dsl
        if scene_objects is not None:
            missing = [name for name in (dsl.object_name, dsl.target_name) if name not in scene_objects]
            if missing:
                dsl.feasible = False
                dsl.reason = f"Current scene does not contain: {', '.join(missing)}."
                return dsl
            object_roles = set(scene_objects[dsl.object_name].get("roles", []))
            target_roles = set(scene_objects[dsl.target_name].get("roles", []))
            if "source" not in object_roles:
                dsl.feasible = False
                dsl.reason = f"Current scene object {dsl.object_name} is not graspable."
                return dsl
            if "target" not in target_roles:
                dsl.feasible = False
                dsl.reason = f"Current scene object {dsl.target_name} is not a placement target."
        return dsl
