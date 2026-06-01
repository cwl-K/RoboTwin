import math
import json
import unittest

import numpy as np

from gapa.object_registry import OBJECT_SPECS, SELECTABLE_OBJECTS, object_options, validate_object_names
from gapa.planner import TaskPlanner

try:
    from envs.gapa_scene import (
        NON_OVERLAP_MARGIN,
        SOURCE_CENTER_X_EXCLUSION,
        SOURCE_X_RANGE,
        SOURCE_Y_RANGE,
        TARGET_X_RANGE,
        TARGET_Y_RANGE,
        _sample_scene_layout,
        _select_scene_specs,
    )
except ModuleNotFoundError as exc:
    if exc.name != "sapien":
        raise
    GAPA_SCENE_AVAILABLE = False
else:
    GAPA_SCENE_AVAILABLE = True


class FakeLLMClient:

    def __init__(self, response, configured=True):
        self.responses = response if isinstance(response, list) else [response]
        self.messages = []
        self.is_configured = configured

    def chat(self, messages, temperature=0.0):
        self.messages.append(messages)
        index = min(len(self.messages) - 1, len(self.responses) - 1)
        return self.responses[index]


def parse_response(object_name, target_name, relation):
    return json.dumps({
        "object_name": object_name,
        "target_name": target_name,
        "relation": relation,
    })


class GapaPlannerTest(unittest.TestCase):
    def setUp(self):
        self.scene = {
            "cup": {"roles": ["source", "target"]},
            "bowl": {"roles": ["source", "target"]},
            "plate": {"roles": ["target"]},
            "red_block": {"roles": ["source", "target"]},
            "green_block": {"roles": ["source", "target"]},
            "blue_block": {"roles": ["source", "target"]},
        }

    def test_parse_english_put_on(self):
        planner = TaskPlanner(llm_client=FakeLLMClient(parse_response("cup", "plate", "on")), use_llm=True)
        result = planner.parse("put cup on plate", self.scene)
        self.assertTrue(result.dsl.feasible)
        self.assertEqual(result.dsl.object_name, "cup")
        self.assertEqual(result.dsl.target_name, "plate")
        self.assertEqual(result.dsl.relation, "on")

    def test_parse_chinese_put_on(self):
        planner = TaskPlanner(llm_client=FakeLLMClient(parse_response("cup", "plate", "on")), use_llm=True)
        result = planner.parse("把杯子放到盘子上", self.scene)
        self.assertTrue(result.dsl.feasible)
        self.assertEqual(result.dsl.object_name, "cup")
        self.assertEqual(result.dsl.target_name, "plate")
        self.assertEqual(result.dsl.relation, "on")

    def test_parse_colored_blocks(self):
        planner = TaskPlanner(llm_client=FakeLLMClient(parse_response("red_block", "green_block", "on")), use_llm=True)
        result = planner.parse("place red block on green block", self.scene)
        self.assertTrue(result.dsl.feasible)
        self.assertEqual(result.dsl.object_name, "red_block")
        self.assertEqual(result.dsl.target_name, "green_block")
        self.assertEqual(result.dsl.relation, "on")

    def test_infeasible_missing_object(self):
        planner = TaskPlanner(llm_client=FakeLLMClient(parse_response("spoon", "basket", "in")), use_llm=True)
        result = planner.parse("put the spoon in the basket", self.scene)
        self.assertFalse(result.dsl.feasible)
        self.assertIn("Unsupported", result.dsl.reason)

    def test_infeasible_known_object_missing_from_scene(self):
        scene = {
            "cup": {"roles": ["source", "target"]},
            "plate": {"roles": ["target"]},
        }
        planner = TaskPlanner(llm_client=FakeLLMClient(parse_response("bowl", "plate", "on")), use_llm=True)
        result = planner.parse("put bowl on plate", scene)
        self.assertFalse(result.dsl.feasible)
        self.assertIn("Current scene does not contain: bowl", result.dsl.reason)

    def test_infeasible_non_graspable_source(self):
        planner = TaskPlanner(llm_client=FakeLLMClient(parse_response("plate", "bowl", "on")), use_llm=True)
        result = planner.parse("put plate on bowl", self.scene)
        self.assertFalse(result.dsl.feasible)
        self.assertIn("Unsupported or missing source object", result.dsl.reason)

    def test_llm_is_required_for_parsing(self):
        planner = TaskPlanner(llm_client=FakeLLMClient("{}", configured=False), use_llm=True)
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            planner.parse("put cup on plate", self.scene)


class GapaRegistryTest(unittest.TestCase):
    def test_registry_contains_only_new_objects(self):
        self.assertEqual(set(SELECTABLE_OBJECTS), {"cup", "bowl", "plate", "red_block", "green_block", "blue_block"})
        self.assertEqual({option["name"] for option in object_options()}, set(SELECTABLE_OBJECTS))

    def test_validate_object_names_rejects_empty_and_unknown(self):
        with self.assertRaisesRegex(ValueError, "Select at least one"):
            validate_object_names([])
        with self.assertRaisesRegex(ValueError, "Unknown GAPA object"):
            validate_object_names(["cup", "bottle"])


@unittest.skipUnless(GAPA_SCENE_AVAILABLE, "SAPIEN is not installed")
class GapaSceneLayoutTest(unittest.TestCase):
    def test_scene_selection_uses_requested_objects(self):
        np.random.seed(7)
        selected = ["cup", "plate", "red_block"]
        specs = _select_scene_specs(selected)

        self.assertEqual([alias for alias, _ in specs], selected)
        self.assertEqual(len({alias for alias, _ in specs}), len(selected))

    def test_sample_non_overlapping_layout(self):
        np.random.seed(13)
        selected = [(alias, OBJECT_SPECS[alias]) for alias in SELECTABLE_OBJECTS]
        placements = _sample_scene_layout(selected)

        accepted = {}
        for alias, spec in selected:
            x, y = placements[alias]
            if alias == "plate":
                self.assertGreaterEqual(x, TARGET_X_RANGE[0])
                self.assertLessEqual(x, TARGET_X_RANGE[1])
                self.assertGreaterEqual(y, TARGET_Y_RANGE[0])
                self.assertLessEqual(y, TARGET_Y_RANGE[1])
            else:
                self.assertGreaterEqual(x, SOURCE_X_RANGE[0])
                self.assertLessEqual(x, SOURCE_X_RANGE[1])
                self.assertGreaterEqual(y, SOURCE_Y_RANGE[0])
                self.assertLessEqual(y, SOURCE_Y_RANGE[1])
                self.assertGreaterEqual(abs(x), SOURCE_CENTER_X_EXCLUSION)
            for other_alias, (other_x, other_y, other_radius) in accepted.items():
                distance = math.hypot(x - other_x, y - other_y)
                min_distance = spec.footprint_radius + other_radius + NON_OVERLAP_MARGIN
                self.assertGreater(distance, min_distance, f"{alias} overlaps {other_alias}")
            accepted[alias] = (x, y, spec.footprint_radius)

        self.assertEqual(len(accepted), len(SELECTABLE_OBJECTS))


if __name__ == "__main__":
    unittest.main()
