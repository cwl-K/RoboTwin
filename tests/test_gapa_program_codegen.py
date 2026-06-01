import json
import unittest

import numpy as np

from gapa.program_api import ProgramCandidate, SafeSkillAPI, execute_program_candidate
from gapa.program_codegen import ProgramCodeGenerator
from gapa.program_safety import ProgramSafetyError, validate_program_source
from gapa.task_dsl import TaskDSL


VALID_SOURCE = """
def play_once(api):
    arm = api.choose_arm("cup")
    api.pose("cup")
    api.grasp("cup", arm=arm, pre_grasp_dis=0.09, grasp_dis=0.0)
    api.move_up(arm, z=0.08)
    api.place_on("cup", "plate", arm=arm, functional_point_id=0, pre_dis=0.08, dis=0.02, constrain="auto")
    api.move_up(arm, z=0.08, move_axis="arm")
""".strip()


class FakeLLMClient:
    def __init__(self, response, configured=True):
        self.response = response
        self.is_configured = configured

    def chat(self, messages, temperature=0.0):
        return self.response


def program_response():
    return json.dumps({
        "programs": [
            {
                "program_id": f"candidate_{index}",
                "description": f"candidate {index}",
                "source": VALID_SOURCE,
                "metadata": {"variant": f"v{index}"},
            }
            for index in range(1, 4)
        ]
    })


class FakePose:
    def __init__(self, p):
        self.p = np.array(p, dtype=float)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


class FakeActor:
    def __init__(self, p):
        self._pose = FakePose(p)

    def get_pose(self):
        return self._pose


class FakeEnv:
    def __init__(self):
        self.plan_success = True
        self.save_data = False
        self.active_task = None
        self.active_plan = "previous"
        self.calls = []
        self.actors = {
            "cup": FakeActor([-0.1, 0.0, 0.76]),
            "plate": FakeActor([0.0, -0.13, 0.74]),
        }
        self.gapa_specs = {}

    def get_actor(self, name):
        return self.actors[name]

    def get_target_pose(self, target, relation="on"):
        self.calls.append(("get_target_pose", target, relation))
        return self.actors[target].get_pose()

    def grasp_actor(self, actor, **kwargs):
        self.calls.append(("grasp_actor", kwargs))
        return kwargs["arm_tag"], ["grasp"]

    def move_by_displacement(self, **kwargs):
        self.calls.append(("move_by_displacement", kwargs))
        return kwargs["arm_tag"], ["move_up"]

    def place_actor(self, actor, **kwargs):
        self.calls.append(("place_actor", kwargs))
        return kwargs["arm_tag"], ["place"]

    def back_to_origin(self, arm_tag):
        self.calls.append(("back_to_origin", arm_tag))
        return arm_tag, ["origin"]

    def move(self, *actions):
        self.calls.append(("move", actions))
        return True

    def check_success(self):
        return True


class ProgramSafetyTest(unittest.TestCase):
    def test_valid_program_passes(self):
        report = validate_program_source(VALID_SOURCE)
        self.assertTrue(report.ok)

    def test_invalid_programs_are_rejected(self):
        invalid_sources = [
            "import os\ndef play_once(api):\n    pass",
            "def play_once(api):\n    open('x')",
            "def play_once(api):\n    eval('1')",
            "def play_once(api):\n    grasp('cup')",
            "def play_once(api):\n    api.fly('cup')",
            "def play_once(api):\n    for i in [1]:\n        api.pose('cup')",
            "class X:\n    pass\ndef play_once(api):\n    pass",
        ]
        for source in invalid_sources:
            with self.subTest(source=source):
                with self.assertRaises(ProgramSafetyError):
                    validate_program_source(source)


class ProgramCodegenTest(unittest.TestCase):
    def test_llm_generates_three_valid_programs(self):
        generator = ProgramCodeGenerator(FakeLLMClient(program_response()))
        dsl = TaskDSL("put cup on plate", "cup", "plate", "on")
        programs = generator.generate_programs("put cup on plate", dsl, {"cup": {}, "plate": {}})

        self.assertEqual(len(programs), 3)
        self.assertTrue(all(program.metadata["program_source"] == "llm" for program in programs))
        self.assertTrue(all(program.safety["ok"] for program in programs))

    def test_llm_not_configured_raises(self):
        generator = ProgramCodeGenerator(FakeLLMClient("{}", configured=False))
        dsl = TaskDSL("put cup on plate", "cup", "plate", "on")
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            generator.generate_programs("put cup on plate", dsl, {})

    def test_non_json_response_raises(self):
        generator = ProgramCodeGenerator(FakeLLMClient("not json"))
        dsl = TaskDSL("put cup on plate", "cup", "plate", "on")
        with self.assertRaisesRegex(ValueError, "LLM response"):
            generator.generate_programs("put cup on plate", dsl, {})

    def test_wrong_program_count_raises(self):
        generator = ProgramCodeGenerator(FakeLLMClient(json.dumps({"programs": []})))
        dsl = TaskDSL("put cup on plate", "cup", "plate", "on")
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            generator.generate_programs("put cup on plate", dsl, {})


class SafeSkillAPITest(unittest.TestCase):
    def test_safe_api_calls_robotwin_wrappers(self):
        env = FakeEnv()
        api = SafeSkillAPI(env)
        arm = api.choose_arm("cup")
        api.grasp("cup", arm=arm, pre_grasp_dis=0.1)
        api.move_up(arm, z=0.07)
        api.place_on("cup", "plate", arm=arm, pre_dis=0.09, dis=0.02)
        api.back_to_origin(arm)

        call_names = [call[0] for call in env.calls]
        self.assertIn("grasp_actor", call_names)
        self.assertIn("move_by_displacement", call_names)
        self.assertIn("place_actor", call_names)
        self.assertIn("back_to_origin", call_names)

    def test_execute_program_candidate(self):
        env = FakeEnv()
        dsl = TaskDSL("put cup on plate", "cup", "plate", "on")
        candidate = ProgramCandidate("candidate_1", VALID_SOURCE)
        failure = execute_program_candidate(candidate, env, dsl)

        self.assertIsNone(failure)
        self.assertIs(env.active_task, dsl)
        self.assertIsNone(env.active_plan)


if __name__ == "__main__":
    unittest.main()
