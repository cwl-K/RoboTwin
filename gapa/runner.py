"""Runtime orchestration for the GAPA MVP."""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v2 as imageio
import numpy as np
import yaml

from .llm_client import LLMClient
from .object_registry import object_options, validate_object_names
from .perception import OraclePerception, VLMPerception
from .planner import TaskPlanner
from .program_api import ProgramCandidate, execute_program_candidate
from .program_codegen import ProgramCodeGenerator
from .task_dsl import TaskDSL

if TYPE_CHECKING:
    from envs.gapa_scene import GapaScene


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs_gapa"
TASK_CONFIG_PATH = ROOT / "task_config" / "gapa_scene.yml"
EMBODIMENT_CONFIG_PATH = ROOT / "task_config" / "_embodiment_config.yml"


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, default=_json_default) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_scene_args(
    seed: int,
    save_path: Path | None = None,
    render_freq: int = 0,
    object_names: list[str] | None = None,
) -> dict[str, Any]:
    with TASK_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)

    with EMBODIMENT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)

    embodiment = args.get("embodiment", ["aloha-agilex"])
    if isinstance(embodiment, str):
        embodiment = [embodiment]
    if len(embodiment) != 1:
        raise ValueError("GAPA MVP supports one symmetric embodiment in gapa_scene.yml.")

    def embodiment_file(name: str) -> str:
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file configured for {name}")
        return robot_file if os.path.isabs(robot_file) else str((ROOT / robot_file).resolve())

    robot_file = embodiment_file(embodiment[0])
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["left_embodiment_config"] = _load_robot_config(robot_file)
    args["right_embodiment_config"] = _load_robot_config(robot_file)
    args["dual_arm_embodied"] = True
    args["embodiment_name"] = embodiment[0]
    args["task_name"] = "gapa_scene"
    args["seed"] = seed
    args["now_ep_num"] = 0
    args["render_freq"] = render_freq
    args["need_plan"] = True
    args["save_data"] = False
    args["gapa_object_names"] = object_names or []
    if save_path is not None:
        args["save_path"] = str(save_path)
    return args


def _load_robot_config(robot_file: str) -> dict[str, Any]:
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as handle:
        return yaml.load(handle.read(), Loader=yaml.FullLoader)


class GapaRunner:
    """Single-user runtime for random scenes and task execution."""

    def __init__(self, runs_root: Path = RUNS_ROOT):
        self.runs_root = runs_root
        self.planner = TaskPlanner(use_llm=True)
        self.oracle_perception = OraclePerception()
        self.vlm_perception = VLMPerception()
        self.current_env: GapaScene | None = None
        self.current_scene_seed: int | None = None
        self.current_scene: dict[str, Any] | None = None
        self.current_object_names: list[str] | None = None
        self.current_run_id: str | None = None

    def scene_options(self) -> dict[str, Any]:
        return {"objects": object_options()}

    def test_llm_api(self) -> dict[str, Any]:
        self.planner.llm_client = LLMClient()
        client = self.planner.llm_client
        config = client.config
        if not client.is_configured:
            raise ValueError("GAPA LLM is not configured. Check gapa/gapa_api.env.")
        raw = client.chat([
            {"role": "system", "content": "You are a connectivity test endpoint. Reply briefly."},
            {"role": "user", "content": "Return exactly: GAPA_LLM_OK"},
        ])
        return {
            "ok": True,
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
            "response_preview": raw[:200],
        }

    def randomize_scene(self, seed: int | None = None, object_names: list[str] | None = None) -> dict[str, Any]:
        self._close_current_env()
        selected = validate_object_names(object_names)
        seed = int(seed if seed is not None else time.time_ns() % 1_000_000)
        env = self._create_env(seed=seed, save_path=self.runs_root / "_scene_cache", object_names=selected)
        scene = env.get_scene_description()
        preview_images = self._save_scene_previews(env, seed)
        self.current_env = env
        self.current_scene_seed = seed
        self.current_scene = scene
        self.current_object_names = selected
        return {
            "seed": seed,
            "selected_objects": selected,
            "objects": scene,
            "preview_images": preview_images,
        }

    def run_task(self, instruction: str) -> dict[str, Any]:
        if self.current_env is None or self.current_scene is None or self.current_scene_seed is None:
            raise ValueError("Generate a scene before running a task.")

        assert self.current_env is not None
        assert self.current_scene is not None
        assert self.current_scene_seed is not None

        run_id = self._new_run_id()
        self.current_run_id = run_id
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        scene_data = {
            "seed": self.current_scene_seed,
            "selected_objects": self.current_object_names,
            "objects": self.current_scene,
        }
        _write_json(run_dir / "scene.json", scene_data)

        parse_result = self.planner.parse(instruction, self.current_scene)
        dsl = parse_result.dsl
        _write_json(run_dir / "task_dsl.json", {
            **dsl.to_dict(),
            "parse_source": parse_result.source,
            "llm_attempted": parse_result.llm_attempted,
        })
        if not dsl.feasible:
            record = {
                "run_id": run_id,
                "status": "infeasible",
                "reason": dsl.reason,
                "task_dsl": dsl.to_dict(),
            }
            _append_jsonl(run_dir / "attempts.jsonl", record)
            _write_json(run_dir / "summary.json", record)
            return self.get_run(run_id)

        candidates = ProgramCodeGenerator(self.planner.llm_client).generate_programs(
            instruction=instruction,
            task=dsl,
            scene_objects=self.current_scene,
        )
        self._write_program_candidates(run_dir, candidates)

        validation = self._validate_program_candidates(candidates, dsl)
        best_program = validation["best_program"]
        _write_json(run_dir / "validation.json", {
            "results": validation["results"],
            "best_program_id": best_program.program_id if best_program else None,
        })

        if best_program is None:
            summary = {
                "run_id": run_id,
                "status": "failed",
                "reason": "No candidate program could be selected.",
            }
            _append_jsonl(run_dir / "attempts.jsonl", summary)
            _write_json(run_dir / "summary.json", summary)
            return self.get_run(run_id)

        self._enable_collect_data_video(self.current_env, run_dir)
        execution = self._execute_program_once(best_program, dsl, run_dir)
        video_path = self._build_video(run_dir, self.current_env)
        summary = {
            "run_id": run_id,
            "status": execution["status"],
            "instruction": instruction,
            "task_dsl": dsl.to_dict(),
            "best_program_id": best_program.program_id,
            "best_program_path": best_program.path,
            "program_source": (best_program.metadata or {}).get("program_source"),
            "validation": validation["results"],
            "video": self._public_path(video_path) if video_path else None,
            "vlm_status": self.vlm_perception.locate(self.current_env, dsl.object_name),
        }
        _write_json(run_dir / "summary.json", summary)
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.runs_root / run_id
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {
            "run_id": run_id,
            "status": "unknown",
        }
        images = [self._public_path(path) for path in sorted(run_dir.glob("gapa/current/*.png"))]
        attempts = _read_jsonl(run_dir / "attempts.jsonl")
        if "video" not in summary:
            video = run_dir / "demo.mp4"
            summary["video"] = self._public_path(video) if video.exists() else None
        return {
            **summary,
            "attempts": attempts,
            "images": images,
            "run_dir": str(run_dir),
        }

    def _create_env(
        self,
        seed: int,
        save_path: Path,
        render_freq: int = 0,
        object_names: list[str] | None = None,
    ) -> "GapaScene":
        from envs.gapa_scene import GapaScene

        env = GapaScene()
        args = _load_scene_args(seed=seed, save_path=save_path, render_freq=render_freq, object_names=object_names)
        env.setup_demo(**args)
        return env

    def _save_scene_previews(self, env: GapaScene, seed: int) -> dict[str, dict[str, str]]:
        preview_dir = self.runs_root / "_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        camera_labels = {
            "left_camera": "Left wrist",
            "right_camera": "Right wrist",
            "head_camera": "Head",
            "world_camera": "World",
        }
        image_paths = {
            camera_name: preview_dir / f"scene_{seed}_{camera_name}.png"
            for camera_name in camera_labels
        }

        env._update_render()
        env.cameras.update_picture()
        rgb = env.cameras.get_rgb()
        for camera_name in ("left_camera", "right_camera", "head_camera"):
            imageio.imwrite(image_paths[camera_name], rgb[camera_name]["rgb"])

        imageio.imwrite(image_paths["world_camera"], self._capture_world_camera_rgb(env))
        return {
            camera_name: {
                "label": label,
                "url": self._public_path(image_paths[camera_name]),
            }
            for camera_name, label in camera_labels.items()
        }

    def _capture_world_camera_rgb(self, env: GapaScene) -> np.ndarray:
        camera = getattr(env.cameras, "world_camera1", None)
        if camera is None:
            return env.cameras.get_observer_rgb()
            camera.take_picture()
        rgba = camera.get_picture("Color")
        return (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]

    def _write_program_candidates(self, run_dir: Path, candidates: list[ProgramCandidate]) -> None:
        programs_dir = run_dir / "programs"
        programs_dir.mkdir(parents=True, exist_ok=True)
        for index, candidate in enumerate(candidates, start=1):
            path = programs_dir / f"candidate_{index}.py"
            path.write_text(candidate.source, encoding="utf-8")
            candidate.path = self._public_path(path)
        _write_json(run_dir / "candidate_programs.json", [candidate.to_dict() for candidate in candidates])

    def _validate_program_candidates(self, candidates: list[ProgramCandidate], dsl: TaskDSL) -> dict[str, Any]:
        validation_seeds = [11, 23, 37]
        results = []
        best_program = None
        best_score = -1.0

        for candidate in candidates:
            success_count = 0
            errors = []
            for seed in validation_seeds:
                env = None
                try:
                    env = self._create_env(
                        seed=seed,
                        save_path=self.runs_root / "_validation",
                        object_names=self.current_object_names,
                    )
                    failure = execute_program_candidate(candidate, env, dsl)
                    success = failure is None
                    success_count += int(success)
                    errors.append(None if success else failure.to_dict())
                except Exception as exc:
                    errors.append({"stage": "exception", "message": str(exc), "traceback": traceback.format_exc()})
                finally:
                    if env is not None:
                        env.close()
            score = success_count / len(validation_seeds)
            result = {
                "program_id": candidate.program_id,
                "success_count": success_count,
                "total": len(validation_seeds),
                "score": score,
                "errors": errors,
            }
            results.append(result)
            if score > best_score:
                best_score = score
                best_program = candidate

        if best_score <= 0:
            best_program = None
        return {"results": results, "best_program": best_program}

    def _execute_program_once(self, candidate: ProgramCandidate, dsl: TaskDSL, run_dir: Path) -> dict[str, Any]:
        assert self.current_env is not None
        failure = execute_program_candidate(
            candidate,
            self.current_env,
            dsl,
            run_dir=str(run_dir),
            generate_id="current",
            attempt_id=1,
        )
        record = {
            "attempt_id": 1,
            "program_id": candidate.program_id,
            "status": "success" if failure is None else "failed",
            "failure": None if failure is None else failure.to_dict(),
        }
        _append_jsonl(run_dir / "attempts.jsonl", record)
        if failure is None:
            return {"status": "success", "attempt_id": 1}
        return {"status": "failed", "failure": failure.to_dict()}

    def _enable_collect_data_video(self, env: "GapaScene", run_dir: Path) -> None:
        env.save_data = True
        env.save_freq = 5
        env.save_dir = str(run_dir / "trajectory")
        env.ep_num = 0
        env.FRAME_IDX = 0
        if hasattr(env, "folder_path"):
            delattr(env, "folder_path")

    def _build_video(self, run_dir: Path, env: "GapaScene" | None = None) -> Path | None:
        collect_video = self._build_collect_data_video(run_dir, env)
        if collect_video is not None:
            return collect_video

        image_files = sorted((run_dir / "gapa" / "current").glob("*.png"))
        if not image_files:
            return None
        frames = []
        for image_file in image_files:
            frames.append(imageio.imread(image_file))
        video_path = run_dir / "demo.mp4"
        try:
            _images_to_video(np.asarray(frames), video_path, fps=2.0)
            return video_path
        except Exception:
            fallback = run_dir / "video_error.txt"
            fallback.write_text(traceback.format_exc(), encoding="utf-8")
            return None

    def _build_collect_data_video(self, run_dir: Path, env: "GapaScene" | None) -> Path | None:
        if env is None or not getattr(env, "save_data", False) or not hasattr(env, "folder_path"):
            return None
        try:
            env.merge_pkl_to_hdf5_video()
            source_video = Path(env.save_dir) / "video" / f"episode{env.ep_num}.mp4"
            if not source_video.exists():
                return None
            target_video = run_dir / "demo.mp4"
            shutil.copyfile(source_video, target_video)
            return target_video
        except Exception:
            fallback = run_dir / "collect_video_error.txt"
            fallback.write_text(traceback.format_exc(), encoding="utf-8")
            return None

    def _new_run_id(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]

    def _public_path(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            rel = path.resolve().relative_to(self.runs_root.resolve())
            return f"/runs_gapa/{rel.as_posix()}"
        except ValueError:
            return str(path)

    def _close_current_env(self) -> None:
        if self.current_env is not None:
            self.current_env.close()
        self.current_env = None
        self.current_scene = None
        self.current_scene_seed = None
        self.current_object_names = None


RUNNER = GapaRunner()


def _images_to_video(imgs: np.ndarray, out_path: Path, fps: float = 2.0) -> None:
    import subprocess

    if imgs.ndim != 4 or imgs.shape[3] not in (3, 4):
        raise ValueError("imgs must have shape (N, H, W, C).")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, height, width, channels = imgs.shape
    pixel_format = "rgb24" if channels == 3 else "rgba"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            pixel_format,
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None
    ffmpeg.stdin.write(imgs.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise IOError("ffmpeg failed while writing GAPA demo video.")
