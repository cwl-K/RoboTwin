# GAPA MVP 代码思路

这份文档说明当前 GAPA MVP 的实现方式。当前版本的核心思路是让大模型生成受限版 `play_once(api)` 程序，而不是生成任意 Python expert code：

1. 前端让用户选择物体并生成一个 RoboTwin 仿真场景。
2. LLM 把自然语言任务解析成结构化 `TaskDSL`。
3. LLM 根据 `TaskDSL`、场景对象和安全 skill 签名生成 3 个 `play_once(api)` 候选程序。
4. Runner 在验证场景上试跑候选程序，选成功率最高的程序。
5. `SafeSkillAPI` 把生成代码里的 `api.*` 调用映射到 RoboTwin 高层 API。

## 入口

启动网页服务：

```bash
python -m uvicorn gapa.web_app:app --host 127.0.0.1 --port 7860
```

主要入口文件是 `gapa/web_app.py`。它提供：

- `GET /`：返回内嵌 HTML 页面。
- `GET /api/scene/options`：返回可勾选物体列表。
- `POST /api/llm/test`：用 `gapa/gapa_api.env` 的配置测试一次 LLM API。
- `POST /api/scene/randomize`：按用户选择生成当前场景。
- `POST /api/task/run`：执行自然语言任务。
- `GET /api/run/{run_id}`：读取一次运行的结果。
- `/runs_gapa/...`：静态访问截图、视频和 JSON 产物。

网页左侧是控制区：选择物体、测试 LLM API、生成场景、输入任务、执行任务。右侧显示四个初始相机视角、演示视频、对象列表和运行日志。

## 模块职责

`gapa/object_registry.py`

定义当前支持的物体池和能力。现在只包含 6 个物体：

- `cup`
- `bowl`
- `plate`
- `red_block`
- `green_block`
- `blue_block`

每个 `GapaObjectSpec` 记录资产名、模型 id、可抓/可作为目标、支持 `in/on` 哪些关系、中文英文别名、footprint 半径、质量和初始姿态等信息。前端物体按钮、planner 支持对象、场景加载都从这里取配置。

`envs/gapa_scene.py`

定义 RoboTwin 环境 `GapaScene`。它继承 `Base_Task`，负责：

- 接收 `gapa_object_names`。
- 根据用户勾选列表加载实际物体。
- 在可达桌面范围内采样初始位置。
- 用 footprint 半径做 XY 防重叠检查。
- 返回 scene description 给 planner。
- 提供 `get_actor()`、`get_target_pose()`、`check_success()` 等执行辅助接口。

当前采样范围参考了 `envs/blocks_ranking_rgb.py` 和 `envs/place_container_plate.py`：

- 可抓物体：`x=[-0.28, 0.28]`，`y=[-0.10, 0.05]`，并避开中线 `abs(x)<0.05`。
- `plate` 这类目标区：`x=[-0.08, 0.08]`，`y=[-0.15, -0.10]`。

`GapaScene.play_once()` 仍然存在，但网页 runtime 不依赖它生成 expert code。当前网页流程是在 runner 中直接执行 LLM 生成的 `play_once(api)` 程序。

`gapa/task_dsl.py`

定义结构化数据：

- `TaskDSL`：自然语言任务解析结果，比如 `object_name=cup`、`target_name=plate`、`relation=on`。
- `SkillStep` / `SkillPlan`：旧结构化计划路线的数据结构，当前网页 runtime 不再用它生成候选。
- `FailureReport`：失败阶段、失败原因和后续动作建议。

这些结构都支持 `to_dict()`，所以可以直接保存到 JSON。

`gapa/api_env.py` 和 `gapa/llm_client.py`

负责 LLM 配置和调用。当前强制只读：

```text
gapa/gapa_api.env
```

不会读根目录 `gapa_api.env`，也不会使用系统环境变量。`llm_client.py` 使用 OpenAI-compatible Chat Completions API。默认 provider 是 `deepseek`，但实际 provider/model/base_url/api_key 都以 `gapa/gapa_api.env` 为准。

`gapa/planner.py`

负责一件事：

1. 把自然语言解析成 `TaskDSL`。

当前 `GapaRunner` 初始化时使用 `TaskPlanner(use_llm=True)`，规则解析已经关闭。解析流程是：

1. 检查 LLM 是否已按 `gapa/gapa_api.env` 配置。
2. 对 LLM 输出做本地校验：对象是否支持、关系是否支持、当前场景里是否真的有这些物体。
3. 如果 LLM 未配置、API 调用失败、输出不是合法 JSON，接口直接报错。
4. 如果 LLM 输出的是不支持或当前场景缺失的对象，任务会被标记为不可行。
5. `task_dsl.json` 会记录 `parse_source` 和 `llm_attempted`，方便确认是否用了 LLM。

`gapa/program_codegen.py`

负责让 LLM 生成 3 个受限 `play_once(api)` 候选程序。Prompt 会包含：

- 自然语言任务
- 已校验的 `TaskDSL`
- 当前场景对象、roles、target relations 和 pose 摘要
- 允许调用的 `api.*` skill 函数签名
- 一个简短 `play_once(api)` 示例

LLM 必须返回 JSON：`{"programs": [{"program_id": "...", "source": "def play_once(api): ..."}]}`，数量必须是 3。

`gapa/program_safety.py`

负责 AST 安全检查。生成代码只允许一个 `play_once(api)` 函数、局部变量赋值、常量、以及 `api.<allowed_skill>(...)` 调用。禁止 import、文件/系统/网络调用、类、循环、条件分支、异常处理、未知函数和任意属性访问。

`gapa/program_api.py`

这是生成程序能访问的安全执行接口。它暴露：

- `api.pose(name)`
- `api.choose_arm(name)`
- `api.grasp(...)`
- `api.move_up(...)`
- `api.place_on(...)`
- `api.place_in(...)`
- `api.back_to_origin(...)`

这些方法内部再调用 `env.grasp_actor(...)`、`env.place_actor(...)`、`env.move_by_displacement(...)`、`env.back_to_origin(...)` 等 RoboTwin 高层函数。左右臂选择仍按被抓物体的 x 坐标决定：

```text
x < 0 -> left
x >= 0 -> right
```

对于 cup/bowl，会根据左右臂自动选择接触点。现在没有 oracle teleport 修正：如果生成程序调用的动作失败，或者最后 success check 失败，就记录失败。

`gapa/runner.py`

这是后端调度中心，也是单用户运行状态的持有者。它维护：

- 当前仿真环境 `current_env`
- 当前 scene seed
- 当前 scene description
- 当前用户选择的物体
- 当前 run id

主要流程：

1. `randomize_scene()`：关闭旧环境，创建新 `GapaScene`，保存四个初始相机预览。
2. `run_task()`：创建 `runs_gapa/{run_id}`，保存场景，解析任务，生成候选程序，验证候选，执行最佳程序，生成视频和 summary。
3. `_validate_program_candidates()`：用 seed `11/23/37` 创建验证环境，跑每个候选程序，按成功率打分。
4. `_execute_program_once()`：在当前网页场景上执行最佳程序一次；不做规则 fallback，也不做参数级 retry。
5. `_build_video()`：优先使用 RoboTwin collect-data 视频；失败时用关键帧拼一个 fallback 视频。

`gapa/perception.py`

当前感知还是 oracle。也就是说系统直接读仿真对象 pose，不做真实 VLM 图像定位。`VLMPerception` 只是预留接口，目前返回 `not_implemented`。

## 一次任务的完整链路

以 `put cup on plate` 为例：

1. 用户在网页勾选 `cup + plate`。
2. 点击“生成随机场景”。
3. `POST /api/scene/randomize` 调 `RUNNER.randomize_scene(...)`。
4. `GapaScene.load_actors()` 加载 cup 和 plate，并防重叠采样。
5. 后端保存四张初始图：left wrist、right wrist、head、world。
6. 用户输入 `put cup on plate` 并点击“执行任务”。
7. `POST /api/task/run` 调 `RUNNER.run_task(...)`。
8. Planner 调 LLM 解析成：

```json
{
  "object_name": "cup",
  "target_name": "plate",
  "relation": "on"
}
```

9. Planner 本地校验：cup 可抓、plate 可作为 `on` 目标、两者都在当前场景中。
10. Program generator 让 LLM 生成 3 个 `play_once(api)` 候选程序。
11. AST safety 校验每个候选程序，只允许安全 `api.*` 调用。
12. Runner 在 3 个验证 seed 上执行候选程序，写 `validation.json`。
13. Runner 选择最佳程序，在当前网页场景执行。
14. `SafeSkillAPI` 调 RoboTwin 高层 API 抓 cup、放到 plate 上。
15. `GapaScene.check_success()` 检查 cup 和 plate 的 XY 距离、高度关系。
16. Runner 写 `attempts.jsonl`、`summary.json`、截图和 `demo.mp4`。
17. 网页显示运行日志和演示视频。

## 运行产物

每次任务会生成：

```text
runs_gapa/{run_id}/
```

常见文件：

- `scene.json`：当前场景 seed、选择的物体和对象位姿。
- `task_dsl.json`：自然语言解析结果，以及是否尝试 LLM。
- `candidate_programs.json`：候选 `play_once(api)` 程序、metadata 和 safety 结果。
- `programs/candidate_*.py`：保存的候选程序源码。
- `validation.json`：候选程序在验证 seed 上的成功率和错误。
- `attempts.jsonl`：当前场景实际执行的每次尝试。
- `summary.json`：网页展示的汇总结果。
- `demo.mp4`：演示视频。
- `gapa/current/*.png`：关键帧截图。
- `trajectory/`：RoboTwin collect-data 轨迹和视频缓存。

判断一次任务是否用了 LLM，看：

```text
runs_gapa/{run_id}/task_dsl.json
```

重点字段：

```json
{
  "parse_source": "llm",
  "llm_attempted": true
}
```

如果 LLM 不能使用，接口会直接返回错误，通常不会生成完整的 run 产物。

## 当前支持和限制

当前支持：

- 单步 `put/place source in target`
- 单步 `put/place source on target`
- 中英文物体别名
- 用户手动选择场景物体
- 四相机初始图
- 演示视频
- LLM 解析任务
- LLM 生成受限 `play_once(api)` 候选程序

当前不支持：

- 多步任务，例如“一次把三个方块叠起来”。
- 让 LLM 生成不受限制的原生 Python expert code。
- VLM 图像定位。
- 真实世界机器人执行。
- 并发多用户场景。
- 自动根据任务重生成场景。

## 调试建议

确认网页服务是否用到了最新代码：重启 `uvicorn`。`RUNNER = GapaRunner()` 在服务启动时创建，如果服务没重启，旧 planner 配置会继续存在。

确认 LLM 是否启用：看最新 run 的 `task_dsl.json`：

```text
parse_source
llm_attempted
```

确认程序为什么失败：看 `candidate_programs.json`、`validation.json` 和 `attempts.jsonl`。前者保存生成代码和 safety 结果，后两者说明验证和当前网页场景实际执行到哪一步失败。

跑基础检查：

```bash
python -m py_compile gapa/*.py envs/gapa_scene.py
python -m unittest tests.test_gapa_planner
python -m unittest tests.test_gapa_program_codegen
```
