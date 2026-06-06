cat > ~/Desktop/RoboTwin-fork/gapa/VLM_LLM_FEEDBACK_README.md << 'EOF'
# GAPA VLM + LLM Feedback 闭环模块

## 概述

本模块实现了 GAPA 项目中的执行反馈与自纠错闭环：
任务执行 → 失败检测 → failure_report → LLM 重规划 → 重新执行 → 最多 3 次尝试
text


## 文件结构

| 文件 | 说明 |
|------|------|
| `gapa/feedback.py` | 核心数据结构 (`StageEvent`, `FailureReport`) 和反馈提供者 (`RuleBasedFeedbackProvider`, `FakeFeedbackProvider`) |
| `gapa/runner.py` | 主执行循环，feedback 闭环在 `run_task()` 中实现 |
| `gapa/program_codegen.py` | LLM 代码生成，支持 `failure_report` 和 `previous_program` 作为输入进行重规划 |

## 核心接口

### 1. `StageEvent` (gapa/feedback.py)

每个关键技能执行后产生的阶段事件。

```python
@dataclass
class StageEvent:
    run_id: str              # 运行 ID
    attempt_id: int          # 当前尝试次数
    program_id: str          # 程序 ID
    stage: str               # "grasp" | "lift" | "place" | "open_drawer" | "final_success"
    api_call: str            # "grasp_at" | "place_at" | ...
    object_name: str         # 操作对象名
    target_name: str | None  # 目标对象名
    relation: str | None     # 关系 ("in" | "on" | "row")
    arm: str                 # 机械臂 ("left" | "right")
    args: dict               # API 调用参数
    before: dict             # 执行前状态 {"object_pose": [...], "camera_snapshot": "..."}
    after: dict              # 执行后状态
    exception: str | None    # 异常信息

2. FailureReport (gapa/feedback.py)

反馈提供者生成的失败报告。
python

@dataclass
class FailureReport:
    status: str                    # "ok" | "failed"
    failed_stage: str | None       # 失败阶段
    failure_type: str | None       # 失败类型
    confidence: float              # 置信度 (0~1)
    evidence: list[str]            # 证据列表
    suggested_action: str          # "none" | "parameter_adjust" | "perception_reestimate" |
                                   # "strategy_switch" | "code_regeneration"
    llm_feedback: str | None       # 给 LLM 的反馈文本
    perception_requests: list      # 感知请求（预留）
    retry_policy: dict             # 重试策略 {"should_replan": bool, "max_extra_attempts": int}

3. FeedbackProvider (gapa/feedback.py)

反馈提供者基类。
python

class FeedbackProvider:
    def evaluate(self, event: StageEvent, context: dict) -> FailureReport:
        """判断阶段是否成功，失败时生成 FailureReport。"""
        raise NotImplementedError

4. ProgramCodeGenerator.generate_programs() (gapa/program_codegen.py)

支持重规划的方法签名。
python

def generate_programs(
    self,
    instruction: str,
    task: TaskDSL,
    scene_objects: dict,
    failure_report: FailureReport | None = None,  # 失败时传入
    previous_program: str | None = None,          # 上次的程序源码
) -> list[ProgramCandidate]:

执行流程
text

run_task(instruction)
  │
  ├─ 1. 任务解析 (TaskPlanner)
  ├─ 2. LLM 生成候选程序 (ProgramCodeGenerator)
  ├─ 3. 离线验证 (validate_program_candidates)
  ├─ 4. 选择最优程序
  │
  └─ 5. Feedback 闭环 (最多 3 次尝试):
       │
       ├─ 执行 play_once(api)
       ├─ 收集 StageEvent
       ├─ 判断成功/失败 (FeedbackProvider.evaluate)
       │
       ├─ 成功 → 保存 summary, 返回结果
       │
       └─ 失败:
            ├─ 生成 FailureReport
            ├─ 保存 failure_reports.jsonl
            ├─ 构建 replan_request
            ├─ LLM 重新生成 3 个候选程序
            ├─ 重新验证
            └─ 重新执行 (回到步骤 5)

运行产物

每次运行在 runs_gapa/<run_id>/ 下生成：
文件	说明
stage_events.jsonl	每行一个阶段事件列表
failure_reports.jsonl	每行一个失败报告列表
replan_requests.jsonl	每行一个重规划请求
replan_programs_attempt_N.json	第 N 次重规划生成的候选程序
attempts.jsonl	每次尝试的执行记录
summary.json	运行总结（含 total_attempts）
实际案例
案例 1：成功任务（无重试）
text

指令: put the green block on the red block
场景: red_block, green_block, blue_block
结果: success
total_attempts: 1

产物：

    stage_events.jsonl ✅

    failure_reports.jsonl ❌ (未触发，因任务成功)

案例 2：失败任务（3 次尝试）
text

指令: put the green block on the red block
场景: red_block, green_block, blue_block
结果: failed
total_attempts: 3

failure_reports.jsonl 内容：
json

[
  {
    "status": "failed",
    "failed_stage": "final_success",
    "failure_type": "program_exception",
    "confidence": 1.0,
    "evidence": ["Exception: <error detail>"],
    "suggested_action": "code_regeneration",
    "llm_feedback": "Program exception at final_success. Regenerate play_once.",
    "retry_policy": { "should_replan": true, "max_extra_attempts": 1 }
  }
]

replan_requests.jsonl 内容：
json

[
  {
    "task_dsl": { "object_name": "green_block", "target_name": "red_block", "relation": "on" },
    "scene_objects": { ... },
    "previous_program": "...",
    "failure_report": { ... },
    "required_output": { "program_count": 3, "function_name": "play_once" }
  }
]

已知限制

    failure_type 精度不足：当前 RuleBasedFeedbackProvider 在 final_success 阶段主要依赖 exception 字段，可能将 relation_not_satisfied 误报为 program_exception。需要与 gapa_scene.py 的 get_success_details() 更深集成。

    VLM 反馈未接入：当前使用基于规则的 RuleBasedFeedbackProvider，尚未接入视觉语言模型。后续应实现 VlmFeedbackProvider 使用相机图像判断阶段成功。

    StageEvent 信息有限：当前 gapa_scene.py 的 check_success() 未输出分阶段信息，导致 _collect_stage_events 只能生成一个汇总事件。需要在 program_api.py 的关键 API（grasp_at, place_at 等）中添加阶段事件钩子。

后续开发建议

    在 SafeSkillAPI 的 grasp_at, place_at, open_drawer 等方法中发射 StageEvent

    实现 VlmFeedbackProvider 替代 RuleBasedFeedbackProvider

    改进 failure_type 分类精度

    前端展示纠错过程（失败原因、重试记录）

协作者接入指南
如果你负责 VLM 定位

你的 pose provider 输出会被 SafeSkillAPI.pose() 使用。Feedback 模块会检查执行前后物体位置变化，你的 pose 质量直接影响 feedback 准确性。
如果你负责扩展任务/skill

在 SafeSkillAPI 中添加新 skill 时，请同时添加 StageEvent 发射逻辑，让 feedback 模块能检测该阶段是否成功。
如果你负责前端

summary.json 中包含 total_attempts、failure_reports.jsonl 中有详细失败信息，可用于展示纠错过程。
EOF

echo "README saved to gapa/VLM_LLM_FEEDBACK_README.md"
text


```bash
cat ~/Desktop/RoboTwin-fork/gapa/VLM_LLM_FEEDBACK_README.md | head -10