bash

cat > /home/kk/Desktop/RoboTwin-fork/feedback_README.md << 'ENDOFFILE'
# GAPA VLM + LLM Feedback 闭环 —— 接口文档与开发报告

## 概述

本模块实现了 GAPA 项目中的执行反馈与自纠错闭环：

任务执行 → 分阶段事件收集 → 失败检测 → failure_report → LLM 重规划 → 重新执行 → 最多 3 次尝试
text


**核心交付**：VLM 定位开发者只需按照本文档的接口格式输出 pose 结果，feedback 模块会自动判断抓取/放置是否成功。

---

## 文件结构

| 文件 | 说明 |
|------|------|
| `gapa/feedback.py` | `StageEvent`, `FailureReport`, `FeedbackProvider`, `RuleBasedFeedbackProvider`, `FakeFeedbackProvider` |
| `gapa/runner.py` | 主执行循环，feedback 闭环在 `run_task()` 中（大致第 162 行起），`_collect_stage_events()` 方法（大致第 559 行） |
| `gapa/program_codegen.py` | LLM 代码生成，`generate_programs()` 支持 `failure_report` 和 `previous_program` 参数 |
| `gapa/program_api.py` | `SafeSkillAPI` 在 `grasp_at`/`_place_at`/`open_drawer` 中自动发射 `StageEvent` 到 `stage_events` 列表 |

---

## 核心接口（给 VLM 开发者）

### 1. StageEvent —— 执行阶段事件

每个 skill 执行后自动产生的事件，VLM 定位开发者**不需要主动创建**，但需要理解其结构。

```python
@dataclass
class StageEvent:
    run_id: str              # 运行 ID，如 "20260606_130305_846ca28b"
    attempt_id: int          # 第几次尝试，从 1 开始
    program_id: str          # 程序 ID，如 "candidate_1_explicit_steps"
    stage: str               # 阶段: "grasp" | "lift" | "place" | "open_drawer" | "final_success"
    api_call: str            # API 名: "grasp_at" | "place_on" | "open_drawer" 等
    object_name: str         # 操作对象名
    target_name: str | None  # 目标对象名
    relation: str | None     # 关系: "in" | "on" | "row"
    arm: str                 # 机械臂: "left" | "right"
    args: dict               # API 调用参数
    before: dict             # 执行前状态（预留，VLM 可填充 object_pose）
    after: dict              # 执行后状态（预留，VLM 可填充 object_pose）
    exception: str | None    # 异常信息

实际案例（来自成功执行）：
json

[
  {
    "stage": "grasp",
    "api_call": "grasp_at",
    "object_name": "green_block",
    "arm": "left",
    "args": { "pre_grasp_dis": 0.09, "grasp_dis": 0 },
    "before": {},
    "after": {},
    "exception": null
  },
  {
    "stage": "place",
    "api_call": "place_on",
    "object_name": "green_block",
    "target_name": "red_block",
    "relation": "on",
    "arm": "left",
    "args": { "pre_dis": 0.08, "dis": 0.02 },
    "before": {},
    "after": {},
    "exception": null
  }
]

2. FailureReport —— 失败报告

Feedback provider 输出的结构化失败报告。
python

@dataclass
class FailureReport:
    status: str                    # "ok" | "failed"
    failed_stage: str | None       # 失败阶段
    failure_type: str | None       # 失败类型（见下表）
    confidence: float              # 置信度 0~1
    evidence: list[str]            # 证据描述
    suggested_action: str          # 建议动作（见下表）
    llm_feedback: str | None       # 给 LLM 的自然语言反馈
    perception_requests: list      # 感知请求（VLM 开发者可填充）
    retry_policy: dict             # { "should_replan": true/false, "max_extra_attempts": int }

失败类型（failure_type）：
值	含义
object_not_found	物体未找到
low_confidence_pose	pose 置信度过低
object_not_grasped	抓取失败，物体未离开桌面
wrong_object_grasped	抓取了错误物体
object_slipped	物体滑落
missed_target	放置位置偏离目标
relation_not_satisfied	最终关系不满足（如不在目标上方）
drawer_not_opened	抽屉未打开
collision_or_stuck	碰撞或卡住
program_exception	程序异常

建议动作（suggested_action）：
值	含义
none	无需操作
parameter_adjust	调整参数（如 pre_grasp_dis）
perception_reestimate	重新 VLM 感知
strategy_switch	换机械臂或策略
code_regeneration	重新生成完整程序

实际案例（失败时）：
json

{
  "status": "failed",
  "failed_stage": "final_success",
  "failure_type": "program_exception",
  "confidence": 1.0,
  "evidence": ["Exception: Task check failed"],
  "suggested_action": "code_regeneration",
  "llm_feedback": "Program exception at final_success. Regenerate play_once.",
  "retry_policy": { "should_replan": true, "max_extra_attempts": 1 }
}

3. FeedbackProvider —— 反馈提供者基类
python

class FeedbackProvider:
    def evaluate(self, event: StageEvent, context: dict) -> FailureReport:
        """判断阶段是否成功，失败时生成 FailureReport。"""
        raise NotImplementedError

context 字典结构：
python

{
    "natural_language_task": "put green block on red block",
    "task_dsl": { "object_name": "green_block", "target_name": "red_block", "relation": "on" },
    "scene_objects": { "green_block": {...}, "red_block": {...} },
    "program_source": "def play_once(api): ...",
    "attempt_history": [...],
    "allowed_api": ["pose", "grasp_at", "open_drawer", "place_in_drawer", "place_at"]
}

VLM 开发者接入指南
你需要做的事

    你的 VlmPoseProvider 输出的 pose 会自动被 SafeSkillAPI.pose() 使用。

    在 RuleBasedFeedbackProvider 中，before.object_pose 和 after.object_pose 字段目前为空 {}。

        如果你在 StageEvent 的 before/after 中填充了 object_pose（世界坐标 7 维数组），feedback 模块就能做精确的抓取/放置判断。

    如果需要 VLM 直接参与反馈判断，实现 VlmFeedbackProvider(FeedbackProvider)，在其中调用 VLM 分析相机图像。

你不需要做的事

    不需要修改 program_codegen.py

    不需要修改 runner.py 的闭环逻辑

    不需要处理 LLM 重规划

StageEvent 数据流
text

SafeSkillAPI.grasp_at() / _place_at() / open_drawer()
  → 执行后自动 appends StageEvent 到 api.stage_events
    → execute_program_candidate() 结束时写入 env.gapa_last_success_details["stages"]
      → runner._collect_stage_events() 读取
        → FeedbackProvider.evaluate() 判断成功/失败

完整执行流程
text

run_task(instruction)
  │
  ├─ 1. 任务解析 (TaskPlanner)
  ├─ 2. LLM 生成 3 个候选程序 (ProgramCodeGenerator)
  ├─ 3. 离线验证 (validate_program_candidates)
  ├─ 4. 选择最优程序
  │
  └─ 5. Feedback 闭环 (最多 3 次尝试):
       │
       ├─ 执行 play_once(api)
       │   ├─ grasp_at() → 发射 StageEvent(stage="grasp")
       │   ├─ move_above_pose()
       │   ├─ place_at() → 发射 StageEvent(stage="place")
       │   └─ check_success() → success_check 包含 stages
       │
       ├─ _collect_stage_events() 收集
       ├─ FeedbackProvider.evaluate() 判断
       │
       ├─ 成功 → 保存 summary, 返回结果
       │
       └─ 失败:
            ├─ 生成 FailureReport → failure_reports.jsonl
            ├─ 构建 replan_request → replan_requests.jsonl
            ├─ LLM 重新生成 3 个候选程序
            ├─ 重新验证
            └─ 重新执行 (回到步骤 5)

运行产物

每次运行在 runs_gapa/<run_id>/ 下生成：
文件	说明
stage_events.jsonl	每行一个阶段事件列表
failure_reports.jsonl	每行一个失败报告列表（仅失败时）
replan_requests.jsonl	每行一个重规划请求（仅失败时）
replan_programs_attempt_N.json	第 N 次重规划生成的候选程序（仅失败时）
attempts.jsonl	每次尝试的执行记录
summary.json	运行总结（含 total_attempts, success_check, video 路径）
实际案例
案例 1：成功任务（put green block on red block）
text

Run: 20260606_130305_846ca28b
Status: success
Total attempts: 1

success_check（含分阶段事件）：
json

{
  "success": true,
  "mode": "on_generic",
  "object_name": "green_block",
  "target_name": "red_block",
  "xy_distance": 0.00052,
  "xy_limit": 0.12,
  "xy_ok": true,
  "height_ok": true,
  "stages": [
    {
      "stage": "grasp",
      "api_call": "grasp_at",
      "object_name": "green_block",
      "arm": "left",
      "args": { "pre_grasp_dis": 0.09, "grasp_dis": 0 }
    },
    {
      "stage": "place",
      "api_call": "place_on",
      "object_name": "green_block",
      "target_name": "red_block",
      "relation": "on",
      "arm": "left",
      "args": { "pre_dis": 0.08, "dis": 0.02 }
    }
  ]
}

产物：stage_events.jsonl ✅, summary.json ✅, demo.mp4 ✅
案例 2：失败任务（put green block on red block，不同场景）
text

Run: 20260606_113038_4e210b17
Status: failed
Total attempts: 3

failure_reports.jsonl（3 次）：
json

[
  {
    "status": "failed",
    "failed_stage": "final_success",
    "failure_type": "program_exception",
    "confidence": 1.0,
    "suggested_action": "code_regeneration",
    "llm_feedback": "Program exception at final_success. Regenerate play_once.",
    "retry_policy": { "should_replan": true, "max_extra_attempts": 1 }
  }
]

产物：

    stage_events.jsonl ✅

    failure_reports.jsonl ✅（3 行）

    replan_requests.jsonl ✅（2 行）

    replan_programs_attempt_2.json ✅

    replan_programs_attempt_3.json ✅

    demo.mp4 ✅

已知限制与改进方向

    before/after 未填充真实 pose：当前 StageEvent.before 和 StageEvent.after 为空 {}。VLM 开发者可在这些字段中填入世界坐标的 object_pose，使 RuleBasedFeedbackProvider 能做精确的抓取高度/放置距离判断。

    VLM 反馈未接入：当前使用 RuleBasedFeedbackProvider（基于规则），尚未实现 VlmFeedbackProvider（基于视觉语言模型分析相机图像）。这是 VLM 开发者的后续工作。

    failure_type 精度：当前 final_success 阶段的 failure_type 主要是 program_exception，与 gapa_scene.py 的 get_success_details() 更深集成后可细化为 relation_not_satisfied 等。

单元测试

测试文件位置：tests/test_gapa_feedback.py

建议测试用例：

    FakeFeedbackProvider 在 exception 时返回 failed

    FakeFeedbackProvider 在正常时返回 ok

    RuleBasedFeedbackProvider 检测 z 轴变化判断抓取

    StageEvent.to_dict() 和 FailureReport.to_dict() 序列化正确

    模拟失败→replan→重试流程不抛异常

总结
需求	完成状态
StageEvent / FailureReport 数据结构	✅
FeedbackProvider / RuleBasedFeedbackProvider	✅
program_api.py 分阶段事件发射	✅
runner.py feedback 闭环（失败→重规划→重试）	✅
program_codegen.py 支持 failure_report	✅
运行产物（stage_events, failure_reports, replan）	✅
成功时不触发重试	✅
失败时最多 3 次尝试	✅
不破坏一次性执行流程	✅
VLM FeedbackProvider	❌（VLM 开发者后续工作）
单元测试	❌（待添加）
ENDOFFILE	