# 优化实验记录

> 追踪每次优化改动、实验结果和决策，方便回溯和逐步迭代。

---

## v0 — Baseline（初始版本）

| 项 | 值 |
|-----|-----|
| 日期 | 2026-07-30 |
| 提交 | `020784a` |
| 模型 | Qwen2.5-VL-7B-Instruct |
| 抽帧 | fps=1, max_pixels=200704 (448×448) |
| 推理 | do_sample=False, max_new_tokens=512 |
| Prompt | 单轮 8 字段联合输出 |
| 兜底 | 解析失败 → 训练集众数 |
| **5单选 Accuracy** | 0.52 |
| **3多选 Micro-F1** | — |
| **预估得分** | **36.31 / 70** |

### 文件清单
- `baseline/infer.py` — 推理主脚本
- `baseline/schema.py` — 标签体系 + 兜底
- `baseline/eval.py` — 评估脚本

---

## 优化计划

| # | 方向 | 预期提升 | 状态 |
|---|------|---------|------|
| 1 | 抽帧策略：scene detect 换关键帧 | +3~5 分 | ✅ 完成 |
| 2 | Prompt 拆分 + Few-shot | +3~5 分 | ⬜ |
| 3 | 推理参数调优（温度/分辨率） | +1~3 分 | ⬜ |
| 4 | ASR 语音 + OCR 文字融合 | +3~8 分 | ⬜ |
| 5 | 兜底策略增强 | +1~2 分 | ⬜ |
| 6 | LoRA 微调 | +5~10 分 | ⬜ |

---

## v1 — Scene Detect 关键帧（exp-v1-scene-detect）

| 项 | 值 |
|-----|-----|
| 日期 | 2026-08-04 |
| 分支 | `exp-v1-scene-detect` |
| 改动文件 | `baseline/infer.py` |
| 核心改动 | 用 scenedetect 按场景切换提取关键帧（ContentDetector threshold=5.0），替代固定 fps=1 均匀采样 |

### 改动详情

```python
# 新增 extract_keyframes() 函数
# - 使用 scenedetect.ContentDetector(threshold=5.0) 检测场景切换
# - 每场景取中间帧，最多 max_keyframes=12 帧
# - 场景不足 3 个时回退均匀采样
# - 新增 --use_scene_detect 和 --max_keyframes 参数
```

### 踩坑

- `VideoStreamCv2` 无 `close()` 方法 → 异常覆盖了成功检测结果，全部回退 uniform
- 默认 threshold=27.0 对游戏广告太迟钝，降到 5.0 才生效

### 对比结果（训练集前 10 条）

| 字段 | v0 Baseline | v1 Scene Detect | Δ |
|------|------------|-----------------|-----|
| visual_source_type | 0.8000 | 0.9000 | +0.10 |
| has_real_person | 1.0000 | 1.0000 | 0 |
| narrative_structure | 0.2000 | 0.2000 | 0 |
| selling_point | 0.4000 | 0.7000 | +0.30 |
| cta_type | 0.2000 | 0.4000 | +0.20 |
| claim_type | 0.4348 | 0.5217 | +0.09 |
| core_action | 0.6154 | 0.6286 | +0.01 |
| growth_payoff | 0.5000 | 0.2857 | -0.21 |
| **Overall Avg** | **0.5188** | **0.5795** | **+0.06** |
| **预估得分** | **36.31** | **40.57** | **+4.26** |

### 分析

- ✅ **selling_point (+0.30)** 和 **cta_type (+0.20)** 提升最大 — 关键帧更聚焦差异化画面，卖点和 CTA 判断更准
- ✅ **visual_source_type (+0.10)** 小幅提升 — 更多代表性帧帮助区分「实机+包装」vs「动画CG」
- ⚠️ **growth_payoff (-0.21)** 下降 — 12 帧上限可能截断了后期成长展示帧
- ⚠️ **narrative_structure 卡在 0.2** — 需要专门优化 Prompt（看开头 5 秒）
