# 多模态视频语义标注 Baseline

基于 **Qwen2.5-VL-7B-Instruct** 的游戏买量广告短视频自动标注系统，用于竞赛「初赛」阶段。

## 任务

输入一条游戏广告短视频（.mp4），输出 8 个语义标注字段：

| 字段 | 类型 | 标签数 | 含义 |
|------|------|--------|------|
| `visual_source_type` | 单选 | 6 | 画面素材来源（实机+包装 / 动画CG / 情景剧 / 录屏 / 直播切片 / 其他） |
| `has_real_person` | 单选 | 2 | 是否真人出镜 |
| `narrative_structure` | 单选 | 8 | 开头叙事手法（爽感直给 / 冲突引入 / 悬念提问 / …） |
| `selling_point` | 单选 | 7 | 核心卖点（爆装刺激 / 低门槛变强 / 世界观IP代入 / …） |
| `cta_type` | 单选 | 3 | 结尾行动号召（无明确CTA / 立即体验 / 试玩挑战） |
| `claim_type` | 多选≤2 | 4 | 利益承诺话术（高爆率 / 挂机变强 / 登录送 / …） |
| `core_action` | 多选≤3 | 9 | 主要游戏动作（Boss战 / 自动战斗挂机 / 副本挑战 / …） |
| `growth_payoff` | 多选≤3 | 7 | 成长收益展示（战力大幅提升 / 装备升级 / 外观进化 / …） |

## 项目结构

```
.
├── baseline/
│   ├── infer.py          # 推理脚本：视频 → 8字段标注
│   ├── eval.py           # 评估脚本：Accuracy + Micro-F1
│   └── schema.py         # 标签体系定义 + 输出规整/兜底
├── label_schema.json     # 所有字段枚举值
├── models/               # Qwen2.5-VL-7B 模型权重（不入仓库）
├── 初赛数据集/            # 训练/测试视频（不入仓库）
└── github_setup_guide.md # GitHub 连接配置指南
```

## 技术栈

- **模型**: Qwen2.5-VL-7B-Instruct（多模态视觉语言模型）
- **框架**: PyTorch + Transformers + qwen-vl-utils
- **GPU**: RTX 4090D 24GB

## 快速开始

```bash
# 环境
conda activate vl

# 推理（前10条自测）
python baseline/infer.py \
  --video_dir 初赛数据集/初赛数据集/训练集/videos \
  --out train_pred_10.jsonl \
  --limit 10

# 评估
python baseline/eval.py --pred train_pred_10.jsonl --gt gt_10.jsonl

# 全量推理
python baseline/infer.py \
  --video_dir 初赛数据集/初赛数据集/测试集/videos \
  --out submit.jsonl
```

## 评估指标

- 5 个单选字段：**Accuracy**
- 3 个多选字段：**Micro-F1**
- 综合得分 = 平均分 × 70（满分 70）

## Baseline 得分

| 指标 | 值 |
|------|-----|
| Avg Accuracy (5 fields) | 0.52 |
| 预估竞赛得分 | 36.31 / 70 |
