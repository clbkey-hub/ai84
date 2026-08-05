# 项目操作总结（截至 2026-08-04）

## 当前分支

```
main ──────────────────● baseline（36.31 分）
exp-v1-scene-detect ───● v1（40.57 分，+4.26）  ← 当前
```

---

## 已完成

### v0 Baseline（main 分支）
- 用 Qwen2.5-VL-7B-Instruct 对 36 条测试视频做 8 字段标注
- 抽帧：fps=1 均匀采样，max_pixels=448×448
- 提交文件：`baseline/submit.jsonl`
- 预估得分：**36.31 / 70**

### v1 Scene Detect（exp-v1-scene-detect 分支）
- 用 scenedetect 按场景切换提取关键帧（threshold=5.0，最多 12 帧）
- 其他不变
- 提交文件：`experiments/v1/submit_v1.jsonl`
- 预估得分：**40.57 / 70**（+4.26）

---

## 关键文件

```
/root/autodl-tmp/
├── baseline/
│   ├── infer.py           # 推理脚本（v1 新增 --use_scene_detect）
│   ├── schema.py          # 标签体系 + 兜底
│   ├── eval.py            # 评估脚本
│   └── submit.jsonl       # v0 提交文件
├── experiments/
│   ├── experiments.md     # 实验日志（每一步改动和结果）
│   └── v1/
│       ├── submit_v1.jsonl        # v1 提交文件（36行）
│       └── train_pred_10_v1.jsonl # v1 自测（10条）
├── github_setup_guide.md # GitHub SSH/Token 配置指南
├── README.md             # 项目说明
└── models/               # Qwen2.5-VL-7B（只读，不动）
```

---

## 常用命令

```bash
# 激活环境
conda activate vl

# 切换分支
git checkout exp-v1-scene-detect   # v1 实验
git checkout main                  # 回到 baseline

# v1 推理（scene detect）
python baseline/infer.py \
  --video_dir 初赛数据集/初赛数据集/测试集/videos \
  --out experiments/v1/submit_v1.jsonl \
  --use_scene_detect

# v1 自测（10条 + 评估）
python baseline/infer.py \
  --video_dir 初赛数据集/初赛数据集/训练集/videos \
  --out experiments/v1/train_pred_10_v1.jsonl \
  --limit 10 --use_scene_detect

python baseline/eval.py \
  --pred experiments/v1/train_pred_10_v1.jsonl \
  --gt baseline/gt_10.jsonl
```

---

## 赛事提交要求

- 文件名：`submit.jsonl`
- 36 行，对应 video_001~036.mp4
- 9 字段：video_file + 5 单选(str) + 3 多选(list[str])
- 任一格式错误直接 0 分

---

## 下一步（v2 计划）

1. **Prompt 拆分** — 8 字段一次输出改分轮提问（先看画面→再看开头→再看结尾）
2. **ASR/OCR** — 提取视频音频文字 + 画面文字，拼入 prompt
3. **推理参数** — 调 temperature、分辨率、重复采样投票

记录在 `experiments.md`
