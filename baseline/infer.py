# -*- coding: utf-8 -*-
"""Baseline 推理脚本：Qwen2.5-VL-7B 对短视频做8字段语义标注

用法:
  # 原始模式（固定fps抽帧）
  python infer.py --video_dir <视频目录> --out <输出jsonl> [--limit N]

  # v1: Scene detect 模式（按场景切换提取关键帧）
  python infer.py --video_dir <视频目录> --out <输出jsonl> --use_scene_detect
"""
import argparse, json, os, re, sys, time, tempfile

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import ENUMS, MULTI_LIMIT, SINGLE_FIELDS, MULTI_FIELDS, normalize_record

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"

PROMPT = f"""你是游戏买量广告视频的专业标注员。请观看这条游戏广告短视频（含画面与字幕），从画面内容、文字信息、叙事方式等角度分析，输出8个标注字段。

各字段定义与可选值（必须严格从可选值中选择，不得自造标签）：

1. visual_source_type（画面素材来源，单选）: {json.dumps(ENUMS['visual_source_type'], ensure_ascii=False)}
   - "实机+包装"=游戏实机画面加特效包装; "录屏"=纯游戏录屏; "情景剧/短剧"=真人表演剧情
2. has_real_person（是否真人出镜，单选）: ["有", "无"]
3. narrative_structure（视频开头叙事手法，单选，重点看前5秒）: {json.dumps(ENUMS['narrative_structure'], ensure_ascii=False)}
   - "爽感直给"=开局直接展示爽点; "冲突引入"=以矛盾冲突开场; "攻略建议"=以攻略教学口吻开场; "悬念提问"=以疑问句开场
4. selling_point（整条素材核心卖点，单选）: {json.dumps(ENUMS['selling_point'], ensure_ascii=False)}
   - "爆装刺激"=强调打怪爆装备; "低门槛变强"=强调轻松挂机变强; "世界观/IP代入"=强调IP情怀世界观
5. cta_type（结尾行动号召，单选，重点看最后5秒）: {json.dumps(ENUMS['cta_type'], ensure_ascii=False)}
6. claim_type（利益承诺话术，多选0-{MULTI_LIMIT['claim_type']}个）: {json.dumps(ENUMS['claim_type'], ensure_ascii=False)}
   - "高爆率"=承诺装备爆率高; "挂机变强"=承诺挂机就能变强; "登录送"=承诺登录送福利
7. core_action（主要游戏动作展示，多选1-{MULTI_LIMIT['core_action']}个）: {json.dumps(ENUMS['core_action'], ensure_ascii=False)}
8. growth_payoff（成长收益展示，多选1-{MULTI_LIMIT['growth_payoff']}个）: {json.dumps(ENUMS['growth_payoff'], ensure_ascii=False)}
   - "战力大幅提升"=战力数值大涨; "里程碑突破"=等级/境界突破; "外观进化"=角色外观变化

只输出一个JSON对象，不要输出任何其他文字。格式：
{{"visual_source_type": "...", "has_real_person": "...", "narrative_structure": "...", "selling_point": "...", "cta_type": "...", "claim_type": [...], "core_action": [...], "growth_payoff": [...]}}"""

# ── v2: Chain-of-Thought 逐步推理 prompt ──
COT_PROMPT = f"""你是游戏买量广告视频的专业标注员。关键帧按时间顺序排列：前几帧=视频开头，后几帧=视频结尾。

请按以下步骤逐步分析，每步用1-2句话描述观察，最后输出JSON：

1.【画面类型】看全部关键帧，判断素材来源（{json.dumps(ENUMS['visual_source_type'], ensure_ascii=False)}）和真人出镜（有/无）。

2.【开头叙事】重点看前几帧（开头约5秒）：视频如何吸引人？
   - 爽感直给：开局直接展示暴击/装备掉落/战力暴涨画面
   - 冲突引入：展示失败/卡关/陷入困境
   - 攻略建议：教学口吻"教你如何..."
   - 悬念提问：疑问句开场"你知道..."
   - 逆袭叙事：先弱后强
   - 卡关反转：打不过→换策略→成功
   - 福利展示：展示福利奖励
   - 其他

3.【核心卖点】整条视频最吸引玩家的点（{json.dumps(ENUMS['selling_point'], ensure_ascii=False)}），必须选1个。

4.【游戏动作】展示了哪些玩法（多选≤{MULTI_LIMIT['core_action']}）：{json.dumps(ENUMS['core_action'], ensure_ascii=False)}

5.【成长收益】展示了什么成果（多选≤{MULTI_LIMIT['growth_payoff']}）：{json.dumps(ENUMS['growth_payoff'], ensure_ascii=False)}

6.【结尾CTA】看后几帧（结尾约5秒）：行动号召（{json.dumps(ENUMS['cta_type'], ensure_ascii=False)}）

7.【利益承诺】承诺了什么（多选≤{MULTI_LIMIT['claim_type']}）：{json.dumps(ENUMS['claim_type'], ensure_ascii=False)}

示例参考：
视频开头角色打怪爆满地装备，中间角色升级战力涨到9999，结尾出现"立即下载"。
→ 开头展示爆装场景=爽感直给，卖点=爆装刺激，动作=自动战斗+Boss战，收益=战力大幅提升+装备获得，CTA=立即体验，承诺=高爆率
→ {{"visual_source_type": "实机+包装", "has_real_person": "无", "narrative_structure": "爽感直给", "selling_point": "爆装刺激", "cta_type": "立即体验", "claim_type": ["高爆率"], "core_action": ["自动战斗/挂机", "Boss战"], "growth_payoff": ["战力大幅提升", "装备获得或升级"]}}

现在分析并输出JSON："""


def extract_json(text: str) -> dict:
    """从模型输出中提取JSON对象"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:  # 常见修复：单引号
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return {}


def extract_keyframes(video_path: str, max_frames: int = 12, fps: float = 1.0):
    """Scene detect: 按场景切换检测关键帧，场景过少时回退到均匀采样

    返回: (frame_paths, temp_dir) — frame_paths 为图像路径列表，temp_dir 用完需清理
    """
    import cv2
    from scenedetect import detect, ContentDetector, open_video

    tmpdir = tempfile.mkdtemp(prefix="keyframes_")
    frames = []

    try:
        video = open_video(video_path)
        scene_list = detect(video_path, ContentDetector(threshold=5.0))
    except Exception:
        scene_list = []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # 场景数足够 → 取每场景中间帧；场景太少 → 回退均匀采样
    if scene_list and len(scene_list) >= 3:
        for i, (start, end) in enumerate(scene_list):
            if i >= max_frames:
                break
            mid = int((start.frame_num + end.frame_num) / 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            if ret:
                path = os.path.join(tmpdir, f"scene_{i:03d}.jpg")
                cv2.imwrite(path, frame)
                frames.append(path)
        method = f"scene_detect({len(frames)}帧)"
    else:
        # 回退：按 fps 均匀采样
        interval = int(video_fps / fps) if fps > 0 else int(video_fps)
        interval = max(interval, 1)
        count = 0
        for fn in range(0, total_frames, interval):
            if count >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if ret:
                path = os.path.join(tmpdir, f"uniform_{count:03d}.jpg")
                cv2.imwrite(path, frame)
                frames.append(path)
                count += 1
        method = f"uniform({len(frames)}帧)"

    cap.release()
    return frames, tmpdir, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前N条(调试用)")
    ap.add_argument("--max_pixels", type=int, default=200704, help="每帧最大像素(默认448*448)")
    ap.add_argument("--fps", type=float, default=1.0, help="抽帧率(原始模式)")
    ap.add_argument("--use_scene_detect", action="store_true", help="v1: 用场景切换检测替换固定fps")
    ap.add_argument("--max_keyframes", type=int, default=12, help="scene detect 最大关键帧数")
    ap.add_argument("--use_cot", action="store_true", help="v2: 用Chain-of-Thought逐步推理prompt")
    ap.add_argument("--max_tokens", type=int, default=512, help="最大生成token数(CoT需>512)")
    args = ap.parse_args()

    videos = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".mp4"))
    if args.limit:
        videos = videos[: args.limit]
    print(f"待推理视频数: {len(videos)}  mode={'scene_detect' if args.use_scene_detect else 'fps'}"
          f"  prompt={'cot' if args.use_cot else 'baseline'}")

    prompt = COT_PROMPT if args.use_cot else PROMPT
    max_tokens = args.max_tokens if args.use_cot else 512

    print("加载模型...", MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()

    results, raw_log = [], []
    for i, vf in enumerate(videos):
        t0 = time.time()
        video_path = os.path.join(args.video_dir, vf)

        if args.use_scene_detect:
            # ── v1: Scene detect 关键帧 ──
            frame_paths, tmpdir, method = extract_keyframes(
                video_path, max_frames=args.max_keyframes, fps=args.fps)
            content = []
            for fp in frame_paths:
                content.append({"type": "image", "image": fp,
                                "max_pixels": args.max_pixels})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt", **video_kwargs).to("cuda:0")
        else:
            # ── baseline: 原始 fps 视频模式 ──
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path,
                     "max_pixels": args.max_pixels, "fps": args.fps},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt", **video_kwargs).to("cuda:0")
            method = f"fps{args.fps}"

        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        out_ids = gen[:, inputs.input_ids.shape[1]:]
        out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        raw = extract_json(out_text)
        rec = normalize_record(vf, raw)
        results.append(rec)
        raw_log.append({"video_file": vf, "raw_output": out_text, "method": method})
        print(f"[{i+1}/{len(videos)}] {vf} {time.time()-t0:.1f}s [{method}] -> "
              f"{rec['visual_source_type']} | {rec['selling_point']}")

        # 清理临时帧文件
        if args.use_scene_detect:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out + ".raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_log, f, ensure_ascii=False, indent=1)
    print(f"完成! 结果: {args.out}  原始输出: {args.out}.raw.json")


if __name__ == "__main__":
    main()
