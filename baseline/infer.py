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

# ── v2: Few-shot 示例增强 prompt ──
FEWSHOT_PROMPT = f"""你是游戏买量广告视频的专业标注员。请观看关键帧，输出8个标注字段。

字段定义（必须严格从可选值中选择，不得自造标签）：

1. visual_source_type（画面素材，单选）: {json.dumps(ENUMS['visual_source_type'], ensure_ascii=False)}
2. has_real_person（真人出镜，单选）: ["有", "无"]
3. narrative_structure（开头5秒叙事，单选）: {json.dumps(ENUMS['narrative_structure'], ensure_ascii=False)}
   "爽感直给"=开局展示暴击/掉落/暴涨; "冲突引入"=展示失败/困境; "攻略建议"=教学口吻; "悬念提问"=疑问开场; "逆袭叙事"=先弱后强; "卡关反转"=失败→换策略→成功; "福利展示"=展示奖励; "其他"
4. selling_point（核心卖点，单选）: {json.dumps(ENUMS['selling_point'], ensure_ascii=False)}
   "爆装刺激"=满地爆装备; "低门槛变强"=轻松挂机变强; "世界观/IP代入"=IP情怀; "失败纠正"=纠正操作; "弱者逆袭"=弱者变强; "收集养成"=收集养成; "其他"
5. cta_type（结尾行动号召，单选）: {json.dumps(ENUMS['cta_type'], ensure_ascii=False)}
6. claim_type（利益承诺，多选≤{MULTI_LIMIT['claim_type']}）: {json.dumps(ENUMS['claim_type'], ensure_ascii=False)}
7. core_action（游戏动作，多选≤{MULTI_LIMIT['core_action']}）: {json.dumps(ENUMS['core_action'], ensure_ascii=False)}
8. growth_payoff（成长收益，多选≤{MULTI_LIMIT['growth_payoff']}）: {json.dumps(ENUMS['growth_payoff'], ensure_ascii=False)}

标注示例（供参考风格，不要照抄）：
示例视频：开头角色打怪爆满地装备，中间战力暴涨到9999，结尾"立即下载"按钮。
正确输出：{{"visual_source_type": "实机+包装", "has_real_person": "无", "narrative_structure": "爽感直给", "selling_point": "爆装刺激", "cta_type": "立即体验", "claim_type": ["高爆率"], "core_action": ["自动战斗/挂机", "Boss战"], "growth_payoff": ["战力大幅提升", "装备获得或升级"]}}

只输出JSON，不要其他文字："""


def build_multimodal_prompt(multi_text: str, base_prompt: str) -> str:
    """v3: 多模态文字直接拼入prompt头部"""
    if not multi_text:
        return base_prompt
    return multi_text + "\n\n" + base_prompt


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


# ── v3: ASR + OCR 多模态融合 ──
_whisper_model = None

OCR_PROMPT = """请仔细阅读这些视频关键帧中的所有可见文字，包括：
- 画面上的字幕、标题、弹窗文字
- 按钮文字（如"立即下载""登录领取"）
- 数值信息（如战斗力、等级、奖励数量）
- 任何其他可见的中文或英文文字

请按原文原样逐条列出，一行一条。如果没有可见文字，回复"无文字"。"""


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _vlm_ocr(frame_paths: list, model, processor, max_pixels: int = 200704) -> str:
    """用 Qwen2.5-VL 自身做 OCR，返回提取的文字"""
    from qwen_vl_utils import process_vision_info

    content = []
    for fp in frame_paths:
        content.append({"type": "image", "image": fp, "max_pixels": max_pixels})
    content.append({"type": "text", "text": OCR_PROMPT})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt", **video_kwargs).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    if "无文字" in result:
        return ""
    return result


def extract_multimodal_text(video_path: str, frame_paths: list, model=None, processor=None):
    """v3: 从视频提取音频文字(ASR) + 画面文字(OCR)，返回额外 prompt 文本"""
    info_parts = []
    import subprocess

    # ── ASR: 提取音频文字 ──
    audio_path = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", "-y", audio_path, "-loglevel", "error"],
            check=True, timeout=30)
        model = _get_whisper()
        result = model.transcribe(audio_path, language="zh", fp16=False)
        audio_text = result["text"].strip()
        if audio_text:
            info_parts.append(f"【语音内容】{audio_text}")
    except Exception:
        pass
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    # ── OCR: Qwen自身提取关键帧画面文字 ──
    if model is not None and processor is not None:
        ocr_frames = frame_paths[:2] + frame_paths[-2:] if len(frame_paths) > 4 else frame_paths
        try:
            ocr_result = _vlm_ocr(ocr_frames, model, processor)
            if ocr_result and "无文字" not in ocr_result:
                info_parts.append(f"【画面文字】{ocr_result}")
        except Exception:
            pass

    return "\n".join(info_parts) if info_parts else ""


# ── v4: 标签联动规则后处理 ──
def apply_label_rules(record: dict) -> dict:
    """v4: 根据业务逻辑修正不可能的组合"""
    # 录屏/动画CG → 几乎不可能有真人
    if record["visual_source_type"] in ("录屏", "动画/CG") and record["has_real_person"] == "有":
        record["has_real_person"] = "无"
    # 情景剧 → 一定有真人
    if record["visual_source_type"] == "情景剧" and record["has_real_person"] == "无":
        record["has_real_person"] = "有"
    # 有真人 → 不可能是录屏
    if record["has_real_person"] == "有" and record["visual_source_type"] == "录屏":
        record["visual_source_type"] = "情景剧"
    return record


NARRATIVE_PROMPT = """请专注分析这3张视频开头帧，判断叙事结构（单选）：
- "爽感直给": 开局直接展示暴击/掉落/暴涨/炫酷技能
- "冲突引入": 展示失败/困境/敌人强大
- "攻略建议": 教学/攻略口吻，教玩家怎么玩
- "悬念提问": 疑问句开场，制造悬念
- "逆袭叙事": 先展示弱→后变强
- "卡关反转": 失败→换策略→成功
- "福利展示": 展示登录奖励/礼包/福利
- "其他": 上述都不符合

只输出一个选项，不要其他文字："""


def _vlm_narrative(first_frames: list, model, processor) -> str:
    """v4: 用前3帧单独推理叙事结构"""
    from qwen_vl_utils import process_vision_info
    content = []
    for fp in first_frames:
        content.append({"type": "image", "image": fp, "max_pixels": 200704})
    content.append({"type": "text", "text": NARRATIVE_PROMPT})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt", **video_kwargs).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, temperature=0, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前N条(调试用)")
    ap.add_argument("--max_pixels", type=int, default=200704, help="每帧最大像素(默认448*448)")
    ap.add_argument("--fps", type=float, default=1.0, help="抽帧率(原始模式)")
    ap.add_argument("--use_scene_detect", action="store_true", help="v1: 用场景切换检测替换固定fps")
    ap.add_argument("--max_keyframes", type=int, default=12, help="scene detect 最大关键帧数")
    ap.add_argument("--use_cot", action="store_true", help="v2: 用Few-shot增强prompt")
    ap.add_argument("--max_tokens", type=int, default=512, help="最大生成token数")
    ap.add_argument("--use_multimodal", action="store_true", help="v3: ASR语音+OCR画面文字融合")
    args = ap.parse_args()

    videos = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".mp4"))
    if args.limit:
        videos = videos[: args.limit]
    print(f"待推理视频数: {len(videos)}  mode={'scene_detect' if args.use_scene_detect else 'fps'}"
          f"  prompt={'cot' if args.use_cot else 'baseline'}")

    prompt = FEWSHOT_PROMPT if args.use_cot else PROMPT
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

            # ── v3: 多模态文字提取 ──
            multi_text = ""
            if args.use_multimodal:
                try:
                    multi_text = extract_multimodal_text(video_path, frame_paths, model, processor)
                    if multi_text:
                        method += "+multimodal"
                except Exception:
                    pass

            content = []
            for fp in frame_paths:
                content.append({"type": "image", "image": fp,
                                "max_pixels": args.max_pixels})
            # 多模态文字拼在 prompt 前面（结构化路由）
            final_prompt = build_multimodal_prompt(multi_text, prompt)
            content.append({"type": "text", "text": final_prompt})
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

        # ── v4: narrative专项（前3帧单独推理，覆盖主推理的narrative）──
        if args.use_multimodal and args.use_scene_detect and len(frame_paths) >= 3:
            try:
                nar_text = _vlm_narrative(frame_paths[:3], model, processor)
                enum = ENUMS["narrative_structure"]
                matched = [v for v in enum if v in nar_text]
                if matched:
                    rec["narrative_structure"] = matched[0]
                    method += "+nar"
            except Exception:
                pass

        rec = apply_label_rules(rec)  # v4: 标签联动修正
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
