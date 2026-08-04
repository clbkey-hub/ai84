# -*- coding: utf-8 -*-
"""Baseline 推理脚本：Qwen2.5-VL-7B 对短视频做8字段语义标注
用法:
  python infer.py --video_dir <视频目录> --out <输出jsonl> [--limit N]
"""
import argparse, json, os, re, sys, time

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前N条(调试用)")
    ap.add_argument("--max_pixels", type=int, default=200704, help="每帧最大像素(默认448*448)")
    ap.add_argument("--fps", type=float, default=1.0, help="抽帧率")
    args = ap.parse_args()

    videos = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".mp4"))
    if args.limit:
        videos = videos[: args.limit]
    print(f"待推理视频数: {len(videos)}")

    print("加载模型...", MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()

    results, raw_log = [], []
    for i, vf in enumerate(videos):
        t0 = time.time()
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": os.path.join(args.video_dir, vf),
                 "max_pixels": args.max_pixels, "fps": args.fps},
                {"type": "text", "text": PROMPT},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt", **video_kwargs).to("cuda:0")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        out_ids = gen[:, inputs.input_ids.shape[1]:]
        out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        raw = extract_json(out_text)
        rec = normalize_record(vf, raw)
        results.append(rec)
        raw_log.append({"video_file": vf, "raw_output": out_text})
        print(f"[{i+1}/{len(videos)}] {vf} {time.time()-t0:.1f}s -> "
              f"{rec['visual_source_type']} | {rec['selling_point']}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out + ".raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_log, f, ensure_ascii=False, indent=1)
    print(f"完成! 结果: {args.out}  原始输出: {args.out}.raw.json")


if __name__ == "__main__":
    main()
