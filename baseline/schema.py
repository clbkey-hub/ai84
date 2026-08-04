# -*- coding: utf-8 -*-
"""标签体系定义与提交校验工具（来源：官方SUBMISSION_GUIDE + 训练集统计）"""

SINGLE_FIELDS = ["visual_source_type", "has_real_person", "narrative_structure",
                 "selling_point", "cta_type"]
MULTI_FIELDS = ["claim_type", "core_action", "growth_payoff"]
MULTI_LIMIT = {"claim_type": 2, "core_action": 3, "growth_payoff": 3}

# 训练集中实际出现的枚举值（label_schema.json 统计所得）
ENUMS = {
    "visual_source_type": ["实机+包装", "动画/CG", "情景剧/短剧", "录屏", "直播切片", "其他"],
    "has_real_person": ["有", "无"],
    "narrative_structure": ["爽感直给", "冲突引入", "攻略建议", "悬念提问", "逆袭叙事",
                            "卡关反转", "福利展示", "其他"],
    "selling_point": ["爆装刺激", "低门槛变强", "世界观/IP代入", "失败纠正", "弱者逆袭",
                      "收集养成", "其他"],
    "cta_type": ["无明确CTA", "立即体验", "试玩挑战", "立即下载", "引导下载"],
    "claim_type": ["无明确承诺", "高爆率", "挂机变强", "登录送"],
    "core_action": ["自动战斗/挂机", "攻略配置", "角色/装备展示", "换装提战", "Boss战",
                    "进阶挑战", "小游戏互动", "副本挑战", "其他"],
    "growth_payoff": ["装备获得或升级", "战力大幅提升", "里程碑突破", "外观进化",
                      "击败强敌", "资源暴增", "其他"],
}

# 各字段众数（训练集频率最高值），用于解析失败时的兜底
FALLBACK = {
    "visual_source_type": "实机+包装",
    "has_real_person": "无",
    "narrative_structure": "爽感直给",
    "selling_point": "爆装刺激",
    "cta_type": "无明确CTA",
    "claim_type": ["无明确承诺"],
    "core_action": ["自动战斗/挂机"],
    "growth_payoff": ["装备获得或升级"],
}


def _match_enum(value: str, field: str):
    """把模型输出的标签值规整到官方枚举；匹配不上返回 None"""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if v in ENUMS[field]:
        return v
    # 宽松匹配：去空格、包含关系
    v2 = v.replace(" ", "")
    for e in ENUMS[field]:
        if v2 == e.replace(" ", "") or v2 in e or e in v2:
            return e
    return None


def normalize_record(video_file: str, raw: dict) -> dict:
    """把模型原始输出规整成合法提交行（保证100%通过格式校验）"""
    rec = {"video_file": video_file}
    for f in SINGLE_FIELDS:
        m = _match_enum(raw.get(f, ""), f)
        rec[f] = m if m else FALLBACK[f]
    for f in MULTI_FIELDS:
        vals = raw.get(f, [])
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            vals = []
        out, seen = [], set()
        for v in vals:
            m = _match_enum(v, f)
            if m and m not in seen:
                out.append(m); seen.add(m)
        rec[f] = out[: MULTI_LIMIT[f]] if out else FALLBACK[f]
    return rec
